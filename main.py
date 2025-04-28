"""
Joint Diffusion and Autoregressive Transformer (DART)

This module implements a unified model architecture that combines diffusion and 
autoregressive approaches for text generation. The model leverages DiT (Diffusion Transformer)
blocks alongside traditional autoregressive transformer components to create a hybrid
generation process.

Key features:
- Combines diffusion and autoregressive generation paradigms
- Uses DiT blocks for capturing global dependencies
- Implements noise scheduling for controlled diffusion
- Production-ready with full type annotations and logging
- Configurable architecture parameters

Example usage:
    # Initialize configuration
    config = DARTConfig(
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        diffusion_steps=1000,
        ar_weight=0.5,
    )
    
    # Initialize model
    model = DART(config)
    
    # Training example
    input_ids = torch.randint(0, config.vocab_size, (4, 128))
    loss_dict = model.compute_loss(input_ids)
    loss = loss_dict["loss"]
    loss.backward()
    
    # Generation example
    generated = model.generate(
        input_ids=torch.tensor([[0, 1, 2, 3]]),
        max_length=128,
        temperature=0.8,
        do_sample=True,
    )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
import numpy as np
from loguru import logger
import time


@dataclass
class DARTConfig:
    """Configuration class for Joint Diffusion Autoregressive Transformer model."""
    vocab_size: int = 50257  # Default GPT-2 vocabulary size
    max_seq_length: int = 1024
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    diffusion_steps: int = 1000
    diffusion_schedule: str = "cosine"  # Options: linear, cosine, quadratic
    ar_weight: float = 0.5  # Weight between autoregressive and diffusion outputs
    use_cache: bool = True
    gradient_checkpointing: bool = False


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for fixed positions."""
    
    def __init__(self, dim, max_seq_length: int = 1024):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        pe = torch.zeros(max_seq_length, dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return self.pe[:, :x.size(1)]


class DiffusionEmbedding(nn.Module):
    """Diffusion step embeddings used to condition the model on noise level."""
    
    def __init__(self, dim: int, max_steps: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps
        self.embedding = nn.Embedding(max_steps, dim)
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of diffusion timesteps, shape [batch_size]
        
        Returns:
            Embeddings for each timestep, shape [batch_size, dim]
        """
        return self.embedding(timesteps)



class DiTSelfAttention(nn.Module):
    """Self-attention mechanism modified for Diffusion Transformer blocks."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_prob: float = 0.1,
        is_causal: bool = False,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by the number of attention heads {num_attention_heads}."
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.is_causal = is_causal
        
        # Layers
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch_size, seq_len] or [batch_size, 1, seq_len]
                or [batch_size, 1, 1, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_length = hidden_states.size()[:2]
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure proper broadcasting shape [batch_size, 1, 1, seq_length]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
                
            # Convert from [0, 1] to [0, -10000] for proper masking
            if attention_mask.dtype != torch.float32:
                attention_mask = (1.0 - attention_mask.float()) * -10000.0
                
            attention_scores = attention_scores + attention_mask
            
        # Apply causal mask if needed
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=hidden_states.device) * -10000.0, 
                diagonal=1
            )
            attention_scores = attention_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Normalize scores and apply dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute context vectors and reshape
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        
        # Project back to hidden size
        output = self.output(context_layer)
        
        return output


class DiTFeedForward(nn.Module):
    """Feed-forward network for DiT blocks."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_prob: float = 0.1,
        activation: Callable = F.gelu,
    ):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = activation
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class DiTBlock(nn.Module):
    """
    Diffusion Transformer (DiT) block that combines attention with diffusion step conditioning.
    """
    
    def __init__(
        self,
        config: DARTConfig,
        is_causal: bool = False,
    ):
        super().__init__()
        self.config = config
        
        # Normalization layers (using pre-norm design)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ff_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Attention and feed-forward layers
        self.attention = DiTSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.attention_probs_dropout_prob,
            is_causal=is_causal,
        )
        self.feed_forward = DiTFeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout_prob=config.hidden_dropout_prob,
        )
        
        # Diffusion step conditioning
        self.diffusion_proj = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, 2 * config.hidden_size),
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        diffusion_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            diffusion_emb: Tensor of shape [batch_size, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Project diffusion embedding to gain and shift for adaptive layer norm
        diffusion_proj = self.diffusion_proj(diffusion_emb).unsqueeze(1)
        scale, shift = diffusion_proj.chunk(2, dim=-1)
        
        # Self-attention block with residual connection
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        # Apply diffusion conditioning
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Feed-forward block with residual connection
        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        # Apply diffusion conditioning
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class AutoregressiveBlock(nn.Module):
    """Standard autoregressive transformer block with causal attention."""
    
    def __init__(self, config: DARTConfig):
        super().__init__()
        self.config = config
        
        # Pre-norm design
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ff_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Causal self-attention
        self.attention = DiTSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.attention_probs_dropout_prob,
            is_causal=True,  # Always causal for autoregressive generation
        )
        
        # Feed-forward network
        self.feed_forward = DiTFeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout_prob=config.hidden_dropout_prob,
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Self-attention block with residual connection
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Feed-forward block with residual connection
        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class DiffusionScheduler:
    """
    Noise scheduler for the diffusion process.
    Handles variance scheduling for the forward and reverse processes.
    """
    
    def __init__(
        self, 
        num_diffusion_steps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """
        Args:
            num_diffusion_steps: Total number of diffusion steps
            schedule_type: Type of beta schedule (linear, cosine, or quadratic)
            beta_start: Starting value for linear schedule
            beta_end: Ending value for linear schedule
        """
        self.num_diffusion_steps = num_diffusion_steps
        self.schedule_type = schedule_type
        
        # Create beta schedule
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        elif schedule_type == "cosine":
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = num_diffusion_steps + 1
            x = torch.linspace(0, num_diffusion_steps, steps)
            alphas_cumprod = torch.cos(((x / num_diffusion_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        elif schedule_type == "quadratic":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_steps) ** 2
        else:
            raise ValueError(f"Unknown schedule type {schedule_type}")
        
        # Precompute diffusion parameters
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(
        self, 
        x_start: torch.Tensor, 
        x_noise: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0) = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        
        Args:
            x_start: Starting clean data [B, L, D]
            x_noise: Noise to add [B, L, D]
            timesteps: Timesteps to use for each batch element [B]
            
        Returns:
            Noisy version of x_start at specified timesteps
        """
        sqrt_alphas_cumprod_t = self._extract_at_t(self.sqrt_alphas_cumprod, timesteps)
        sqrt_one_minus_alphas_cumprod_t = self._extract_at_t(self.sqrt_one_minus_alphas_cumprod, timesteps)
        
        # Expand dimensions for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)
        
        # Add noise according to forward process
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * x_noise
    
    def _extract_at_t(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Extract values from a tensor at specified timesteps.
        
        Args:
            x: Tensor to extract from [T, ...]
            t: Timesteps to extract at [B]
            
        Returns:
            Extracted values [B, ...]
        """
        device = t.device
        out = torch.gather(x.to(device), 0, t)
        return out
    
    def get_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get variance at specified timesteps."""
        return self._extract_at_t(self.posterior_variance, timesteps)
    
    def get_log_variance(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get log variance at specified timesteps."""
        return self._extract_at_t(self.posterior_log_variance_clipped, timesteps)
    
    def q_posterior_mean(
        self, 
        x_start: torch.Tensor, 
        x_t: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean of q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start: Predicted clean data [B, L, D]
            x_t: Noisy data [B, L, D]
            timesteps: Current timesteps [B]
            
        Returns:
            Mean of the posterior distribution
        """
        coef1 = self._extract_at_t(self.posterior_mean_coef1, timesteps).view(-1, 1, 1)
        coef2 = self._extract_at_t(self.posterior_mean_coef2, timesteps).view(-1, 1, 1)
        return coef1 * x_start + coef2 * x_t


class DART(nn.Module):
    """
    Joint Diffusion and Autoregressive Transformer (DART) for text generation.
    
    This model combines diffusion-based and autoregressive approaches by:
    1. Using diffusion to model global dependencies and handle uncertainty
    2. Using autoregressive components for local coherence and fast sampling
    3. Fusing both approaches with a unified architecture
    
    Attributes:
        config (DARTConfig): Model configuration
        diffusion_scheduler (DiffusionScheduler): Handles noise scheduling
        token_embedding (nn.Embedding): Embeds input tokens
        position_embedding (SinusoidalPositionalEmbedding): Adds positional information
        diffusion_embedding (DiffusionEmbedding): Embeds diffusion timesteps
        embedding_dropout (nn.Dropout): Dropout for embeddings
        dit_layers (nn.ModuleList): Diffusion transformer layers
        ar_layers (nn.ModuleList): Autoregressive transformer layers
        diffusion_head (nn.Linear): Output projection for diffusion path
        ar_head (nn.Linear): Output projection for autoregressive path
    """
    
    def __init__(self, config: DARTConfig):
        super().__init__()
        self.config = config
        
        # Initialize diffusion scheduler
        self.diffusion_scheduler = DiffusionScheduler(
            num_diffusion_steps=config.diffusion_steps,
            schedule_type=config.diffusion_schedule,
        )
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = SinusoidalPositionalEmbedding(config.hidden_size, config.max_seq_length)
        self.diffusion_embedding = DiffusionEmbedding(config.hidden_size, config.diffusion_steps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Layer initialization
        self.dit_layers = nn.ModuleList([
            DiTBlock(config, is_causal=False) 
            for _ in range(config.num_hidden_layers // 2)
        ])
        
        self.ar_layers = nn.ModuleList([
            AutoregressiveBlock(config) 
            for _ in range(config.num_hidden_layers // 2)
        ])
        
        # Output heads
        self.diffusion_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.ar_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Layer norm for final outputs
        self.dit_final_norm = nn.LayerNorm(config.hidden_size)
        self.ar_final_norm = nn.LayerNorm(config.hidden_size)
        
        # Initialize weights
        self._init_weights()
        
        # Track whether we're in training or evaluation mode
        self.is_training = True
        
        logger.info(f"Initialized DART model with {config.num_hidden_layers} layers")
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=self.config.initializer_range)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def get_input_embeddings(self):
        """Get token embedding module."""
        return self.token_embedding
    
    def set_input_embeddings(self, embeddings):
        """Set token embedding module."""
        self.token_embedding = embeddings
        
    def train(self, mode: bool = True):
        """Set training mode."""
        self.is_training = mode
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode."""
        self.is_training = False
        return super().eval()
    
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input tokens and add positional information.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            Token embeddings [batch_size, seq_len, hidden_size]
        """
        # Get token embeddings
        inputs_embeds = self.token_embedding(input_ids)
        
        # Add positional embeddings
        position_embeds = self.position_embedding(inputs_embeds)
        embeddings = inputs_embeds + position_embeds
        
        # Apply dropout
        embeddings = self.embedding_dropout(embeddings)
        
        return embeddings
    
    def prepare_attention_mask(
        self, 
        input_ids: torch.Tensor,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Create attention mask for padding tokens.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            pad_token_id: ID of padding token
            
        Returns:
            Attention mask [batch_size, 1, 1, seq_len]
        """
        # Create mask: 0 for tokens to attend to, -10000 for tokens to ignore
        attention_mask = (input_ids != pad_token_id).float()
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Reshape for attention computation
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        return attention_mask
    
    def add_noise_to_embeddings(
        self,
        embeddings: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to token embeddings for the diffusion process.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_len, hidden_size]
            timesteps: Diffusion timesteps [batch_size]
            
        Returns:
            Tuple of (noisy_embeddings, noise)
        """
        # Generate random noise
        noise = torch.randn_like(embeddings)
        
        # Add noise according to diffusion schedule
        noisy_embeddings = self.diffusion_scheduler.add_noise(
            x_start=embeddings,
            x_noise=noise,
            timesteps=timesteps
        )
        
        return noisy_embeddings, noise
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model for training.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            timesteps: Diffusion timesteps for each batch element [batch_size]
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Dictionary with model outputs
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Default timesteps if not provided
        if timesteps is None:
            if self.is_training:
                # Sample random timesteps during training
                timesteps = torch.randint(
                    0, self.config.diffusion_steps, (batch_size,), device=device
                )
            else:
                # Start from the highest noise level during inference
                timesteps = torch.ones((batch_size,), device=device).long() * (self.config.diffusion_steps - 1)
        
        # Compute embeddings
        token_embeddings = self.embed_tokens(input_ids)
        
        # For diffusion path: add noise to embeddings
        noisy_embeddings, noise = self.add_noise_to_embeddings(token_embeddings, timesteps)
        
        # Get diffusion step embeddings
        diffusion_emb = self.diffusion_embedding(timesteps)
        
        # Prepare attention mask if not provided
        if attention_mask is None:
            attention_mask = self.prepare_attention_mask(input_ids, self.config.pad_token_id)
        
        # Split processing between diffusion and autoregressive paths
        dit_hidden_states = noisy_embeddings
        ar_hidden_states = token_embeddings
        
        # Process through alternating DiT and AR layers
        for dit_layer, ar_layer in zip(self.dit_layers, self.ar_layers):
            # Process through DiT layer (non-causal diffusion transformer)
            dit_hidden_states = dit_layer(
                dit_hidden_states,
                diffusion_emb=diffusion_emb,
                attention_mask=attention_mask,
            )
            
            # Process through AR layer (causal autoregressive transformer)
            ar_hidden_states = ar_layer(
                ar_hidden_states,
                attention_mask=attention_mask,
            )
            
            # Optional cross-path exchange (fusion of information)
            if self.is_training:
                # Exchange information between paths during training
                exchange_weight = self.config.ar_weight
                dit_hidden_states = (1 - exchange_weight) * dit_hidden_states + exchange_weight * ar_hidden_states.detach()
                ar_hidden_states = exchange_weight * ar_hidden_states + (1 - exchange_weight) * dit_hidden_states.detach()
        
        # Final layer norms
        dit_hidden_states = self.dit_final_norm(dit_hidden_states)
        ar_hidden_states = self.ar_final_norm(ar_hidden_states)
        
        # Compute output predictions
        diffusion_output = self.diffusion_head(dit_hidden_states)  # Predict noise or clean embeddings
        ar_logits = self.ar_head(ar_hidden_states)  # Predict next token logits
        
        if not return_dict:
            return (diffusion_output, ar_logits, noise)
        
        return {
            "diffusion_output": diffusion_output,  # Predicted noise or clean embeddings
            "ar_logits": ar_logits,                # Autoregressive next token logits
            "noise": noise,                        # Target noise for diffusion loss
            "timesteps": timesteps,                # Timesteps used for diffusion
        }
        
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for both diffusion and autoregressive paths.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            timesteps: Optional diffusion timesteps [batch_size]
            label_smoothing: Label smoothing factor for autoregressive loss
            
        Returns:
            Dictionary containing loss values and metrics
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            timesteps=timesteps,
        )
        
        # Extract outputs
        diffusion_output = outputs["diffusion_output"]
        ar_logits = outputs["ar_logits"]
        noise = outputs["noise"]
        
        # Compute diffusion loss (predict the noise)
        diffusion_loss = F.mse_loss(diffusion_output, noise)
        
        # Compute autoregressive loss (shift labels for next token prediction)
        labels = input_ids.clone()
        # Shift right to get next token prediction targets
        labels = torch.cat([labels[:, 1:], torch.full((labels.shape[0], 1), self.config.pad_token_id, device=labels.device)], dim=1)
        
        # Create label mask (ignore padding in loss computation)
        label_mask = (labels != self.config.pad_token_id).float()
        
        # Compute cross entropy loss with optional label smoothing
        ar_loss = F.cross_entropy(
            ar_logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            reduction="none",
            label_smoothing=label_smoothing,
        )
        
        # Apply mask and compute mean
        ar_loss = (ar_loss * label_mask.view(-1)).sum() / label_mask.sum().clamp(min=1.0)
        
        # Combine losses with weighting
        total_loss = self.config.ar_weight * ar_loss + (1 - self.config.ar_weight) * diffusion_loss
        
        # Compute perplexity for autoregressive path
        ar_perplexity = torch.exp(ar_loss)
        
        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "ar_loss": ar_loss,
            "ar_perplexity": ar_perplexity,
        }
        
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_embeddings: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: int = 0,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_diffusion_steps: Optional[int] = None,
        guidance_scale: float = 1.0,
        use_autoregressive_path: bool = True,
        stopping_criteria: Optional[List[Callable]] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text using joint diffusion and autoregressive approach.
        
        Args:
            input_ids: Optional seed tokens [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            prompt_embeddings: Optional pre-embedded prompt [batch_size, seq_len, hidden_size]
            max_length: Maximum generation length
            min_length: Minimum generation length
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to sample or use greedy decoding
            num_diffusion_steps: Number of diffusion steps for denoising
            guidance_scale: Classifier-free guidance strength (1.0 = no guidance)
            use_autoregressive_path: Whether to use autoregressive predictions
            stopping_criteria: Optional stopping criteria
            pad_token_id: ID of padding token
            bos_token_id: ID of beginning of sequence token
            eos_token_id: ID of end of sequence token
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Dictionary with generation outputs
        """
        # Set default values
        batch_size = 1
        device = next(self.parameters()).device
        
        if max_length is None:
            max_length = self.config.max_seq_length
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
            
        if bos_token_id is None:
            bos_token_id = self.config.bos_token_id
            
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
            
        if num_diffusion_steps is None:
            num_diffusion_steps = self.config.diffusion_steps
        
        # Set model to evaluation mode
        self.eval()
        
        # Initialize with input_ids or create new starting tokens
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            input_len = input_ids.shape[1]
            max_effective_len = max_length
        else:
            batch_size = prompt_embeddings.shape[0] if prompt_embeddings is not None else 1
            input_len = 0
            max_effective_len = max_length
            
            # Create initial tokens (BOS tokens)
            input_ids = torch.full(
                (batch_size, 1), 
                bos_token_id, 
                dtype=torch.long, 
                device=device
            )
        
        # If max_length is provided, ensure we don't exceed model's max length
        max_effective_len = min(max_effective_len, self.config.max_seq_length)
        
        # Initialize attention mask if not provided
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids)
        
        # Initialize sequence storage
        generated_ids = input_ids.clone()
        
        # Track stopping conditions
        eos_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        seq_lengths = torch.ones(batch_size, dtype=torch.long, device=device) * max_effective_len
        
        # Create stopping criteria
        if stopping_criteria is None:
            stopping_criteria = []
        stopping_criteria.append(lambda ids, scores: (ids == eos_token_id).any(dim=1))
        
        # Initialize embeddings
        if prompt_embeddings is not None:
            # Use provided embeddings
            embeddings = prompt_embeddings
        else:
            # Compute embeddings from input_ids
            embeddings = self.embed_tokens(input_ids)
        
        # Initialize with random noise
        # Start from max noise and progressively denoise
        noisy_embeddings = torch.randn_like(embeddings)
        
        logger.info(f"Starting generation with batch size {batch_size} and max length {max_effective_len}")
        
        # Main generation loop
        for curr_len in range(input_len, max_effective_len):
            # Skip if all sequences have reached EOS
            if torch.all(eos_flags):
                break
                
            # Diffusion denoising process
            current_embeddings = self._diffusion_denoise(
                noisy_embeddings=noisy_embeddings,
                context_embeddings=embeddings if curr_len == input_len else None,
                num_steps=num_diffusion_steps,
                guidance_scale=guidance_scale,
            )
            
            # For autoregressive path, predict next tokens
            if use_autoregressive_path:
                # Forward pass through autoregressive portion
                ar_hidden_states = current_embeddings
                
                # Create attention mask
                curr_attention_mask = None
                if attention_mask is not None:
                    curr_attention_mask = self.prepare_attention_mask(
                        torch.cat([attention_mask, torch.ones(batch_size, curr_len - input_len, device=device)], dim=1),
                        pad_token_id
                    )
                
                # Process through AR layers
                for ar_layer in self.ar_layers:
                    ar_hidden_states = ar_layer(
                        ar_hidden_states,
                        attention_mask=curr_attention_mask,
                    )
                
                # Get final normalized states and predict next token
                ar_hidden_states = self.ar_final_norm(ar_hidden_states)
                next_token_logits = self.ar_head(ar_hidden_states[:, -1, :])
                
                # Apply sampling techniques
                if temperature > 0:
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply repetition penalty
                    if repetition_penalty > 1.0:
                        # Penalize already generated tokens
                        for batch_idx in range(batch_size):
                            for prev_token in generated_ids[batch_idx]:
                                next_token_logits[batch_idx, prev_token] /= repetition_penalty
                    
                    # Apply min_length constraint
                    if curr_len < min_length:
                        next_token_logits[:, eos_token_id] = -float("inf")
                    
                    # Sample next tokens
                    if do_sample:
                        # Top-k filtering
                        if top_k > 0:
                            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                            next_token_logits[indices_to_remove] = -float("inf")
                        
                        # Top-p (nucleus) filtering
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            
                            # Remove tokens with cumulative probability above the threshold
                            sorted_indices_to_remove = cumulative_probs > top_p
                            # Shift the indices to the right to keep also the first token above the threshold
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            
                            # Scatter sorted indices back to original logits
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                dim=1, index=sorted_indices, src=sorted_indices_to_remove
                            )
                            next_token_logits[indices_to_remove] = -float("inf")
                        
                        # Sample from the filtered distribution
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        # Greedy decoding
                        next_tokens = torch.argmax(next_token_logits, dim=-1)
                else:
                    # Greedy decoding (no temperature)
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Update sequences and check for EOS
                for batch_idx in range(batch_size):
                    if not eos_flags[batch_idx]:
                        # Add token to sequence
                        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
                        
                        # Check if EOS token is generated
                        if next_tokens[batch_idx] == eos_token_id:
                            eos_flags[batch_idx] = True
                            seq_lengths[batch_idx] = curr_len + 1
                
                # Embed new tokens and add to embeddings
                new_token_embeddings = self.token_embedding(next_tokens).unsqueeze(1)
                embeddings = torch.cat([embeddings, new_token_embeddings], dim=1)
                
                # Add new random noise for next iteration
                new_noise = torch.randn_like(new_token_embeddings)
                noisy_embeddings = torch.cat([noisy_embeddings, new_noise], dim=1)
            
            # Optional logging for generations
            if curr_len % 10 == 0:
                logger.debug(f"Generated {curr_len}/{max_effective_len} tokens")
        
        # Ensure all sequences are properly ended
        for batch_idx in range(batch_size):
            if not eos_flags[batch_idx]:
                seq_lengths[batch_idx] = max_effective_len
        
        # Return generation results
        if not return_dict:
            return generated_ids
        
        return {
            "sequences": generated_ids,
            "sequence_lengths": seq_lengths,
        }
    
    @torch.no_grad()
    def _diffusion_denoise(
        self,
        noisy_embeddings: torch.Tensor,
        context_embeddings: Optional[torch.Tensor] = None,
        num_steps: int = 100,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Perform diffusion denoising process.
        
        Args:
            noisy_embeddings: Noisy embeddings to denoise [batch_size, seq_len, hidden_size]
            context_embeddings: Optional conditioning context [batch_size, seq_len, hidden_size]
            num_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance (1.0 = no guidance)
            
        Returns:
            Denoised embeddings [batch_size, seq_len, hidden_size]
        """
        batch_size = noisy_embeddings.shape[0]
        device = noisy_embeddings.device
        
        # Default to full number of steps if not specified
        if num_steps is None or num_steps <= 0:
            num_steps = self.config.diffusion_steps
        
        # Compute step size
        steps = list(range(0, self.config.diffusion_steps, self.config.diffusion_steps // num_steps))
        if not steps or steps[-1] != self.config.diffusion_steps - 1:
            steps.append(self.config.diffusion_steps - 1)
        
        # Start with noisy embeddings
        x_t = noisy_embeddings
        
        # Iteratively denoise
        for step in reversed(steps):
            # Get current timestep for diffusion process
            timestep = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # Get diffusion step embedding
            diffusion_emb = self.diffusion_embedding(timestep)
            
            # Process through DiT layers
            x_hidden = x_t
            
            # Apply diffusion layers
            for dit_layer in self.dit_layers:
                x_hidden = dit_layer(
                    x_hidden,
                    diffusion_emb=diffusion_emb,
                    attention_mask=None,  # No masking for the diffusion path
                )
            
            # Apply final normalization and predict noise or clean data
            x_hidden = self.dit_final_norm(x_hidden)
            predicted_noise = self.diffusion_head(x_hidden)
            
            # Get alpha values for current timestep
            alpha_t = self._extract_at_t(self.diffusion_scheduler.alphas_cumprod, timestep).view(-1, 1, 1)
            alpha_t_prev = self._extract_at_t(
                F.pad(self.diffusion_scheduler.alphas_cumprod[:-1], (1, 0), value=1.0), 
                timestep
            ).view(-1, 1, 1)
            beta_t = 1 - alpha_t / alpha_t_prev
            
            # Compute predicted x_0 (clean data)
            predicted_x0 = (x_t - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
            
            # Classifier-free guidance if scale > 1.0
            if guidance_scale > 1.0 and context_embeddings is not None:
                # Use context embeddings as guidance
                x_prev = predicted_x0 * guidance_scale + context_embeddings * (1 - guidance_scale)
            else:
                # Standard denoising step
                x_prev = predicted_x0
            
            # Add noise for non-final step (following DDIM sampling)
            if step > 0:
                noise = torch.randn_like(x_t)
                x_t = alpha_t_prev.sqrt() * x_prev + (1 - alpha_t_prev).sqrt() * noise
            else:
                x_t = x_prev
        
        return x_t
    
    def _extract_at_t(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Extract values from a tensor at specified timesteps.
        
        Args:
            x: Tensor to extract from [T, ...]
            t: Timesteps to extract at [B]
            
        Returns:
            Extracted values [B, ...]
        """
        device = t.device
        out = torch.gather(x.to(device), 0, t)
        return out
    
    
    
model = DART(DARTConfig())



"""
DART Forward Pass Example

This script demonstrates how to use the Joint Diffusion and Autoregressive Transformer (DART)
model for both training and inference. It shows:
1. How to initialize the model with proper configuration
2. How to prepare inputs for the forward pass
3. How to process the model's outputs
4. How to use the model for both training and inference modes
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logger.configure(handlers=[
    dict(sink=lambda msg: print(msg), format="{time} | {level} | {message}", level="INFO"),
])

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize configuration
    config = DARTConfig(
        vocab_size=50257,  # GPT-2 vocabulary size
        hidden_size=768,
        num_hidden_layers=8,  # 4 DiT layers + 4 AR layers
        num_attention_heads=12,
        intermediate_size=3072,
        diffusion_steps=1000,
        diffusion_schedule="cosine",  # Cosine schedule usually works better
        ar_weight=0.5,  # Equal weight between AR and diffusion
    )
    
    # Initialize model
    model = DART(config).to(device)
    logger.info(f"Initialized DART model with {config.num_hidden_layers} layers")
    
    # Initialize tokenizer (using GPT-2 tokenizer for this example)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    # Sample input text
    sample_texts = [
        "Joint diffusion and autoregressive models combine the best of both worlds.",
        "This architecture leverages global context through diffusion and local dependencies through autoregression."
    ]
    
    # Tokenize inputs
    inputs = tokenizer(
        sample_texts,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    logger.info(f"Input shape: {inputs.input_ids.shape}")
    
    # Sample random timesteps for diffusion process (batch_size,)
    batch_size = inputs.input_ids.shape[0]
    timesteps = torch.randint(0, config.diffusion_steps, (batch_size,), device=device)
    
    # Forward pass
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        timesteps=timesteps,
    )
    
    print(outputs)
    
    # generations = model.generate(
    #     input_ids=inputs.input_ids,
    #     attention_mask=inputs.attention_mask,
    # )
    
    # print(generations)
    
    
    
if __name__ == "__main__":
    main()
