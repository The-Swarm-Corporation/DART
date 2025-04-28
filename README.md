# DART: Diffusion-Autoregressive Recursive Transformer

## Overview
DART (Diffusion-Autoregressive Recursive Transformer) is a novel hybrid architecture that combines diffusion-based and autoregressive approaches for text generation. By leveraging both paradigms, DART achieves robust global coherence through diffusion while maintaining local consistency via autoregressive modeling.

## Key Features
- **Hybrid Architecture**: Integrates diffusion and autoregressive components in a unified framework
- **Adaptive Noise Scheduling**: Implements multiple noise scheduling strategies (linear, cosine, quadratic)
- **Flexible Generation**: Supports both conditional and unconditional text generation
- **Production Ready**: Full type annotations, comprehensive logging, and configurable parameters
- **Efficient Implementation**: Optimized attention mechanisms and memory usage
- **Modular Design**: Easy to extend and modify for specific use cases

## Model Architecture
DART consists of several key components:
- Diffusion Transformer (DiT) blocks for global dependency modeling
- Autoregressive blocks for local coherence
- Adaptive noise scheduling mechanism
- Dual-path information exchange during training
- Classifier-free guidance support

## Installation
```bash
pip install dart-transformer
# or
git clone https://github.com/your-repo/DART.git
cd DART
pip install -e .
```

## Quick Start
```python
from dart import DART, DARTConfig

# Initialize configuration
config = DARTConfig(
    vocab_size=50257,  # GPT-2 vocabulary size
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
```

## Configuration Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| vocab_size | 50257 | Vocabulary size (default: GPT-2) |
| hidden_size | 768 | Dimension of hidden layers |
| num_hidden_layers | 12 | Number of transformer layers |
| num_attention_heads | 12 | Number of attention heads |
| diffusion_steps | 1000 | Number of diffusion steps |
| ar_weight | 0.5 | Weight between AR and diffusion |

## Advanced Usage
### Custom Noise Scheduling
```python
model = DART(DARTConfig(
    diffusion_schedule="cosine",  # Options: linear, cosine, quadratic
    diffusion_steps=1000,
))
```

### Classifier-Free Guidance
```python
generated = model.generate(
    input_ids=input_ids,
    guidance_scale=7.5,  # Higher values = stronger guidance
)
```

## Performance and Benchmarks
- Training efficiency: ~X tokens/second on A100 GPU
- Memory usage: ~Y GB for batch size 32
- Generation speed: ~Z tokens/second

## Citation
If you use DART in your research, please cite:
```bibtex
@article{dart2024,
  title={DART: Diffusion-Autoregressive Recursive Transformer for Text Generation},
  author={Kye Gomez },
  journal={[Journal/Conference]},
  year={2024}
}
```

## Contributing
We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the authors of DiT and related works in diffusion models
- Built with PyTorch and Transformers library
- Special thanks to the research community for valuable feedback

