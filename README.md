# Caching-Quantization

A framework for optimizing diffusion model inference through intelligent activation caching and quantization techniques.

## Overview

This project implements (DPS) for efficient caching of neural network activations during diffusion model inference. By identifying and caching redundant activations across timesteps, we can significantly reduce memory consumption and computational overhead.

## Key Features

- **Activation Recording**: Capture and save model activations across different timesteps
- **DPS Scheduling**: Compute optimal caching schedules based on activation similarity
- **Flexible Caching**: Multiple caching strategies including standard, deep, and CKA-based approaches
- **Diffusion Model Support**: Full integration with diffusion-based generative models

## Project Structure

```
├── main.py                      # Entry point for training/sampling
├── save_activations.py          # Script to record model activations
├── dps.py                       # Dynamic Priority Scheduling implementation
├── check_activations.py         # Validation utilities for saved activations
├── benchmark_generation.py      # Benchmarking tools
├── benchmark_results.json       # Results from benchmarking runs
│
├── caching/                     # Caching implementations
│   ├── caching_wrapper.py       # Base caching wrapper
│   ├── deep_cache_wrapper.py    # Deep caching strategy
│   └── cka_cache_wrapper.py     # CKA-based caching
│
├── configs/                     # Configuration files
│   ├── cifar10.yml
│   ├── cifar10_with_caching.yml
│   ├── celeba.yml
│   ├── bedroom.yml
│   └── church.yml
│
├── datasets/                    # Data handling
│   ├── __init__.py
│   ├── celeba.py
│   ├── ffhq.py
│   ├── lsun.py
│   ├── vision.py
│   └── utils.py
│
├── functions/                   # Utility functions
│   ├── ckpt_util.py            # Checkpoint utilities
│   ├── denoising.py            # Denoising functions
│   ├── losses.py               # Loss functions
│   └── __init__.py
│
├── models/                      # Model definitions
│   ├── diffusion.py            # Diffusion model architecture
│   ├── ema.py                  # Exponential Moving Average
│   └── __init__.py
│
└── runners/                     # Training and evaluation runners
    ├── diffusion.py            # Main diffusion runner
    └── __init__.py
```

## Usage

### Step 1: Record Activations

Save model activations for a specific configuration:

```bash
python save_activations.py \
    --config celeba.yml \
    --ckpt path/to/checkpoint.pth \
    --timesteps 0,1,2,...,999 \
    --save_dir ./activations
```

**Options:**
- `--config`: Configuration file to use (default: `celeba.yml`)
- `--ckpt`: Path to model checkpoint
- `--timesteps`: Comma-separated list of timesteps to record
- `--save_dir`: Directory to save activation files (default: `activations`)
- `--save_per_timestep`: Save each timestep to a separate file
- `--batch_size`: Batch size for generation (default: 1)

### Step 2: Compute Caching Schedule

Calculate the optimal caching schedule using Dynamic Priority Scheduling:

```bash
python dps.py
```

This script:
1. Loads saved activations from the specified directory
2. Computes similarity matrix between timesteps using activation correlation
3. Identifies groups of similar timesteps that can share cached activations
4. Returns an optimized caching schedule

**Parameters** (edit in `dps.py`):
- `timesteps`: List of timesteps to process
- `save_dir`: Directory containing saved activations
- `threshold`: Similarity threshold for grouping (default: 0.98)
- `max_group_size`: Maximum number of timesteps per cache group

### Step 3: Apply Caching Strategy

Use the computed schedule with one of the caching wrappers:

```python
from caching.caching_wrapper import CachingWrapper
from dps import DPS_schedule

# Load the caching schedule
schedule = DPS_schedule(save_dir, timesteps)

# Wrap your model with caching
cached_model = CachingWrapper(model, schedule)

# Use the wrapped model in inference
output = cached_model(x, t)
```

### Training/Sampling

Run the main pipeline:

```bash
python main.py \
    --config cifar10.yml \
    --doc exp_name \
    --sample \
    --ni
```

**Common arguments:**
- `--config`: Configuration file path
- `--doc`: Experiment name/documentation
- `--sample`: Generate samples
- `--test`: Test mode
- `--resume_training`: Resume from checkpoint
- `--seed`: Random seed (default: 1234)
- `--ni`: No interaction mode (for batch jobs)

## Caching Strategies

### Standard Caching (`caching_wrapper.py`)
Basic activation caching that stores and retrieves activations based on timestep similarity.

### Deep Caching (`deep_cache_wrapper.py`)
Implements deeper caching strategies that leverage multiple layers of the network.

### CKA Caching (`cka_cache_wrapper.py`)
Uses Centered Kernel Alignment (CKA) for more sophisticated similarity metrics between activations.

## Configuration Files

Configuration files (YAML format) define:
- Model architecture parameters
- Dataset settings
- Training hyperparameters
- Data augmentation options

Example structure:
```yaml
model:
  in_channels: 3
  out_channels: 3
  # ... more architecture details

data:
  image_size: 32
  dataset: cifar10
  # ... more data settings

training:
  batch_size: 128
  learning_rate: 0.0001
  # ... more training details
```

## Evaluation

### Benchmark Generation

Generate performance benchmarks:

```bash
python benchmark_generation.py
```

This creates `benchmark_results.json` with metrics on:
- Memory consumption
- Inference speed
- Quality metrics (FID, etc.)

### FID Evaluation

```bash
python fid_eval.py
```

### FLOP Analysis

```bash
python flop_eval.py
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- NetworkX (for graph-based scheduling)
- PyYAML (for configuration)

## Installation

```bash
# Clone the repository
git clone <repository>
cd Caching-Quantization

# Install dependencies
pip install torch numpy pyyaml networkx
```

## Workflow Summary

1. **Record**: Capture activations using `save_activations.py`
2. **Schedule**: Compute optimal caching using `dps.py`
3. **Cache**: Apply caching wrapper with the computed schedule
4. **Evaluate**: Benchmark improvements with `benchmark_generation.py`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{...}
```

## License

This project is licensed under the terms in the LICENSE file.

## Contact

For questions or issues, please open a GitHub issue.
