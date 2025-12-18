# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MNIST digit classification using a 2-layer MLP (784 -> 1024 -> 10), implemented progressively from PyTorch to optimized CUDA. Each version demonstrates different optimization techniques.

**Architecture**: Input (784) -> Hidden (1024, ReLU) -> Output (10, Softmax)
**Training config**: 10,000 samples, batch size 32, 10 epochs, SGD lr=0.01, cross-entropy loss

## Build and Run Commands

```bash
# Setup - download MNIST data first
python downloader.py

# Python versions
python v1.py          # PyTorch CUDA baseline
python v2.py          # NumPy CPU

# C version
gcc -o v3 v3.c -lm && ./v3

# CUDA versions (use -arch=native or specify your GPU, e.g., -arch=sm_86)
nvcc -arch=native -o v4 v4.cu && ./v4                 # Naive CUDA kernels
nvcc -arch=native -o v5 v5.cu -lcublas && ./v5        # cuBLAS with GPU-side loss
```

## Version Progression

| Version | Implementation | Key Characteristics |
|---------|---------------|---------------------|
| v1.py | PyTorch CUDA | High-level ops, cuDNN, baseline reference |
| v2.py | NumPy CPU | Manual forward/backward, gradient computation |
| v3.c | Pure C | Manual memory, detailed timing per operation |
| v4.cu | CUDA | Custom matmul kernels, per-batch GPU allocation |
| v5.cu | cuBLAS | SGEMM/SAXPY, persistent buffers, GPU-side loss/gradients |

## Architecture Notes

### Data Format
- Binary files in `data/`: X_train.bin, y_train.bin, X_test.bin, y_test.bin
- Float32 for images (flattened 28x28), int32 for labels
- MNIST normalization applied: mean=0.1307, std=0.3081

### Weight Initialization
All versions use He uniform initialization: `scale = sqrt(2/fan_in)`, weights in `[-scale, scale]`

### Matrix Operations (CUDA versions)
- v4: Custom kernels for A@B, A@B.T, A.T@B with explicit thread/block dims
- v5: cuBLAS column-major SGEMM - note row/column major considerations when porting

### Key Optimization Patterns (v4 -> v5)
- Persistent GPU buffers vs per-batch malloc/free
- cuBLAS optimized GEMM vs naive kernels
- Minimal cudaDeviceSynchronize calls
- GPU-side softmax/loss/gradient computation (eliminates D2H logits + H2D gradients)

## Common Modifications

When adding new features or optimizations:
- Match He initialization across all versions for fair comparison
- Maintain timing instrumentation structure (TimingStats)
- Test numerical correctness against v1.py (PyTorch reference)
- GPU versions require compute capability 5.0+ (Maxwell or newer)
