# MNIST in CUDA

![](assets/mnist-mlp.png)
> Complete implementation progression from PyTorch to optimized CUDA: A step-by-step journey through 5 versions

## Purpose

This project implements a simple 2-layer MLP (Multi-Layer Perceptron) for MNIST digit classification, progressively optimizing from high-level PyTorch to low-level CUDA implementations. Each version demonstrates different optimization techniques and trade-offs between performance and complexity.

**Architecture**: 784 → 256 → 10 (input → hidden → output)  
**Dataset**: 10,000 MNIST training samples, batch size 8, 10 epochs  
**Activation**: ReLU, Loss: Cross-entropy, Optimizer: SGD (lr=0.01)

## Setup
> DISCLAIMER: ensure you have a GPU with compute capability 5.0 or greater (at least maxwell architecture). See compatibility guide: https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html

```bash
git clone https://github.com/Infatoshi/mnist-cuda
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download MNIST data (if not already present)
python downloader.py
```

### CUDA Setup
For CUDA versions (v4, v5), ensure you have NVIDIA CUDA toolkit installed:
```bash
# Check CUDA installation
nvcc --version

# Compile v4 (naive CUDA kernels)
nvcc -o v4 v4.cu

# Compile v5 (cuBLAS optimized)
nvcc -o v5 v5.cu -lcublas
```


## Version Progression

### v1.py - PyTorch Baseline
- **Framework**: PyTorch with CUDA tensors
- **Features**: 
  - High-level PyTorch operations (Linear, ReLU, CrossEntropyLoss)
  - GPU tensors with automatic memory management
  - Built-in optimizations (cuDNN, etc.)
- **Purpose**: Establishes baseline performance and correctness reference

### v2.py - NumPy Implementation  
- **Framework**: Pure NumPy (CPU-only)
- **Features**:
  - Manual forward/backward pass implementation
  - Custom gradient computation and weight updates
  - He initialization for weights
- **Purpose**: Demonstrates the underlying math without GPU acceleration

### v3.c - C/CPU Implementation
- **Framework**: Pure C with timing breakdown
- **Features**:
  - Manual memory management
  - Detailed timing instrumentation per operation
  - CPU-optimized matrix operations
- **Purpose**: Shows CPU performance baseline and prepares for GPU porting

### v4.cu - Naive CUDA Kernels
- **Framework**: CUDA C with custom kernels
- **Features**:
  - Custom matrix multiplication kernels
  - Element-wise operations (ReLU, bias, softmax) on GPU
  - Manual memory transfers between host and device
- **Purpose**: First GPU implementation with basic CUDA kernels

### v5.cu - cuBLAS Optimized
- **Framework**: CUDA with cuBLAS library
- **Features**:
  - cuBLAS optimized matrix operations (SGEMM, SAXPY)  
  - Persistent memory buffers to reduce allocations
  - Minimal host-device synchronization points
  - Optimized memory access patterns
- **Purpose**: Production-quality implementation with maximum performance

## Usage

```bash
# Run each version
python v1.py          # PyTorch baseline
python v2.py          # NumPy CPU implementation  
gcc -o v3 v3.c -lm && ./v3                    # C CPU implementation
nvcc -o v4 v4.cu && ./v4                      # Naive CUDA kernels
nvcc -o v5 v5.cu -lcublas && ./v5             # Optimized cuBLAS
```

## Performance Comparison

Expected performance characteristics (timing will vary by hardware):

| Version | Implementation | Typical Training Time | Relative Performance |
|---------|----------------|---------------------|---------------------|
| v1.py   | PyTorch CUDA   | ~2-5 seconds        | 100% (baseline)     |
| v2.py   | NumPy CPU      | ~60-120 seconds     | 5-10% of baseline   |
| v3.c    | C CPU          | ~40-80 seconds      | 10-15% of baseline  |
| v4.cu   | Naive CUDA     | ~8-15 seconds       | 30-60% of baseline  |
| v5.cu   | cuBLAS CUDA    | ~1-3 seconds        | 120-200% of baseline|

### Timing Breakdown Analysis

Each implementation provides detailed timing breakdowns:

**v1 (PyTorch)**: High-level operations with cuDNN optimization
- Forward pass: ~40% of time
- Backward pass: ~50% of time  
- Weight updates: ~5% of time
- Data loading: ~5% of time

**v5 (cuBLAS Optimized)**: Production-level performance
- GPU compute (forward + backward + updates): ~60% of time
- Memory transfers (H2D + D2H): ~30% of time
- Host computation (loss calculation): ~10% of time

## Performance Insights

Key observations from the implementation progression:
- **Memory Management**: Persistent buffers (v5) vs per-batch allocation (v4) significantly impacts performance
- **Library Optimization**: cuBLAS provides highly optimized GEMM operations that outperform naive kernels
- **CPU-GPU Transfer**: Minimizing host-device synchronization is crucial for GPU performance
- **Numerical Stability**: Proper softmax implementation with max subtraction prevents overflow
- **Hardware Utilization**: v5 achieves the best performance by maximizing GPU compute utilization

## Key CUDA Concepts Demonstrated

- [Row vs Column Major](https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication): Matrix layout considerations for GEMM
- [Tensor Cores](https://docs.nvidia.com/cuda/cublas/#tensor-core-usage): Hardware acceleration for mixed-precision operations
- **Memory Patterns**: Coalesced access and persistent allocation strategies
- **Kernel Launch**: Grid/block dimensions and occupancy considerations

## Advanced Optimizations (Not Implemented)

- [Unified Memory](https://github.com/lintenn/cudaAddVectors-explicit-vs-unified-memory): Simplified memory management
- [CUDA Streams](https://leimao.github.io/blog/CUDA-Stream/): Overlapping computation with data transfer
- **Mixed Precision**: FP16 operations with Tensor Cores
- **Kernel Fusion**: Combining operations to reduce memory bandwidth
