# MNIST in CUDA

![](assets/mnist-mlp.png)
> This is instruction manual for understanding + using the mnist training run in CUDA


## Setup
> DISCLAIMER: ensure you have a GPU with compute capability 5.0 or greater (at least maxwell architecture). See compatibilty guide: https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html
```bash
git clone https://github.com/Infatoshi/mnist-cuda
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Purpose

We train an MLP on the MNIST dataset.
We implement both the batched training run in pytorch, then translate over to CUDA C/C++ using iteratively optimized GPU kernels. I purposely left out batchnorm, residual blocks, lower-precision, and other optimizations to keep the code simple and easy to understand. It would also take wayyyy longer to implement and explain.


## What we need to watch out for + pay attention to:

- [row vs col major](https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication)
- [tensor cores](https://docs.nvidia.com/cuda/cublas/#tensor-core-usage)

## Accelerate the data transfer via Prefetching

- [Unified vs Explicit Memory in CUDA](https://github.com/lintenn/cudaAddVectors-explicit-vs-unified-memory)
- [Maximizing Unified Memory Performance](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)
- Prefetching is automatically taken care of by unified memory via **streams** (this is what is has lower latency in the github link above)
  - [CUDA streams - Lei Mao](https://leimao.github.io/blog/CUDA-Stream/)
  - [NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
  - Streams allow for overlapping data transfer (prefetching) with computation.
  - While one stream is executing a kernel, another stream can be transferring data for the next computation.
  - This technique is often called "double buffering" or "multi-buffering" when extended to more buffers.

## Kernel Conversion
> we will change the following functions to kernels:
matmul_a_bt and matmul_at_b
relu_forward and relu_backward
bias_forward and bias_backward
softmax
compute_grad_output
compute_output_gradients
compute_hidden_gradients
update_gradients
