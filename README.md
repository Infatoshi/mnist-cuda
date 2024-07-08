# MNIST in CUDA

> This is instruction manual for understanding + using the mnist training run in CUDA


## Setup
```bash
git clone https://github.com/Infatoshi/mnist-cuda`
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## High level approach to building the CUDA implementation
- initialize hyperparams and weights/biases on CPU in `int main()`, then copy weights/biases to GPU.
- dataloader: single function if it takes up too much memory, or a class if it doesn't (load all in at once then load batches directly from system DRAM)
- link architecture.png to the README.md -> [architecture.png](./architecture.png)

- update weights:
    - kernel to parallelize weight updates (SGD only)
    - zero out gradients after updating weights to prevent gradient accumulation

- extra features
    - profile time per operation (to optimize around compute/memory bandwidth bottlenecks)
    - etc...


## Purpose
We use a a modern CNN architecture to train a model on the MNIST dataset. 
We implement both the batched training run in pytorch, then translate over to CUDA C/C++ using iteratively optimized GPU kernels. I purposely left out batchnorm, residual blocks, lower-precision, and other optimizations to keep the code simple and easy to understand. It would also take wayyyy longer to implement and explain.



