# MNIST in CUDA

> This is instruction manual for understanding + using the mnist training run in CUDA


## Setup
```bash
git clone https://github.com/Infatoshi/mnist-cuda`
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Purpose
We train an MLP on the MNIST dataset. 
We implement both the batched training run in pytorch, then translate over to CUDA C/C++ using iteratively optimized GPU kernels. I purposely left out batchnorm, residual blocks, lower-precision, and other optimizations to keep the code simple and easy to understand. It would also take wayyyy longer to implement and explain.



