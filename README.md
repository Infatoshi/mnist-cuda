# MNIST in CUDA

> This is instruction manual for understanding + using the mnist training run in CUDA
## Purpose
We use a bunch of modern architectural hacks in this CNN variant with the goal of getting > 99% in 2-3 epochs over the dataset. 
We implement both the batched training run and a fast inference version in pytorch, which we then translate over to CUDA C/C++ using iteratively optimizing CUDA kernels

## Architectural components (unorganized)
- conv2d
- fc layer
- batchnorm2d
- maxpool2d
- residual connections
- maybe dropout (long story short... removing dropout actually slightly improved performance so we can leave it out in the CUDA implementation)

## TODO
- MNIST batch dataloader
- matmul forward
- matmul backward
- conv2d forward
- conv2d backward
- batchnorm forward
- batchnorm backward (might not be required if its just a running mean / stddev)
- tensor level addition for residual stream (fwd & bkwd)
- maxpool2d forward & backward
- reshape/view/permute operations to format tensors as needed (only do this if the ending pytorch implementation requires it)

## Required CUDA libs / functions
- cuFFT for fast conv -> conv(x, knl) = ifft ( fft(x) * fft(knl) )
- 
