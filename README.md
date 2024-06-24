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
- forward pass: 
  - load batch from DRAM to GPU
  - run forward pass kernels on GPU
    - conv2d
    - batchnorm2d
    - relu
    - maxpool2d
    - fc (matmul fwd)
    - res blocks
    - cross entropy loss
- backward pass:
  - our code architecture simplifies the backward pass because we don't have to construct the computational graph. We can just run the backward pass kernels in the reverse order of the forward pass and compute gradients on the level of tensors. we plan to pass the previous tensor gradients into the next bkwd function as we traverse the computational graph in reverse.
  - run backward pass kernels on GPU
    - derivative of cross entropy loss
    - fc (matmul bkwd)
    - res blocks
    - maxpool2d
    - relu
    - batchnorm2d
    - conv2d
- update weights:
    - kernel to parallelize weight updates (SGD only)
    - zero out gradients after updating weights to prevent gradient accumulation

- extra features
    - profile time per operation (to optimize around compute/memory bandwidth bottlenecks)
    - etc...


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
- MNIST batch dataloader (decompress + load image data in batches)
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
