import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def batchnorm2d(x, weight, bias, running_mean, running_var, training, momentum, eps):
    if training:
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = x.var(dim=(0, 2, 3), keepdim=True)
        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * var
    else:
        mean = running_mean
        var = running_var
    x = (x - mean) / (var + eps).sqrt()
    x = x * weight + bias
    return x, running_mean, running_var

def maxpool2d(x, kernel_size, stride, padding):
    N, C, H, W = x.shape
    H_out = (H - kernel_size + 2 * padding) // stride + 1
    W_out = (W - kernel_size + 2 * padding) // stride + 1
    x = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    x = x.reshape(N, C, H_out, W_out, -1)
    x = x.max(dim=-1)[0]
    return x

def conv2d(x, weight, bias, stride, padding):
    N, C, H, W = x.shape
    F, C, HH, WW = weight.shape
    H_out = (H - HH + 2 * padding) // stride + 1
    W_out = (W - WW + 2 * padding) // stride + 1
    x = x.unfold(2, HH, stride).unfold(3, WW, stride)
    x = x.permute(0, 4, 1, 2, 3)
    x = x.reshape(-1, C, HH, WW)
    weight = weight.reshape(F, -1)
    x = x @ weight.T + bias
    x = x.reshape(N, H_out, W_out, F)
    x = x.permute(0, 3, 1, 2)
    return x

def relu(x):
    return x * (x > 0)

def softmax(x):
    x = x - x.max(dim=-1, keepdim=True)[0]
    x = x.exp()
    x = x / x.sum(dim=-1, keepdim=True)
    return x

def cross_entropy(x, y):
    return -torch.log(x[torch.arange(len(x)), y]).mean()

