import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

batch_size = 8
learning_rate = 0.01
num_epochs = 5

# Define necessary naive operations
def conv2d_naive(x, weights, stride=1, padding=1):
    # Apply padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    N, C, H, W = x.shape
    F, _, HH, WW = weights.shape
    H_out = int(1 + (H + 2 * padding - HH) / stride)
    W_out = int(1 + (W + 2 * padding - WW) / stride)
    out = np.zeros((N, F, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + HH
            w_start = j * stride
            w_end = w_start + WW
            x_patched = x_padded[:, :, h_start:h_end, w_start:w_end]
            for k in range(F):
                out[:, k, i, j] = np.sum(x_patched * weights[k, :, :, :], axis=(1, 2, 3))

    return out

def conv2d_backward_naive(dout, x, w, stride=1, padding=1):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_out = int(1 + (H + 2 * padding - HH) / stride)
    W_out = int(1 + (W + 2 * padding - WW) / stride)
    
    # Using padded version of input
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + HH
            w_start = j * stride
            w_end = w_start + WW

            for n in range(N):
                idout = dout[n, :, i, j][:, None, None, None]
                dx_padded[n, :, h_start:h_end, w_start:w_end] += np.sum(w * idout, axis=0)
                dw += x_padded[n, :, h_start:h_end, w_start:w_end] * idout
            
    dx = dx_padded[:, :, padding:-padding, padding:-padding] if padding > 0 else dx_padded    
    
    return dx, dw

# Update Naive BatchNorm to return cache
def batch_norm_naive(x, gamma, beta, eps=1e-5):
    N, C, H, W = x.shape
    mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var = np.var(x, axis=(0, 2, 3), keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    cache = (x, x_norm, mean, var, gamma, beta, eps) # this one
    return out, cache

def d_batch_norm_naive(x, gamma, beta, dout, mean, var, eps=1e-5):
    N, C, H, W = x.shape
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=(0, 2, 3), keepdims=True)
    dmean = np.sum(dx_norm * -1.0 / np.sqrt(var + eps), axis=(0, 2, 3), keepdims=True) + dvar * np.sum(-2.0 * (x - mean), axis=(0, 2, 3), keepdims=True) / N
    dx = dx_norm * 1.0 / np.sqrt(var + eps) + dvar * 2.0 * (x - mean) / N + dmean / N
    
    return dx, dgamma, dbeta

def relu_naive(x):
    return np.maximum(0, x)

def d_relu_naive(x):
    return (x > 0).astype(x.dtype)

def max_pool2d_naive(x, pool_size=2, stride=2):
    N, C, H, W = x.shape
    H_out = int((H - pool_size) / stride + 1)
    W_out = int((W - pool_size) / stride + 1)
    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            x_patched = x[:, :, h_start:h_end, w_start:w_end]
            out[:, :, i, j] = np.max(x_patched, axis=(2, 3))

    return out

def max_pool2d_backward_naive(dout, x, pool_size=2, stride=2):
    N, C, H, W = x.shape
    H_out = int((H - pool_size) / stride + 1)
    W_out = int((W - pool_size) / stride + 1)
    dx = np.zeros_like(x)
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            x_patched = x[:, :, h_start:h_end, w_start:w_end]
            
            max_x_patched = np.max(x_patched, axis=(2, 3), keepdims=True)
            dx[:, :, h_start:h_end, w_start:w_end] += (x_patched == max_x_patched) * dout[:, :, i, j][:, :, None, None]

    return dx

def softmax_naive(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss_naive(pred, target):
    N = pred.shape[0]
    clipped_pred = np.clip(pred, 1e-12, 1. - 1e-12)
    return -np.sum(target * np.log(clipped_pred)) / N

def d_cross_entropy_loss_naive(pred, target):
    N = pred.shape[0]
    return (pred - target) / N



# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std of MNIST
])

train_dataset = datasets.MNIST(root='../../../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../../../data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load all training data into system DRAM
train_data = torch.Tensor()
train_labels = torch.LongTensor()  # Convert to LongTensor for labels
for batchidx, (data, label) in enumerate(train_loader):
    train_data = torch.cat((train_data, data), dim=0)
    train_labels = torch.cat((train_labels, label), dim=0)
print('Train Data Shape:', train_data.shape)
print('Train Data Type:', train_data.dtype)

# Load all test data into system DRAM
test_data = torch.Tensor()
test_labels = torch.LongTensor()  # Convert to LongTensor for labels
for batchidx, (data, label) in enumerate(test_loader):
    test_data = torch.cat((test_data, data), dim=0)
    test_labels = torch.cat((test_labels, label), dim=0)
print('Test Data Shape:', test_data.shape)
print('Test Data Type:', test_data.dtype)

iters_per_epoch = 60_000 // batch_size
print('Iters per epoch:', iters_per_epoch)


"""
Forward pass:
conv2d: kernel size 3x3, stride 1, padding 1, in 1 channel, out 32 channels
batchnorm2d: out 32 channels
relu
maxpool2d: kernel size 2x2, stride 2
fc1: in 32*14*14, out 128
relu
fc2: in 128, out 10
log softmax (dim=1)
"""



class Model():
    def __init__(self):
        self.conv1_w = np.random.randn(32, 1, 3, 3)
        self.bn1_gamma = np.ones((1, 32, 1, 1), dtype=np.float32)
        self.bn1_beta = np.zeros((1, 32, 1, 1), dtype=np.float32)
        self.fc1_w = np.random.randn(32*14*14, 128)
        self.fc2_w = np.random.randn(128, 10)
        print('Model initialized')

    def forward(self, x):
        self.x_conv1 = x
        x = conv2d_naive(x, self.conv1_w, stride=1, padding=1)
        # print('Conv1:', x.shape)

        x, self.bn1_cache = batch_norm_naive(x, self.bn1_gamma, self.bn1_beta)
        # print('BN1:', x.shape)
        
        self.x_relu1 = x
        x = relu_naive(x)
        # print('ReLU1:', x.shape)
        
        self.x_pool1 = x
        x = max_pool2d_naive(x, pool_size=2, stride=2)
        # print('MaxPool1:', x.shape)
        
        self.x_flatten = x
        x = x.reshape(x.shape[0], -1)
        # print('Flatten:', x.shape)
        
        self.x_fc1 = x
        x = np.dot(x, self.fc1_w)
        # print('FC1:', x.shape)
        
        self.x_relu2 = x
        x = relu_naive(x)
        # print('ReLU2:', x.shape)
        
        self.x_fc2 = x
        x = np.dot(x, self.fc2_w)
        # print('FC2:', x.shape)
        
        # x = softmax_naive(x)
        # print('Softmax:', x.shape)
        return x

    def backward(self, y_pred, y_true):
        loss = cross_entropy_loss_naive(y_pred, y_true)
        # print('Loss:', loss)

        dx = d_cross_entropy_loss_naive(y_pred, y_true)
        # print('dLoss:', dx.shape)

        dx = np.dot(dx, self.fc2_w.T)
        # print('dFC2:', dx.shape)

        dx = d_relu_naive(dx)
        # print('dReLU2:', dx.shape)

        dx = np.dot(dx, self.fc1_w.T)
        # print('dFC1:', dx.shape)
        
        dx = dx.reshape(self.x_flatten.shape)
        # print('dReshape:', dx.shape)

        dx = max_pool2d_backward_naive(dx, self.x_pool1, pool_size=2, stride=2)
        # print('dMaxPool1:', dx.shape)

        dx = d_relu_naive(dx)
        # print('dReLU1:', dx.shape)
        
        # print('cache:', self.bn1_cache)
        _, norm, mean, var, gamma, beta, eps = self.bn1_cache
        dx, dgamma, dbeta = d_batch_norm_naive(norm, gamma, beta, dx, mean, var, eps)
        # print('dBN1:', dx.shape)

        dx, dw_conv1 = conv2d_backward_naive(dx, self.x_conv1, self.conv1_w, stride=1, padding=1)
        # print('dConv1:', dx.shape)
        


        return loss, dgamma, dbeta, dw_conv1

model = Model()

for i in range(20):
    start_time = time.time()
 
    data = train_data[i*batch_size:(i+1)*batch_size]
    label = train_labels[i*batch_size:(i+1)*batch_size]
    data = data.numpy()
    label = label.numpy()
    y_pred = model.forward(data)
    print("\n"*2)
    y_true = np.eye(10)[label]
    loss, dgamma, dbeta, dw_conv1 = model.backward(y_pred, y_true)
    print('Epoch:', i, 'Loss:', loss)
    # Update batchnorm parameters
    model.bn1_gamma -= learning_rate * dgamma
    model.bn1_beta -= learning_rate * dbeta
    # gradient descent
    model.conv1_w -= learning_rate * model.conv1_w
    model.fc1_w -= learning_rate * model.fc1_w
    model.fc2_w -= learning_rate * model.fc2_w

    # zero grads
    model.conv1_w = np.zeros_like(model.conv1_w)
    model.fc1_w = np.zeros_like(model.fc1_w)
    model.fc2_w = np.zeros_like(model.fc2_w)
 
    end_time = time.time()
    # print('Epoch:', i, 'Time:', end_time - start_time)