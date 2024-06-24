import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time

batch_size = 128
learning_rate = 3e-3
num_epochs = 5

torch.set_float32_matmul_precision('high')

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std of MNIST
])

train_dataset = datasets.MNIST(root='../../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../../data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print('Train Loader', enumerate(train_loader))
# Training the model
def train(train_loader, epoch):
    # time enumeration for train_loader to see bottleneck
    # start = time.time()
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     end = time.time()
    #     print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}')
    #     print(f'Time taken to load data: {end-start:.6f} seconds')
    #     start = time.time()
    #     data = data.to('cuda')
    #     end = time.time()
    #     print(f'Time taken to move data to GPU: {end-start:.6f} seconds')

    for batch_idx, (data, target) in enumerate(train_loader):
        
        
        
        if batch_idx % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}')
            start = time.time()
            data = data.to('cuda')
            end = time.time()
            print(f'Time taken to move data to GPU: {end-start:.6f} seconds')   
        else:
            data = data.to('cuda')

# Main
for epoch in range(num_epochs):
    train(train_loader, epoch)
    
print('Finished Training')
