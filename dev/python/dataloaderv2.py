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

# load all data into system DRAM
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load all training data into system DRAM
train_data = torch.Tensor()
train_labels = torch.Tensor()
for batchidx, (data, label) in enumerate(train_loader):
    train_data = torch.cat((train_data, data), dim=0)
    train_labels = torch.cat((train_labels, label), dim=0)
print('Train Data Shape:', train_data.shape)
print('Train Data Type:', train_data.dtype)

iters_per_epoch = 60_000 // batch_size
print('Iters per epoch:', iters_per_epoch)

def get_batch(data, labels, i, batch_size):
    return data[i*batch_size:(i+1)*batch_size].to('cuda'), labels[i*batch_size:(i+1)*batch_size].to('cuda')


# # Training the model
def train():

    for epoch in range(num_epochs):
        for i in range(iters_per_epoch):
            start = time.time()
            x, y = get_batch(train_data, train_labels, i, batch_size)
            end = time.time()
            if i % 100 == 99:
                print(f'Epoch: {epoch+1}, Iter: {i+1}')
                print(f'Time taken to move data to GPU: {end-start:.6f} seconds')


print('Finished Training')

if __name__ == '__main__':
    train()

