import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time

batch_size = 64
learning_rate = 5e-3
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


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        # print("conv", x.shape)
        x = self.batchnorm(x)
        # print("bn", x.shape)
        x = self.relu(x)
        # print("relu", x.shape)
        x = self.maxpool(x)
        # print("maxpool", x.shape)
        return x
    
class MLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class CombinedModel(nn.Module):
    def __init__(self, input_channels, num_classes, out_channels=32):
        super(CombinedModel, self).__init__()
        self.cnn_block1 = CNNBlock(input_channels, out_channels)
        # self.cnn_block2 = CNNBlock(32, 64)
        # Define further CNN blocks as needed
        # Flatten the output from the last CNN block
        self.flatten = nn.Flatten()
        # 64*7*7 is the number of features after flattening w/ 64 output channels, 7x7 spatial dimensions
        # 32*14*14 is the number of features after flattening w/ 32 output channels, 14x14 spatial dimensions
        self.mlp_block1 = MLPBlock(out_channels * 14 * 14, 128, num_classes)
        # self.log_sm_out = nn.LogSoftmax(dim=1)
        # self.sm_out = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.cnn_block1(x)
        # x = self.cnn_block2(x)
        # Forward through more CNN blocks if defined
        x = self.flatten(x)
        logits = self.mlp_block1(x)
        # logits = self.sm_out(x)
        return logits
    
model = CombinedModel(input_channels=1, num_classes=10).to('cuda')
# model = torch.compile(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# Training the model
def train(model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for i in range(iters_per_epoch):
        
        optimizer.zero_grad()
        data = train_data[i*batch_size:(i+1)*batch_size].to('cuda')
        target = train_labels[i*batch_size:(i+1)*batch_size].to('cuda')
        start = time.time()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        end = time.time()
        running_loss += loss.item()
        if i % 100 == 99 or i == 0:
            print(f'Epoch: {epoch+1}, Iter: {i+1}, Loss: {loss}')
            print(f'Iteration Time: {(end - start) * 1e3:.4f} sec')
            running_loss = 0.0

# Evaluation function to report average batch accuracy using the loaded test data
def evaluate(model, test_data, test_labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    total_batch_accuracy = torch.tensor(0.0, device=device)
    num_batches = 0
    
    with torch.no_grad():
        for i in range(len(test_data) // batch_size):
            data = test_data[i * batch_size: (i + 1) * batch_size].to(device)
            target = test_labels[i * batch_size: (i + 1) * batch_size].to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct_batch = (predicted == target).sum().item()
            total_batch = target.size(0)
            if total_batch != 0:  # Check to avoid division by zero
                batch_accuracy = correct_batch / total_batch
                total_batch_accuracy += batch_accuracy
                num_batches += 1
    
    avg_batch_accuracy = total_batch_accuracy / num_batches
    print(f'Average Batch Accuracy: {avg_batch_accuracy * 100:.2f}%')

# Main
for epoch in range(10):
    train(model, criterion, optimizer, epoch)
    evaluate(model, test_data, test_labels)
    
print('Finished Training')

