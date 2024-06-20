import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time


# Hyperparameters
batch_size = 128
learning_rate = 3e-3
num_epochs = 5

torch.set_float32_matmul_precision('high')

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std of MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define model with Residual Connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        identity = x # Shape: (batch_size, in_channels, H, W)
        out = self.relu(self.bn1(self.conv1(x))) # Shape: (batch_size, in_channels, H, W)
        out = self.bn2(self.conv2(out)) # Shape: (batch_size, in_channels, H, W)
        out += identity # Adding the input back: (batch_size, in_channels, H, W)
        out = self.relu(out) # Shape: unchanged
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_first = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn_first = nn.BatchNorm2d(32)
        self.resblock1 = ResidualBlock(32)
        self.resblock2 = ResidualBlock(64)
        self.conv_middle = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_middle = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.bn_first(self.conv_first(x))) # Shape: (batch_size, 32, 28, 28)
        x = F.max_pool2d(self.resblock1(x), kernel_size=2, stride=2) # Shape: (batch_size, 32, 14, 14)
        x = F.relu(self.bn_middle(self.conv_middle(x))) # Shape: (batch_size, 64, 14, 14)
        x = F.max_pool2d(self.resblock2(x), kernel_size=2, stride=2) # Shape: (batch_size, 64, 7, 7)
        x = torch.flatten(x, 1) # Shape: (batch_size, 64*7*7)
        x = F.relu(self.fc1(x)) # Shape: (batch_size, 128)
        x = self.fc2(x) # Shape: (batch_size, 10)
        # return self.dropout(F.log_softmax(x, dim=1)) # Shape: (batch_size, 10)
        return F.log_softmax(x, dim=1)
model = CNN().to("cuda")
model = torch.compile(model)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Training the model
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    train_iter_time = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # print(data.shape)
        data = data.to('cuda')
        start = time.time()
        outputs = model(data)
        end = time.time()
        if batch_idx > 10 and batch_idx % 100 == 99:
            train_iter_time.append(end-start)
        loss = criterion(outputs, target.to('cuda'))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if batch_idx % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {running_loss / 100:.3f}')
            print(f'Avg iter time {np.mean(train_iter_time) * 1e3:.6f}ms')
            train_iter_time = []
            running_loss = 0.0
    print(np.mean(train_iter_time), train_iter_time)

# Evaluation function to report average batch accuracy using PyTorch tensors with CUDA support
def evaluate(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    total_batch_accuracy = torch.tensor(0.0, device=device)
    num_batches = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
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
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    evaluate(model, test_loader)
    
print('Finished Training')


# check performance on inference

