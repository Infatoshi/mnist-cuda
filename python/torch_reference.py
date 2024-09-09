import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRAIN_SIZE = 10000
epochs = 3
learning_rate = 1e-3
batch_size = 4
num_epochs = 3
data_dir = "../../../data"

torch.set_float32_matmul_precision("high")

# MNIST Dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std of MNIST
    ]
)


train_dataset = datasets.MNIST(
    root=data_dir, train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root=data_dir, train=False, transform=transform, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Pre-allocate tensors of the appropriate size
train_data = torch.zeros(len(train_dataset), 1, 28, 28)
train_labels = torch.zeros(len(train_dataset), dtype=torch.long)
test_data = torch.zeros(len(test_dataset), 1, 28, 28)
test_labels = torch.zeros(len(test_dataset), dtype=torch.long)

# Load all training data into RAM
for idx, (data, label) in enumerate(train_loader):
    start_idx = idx * batch_size
    end_idx = start_idx + data.size(0)
    train_data[start_idx:end_idx] = data
    train_labels[start_idx:end_idx] = label

print("Train Data Shape:", train_data.shape)
print("Train Data Type:", train_data.dtype)

# Load all test data into RAM
for idx, (data, label) in enumerate(test_loader):
    start_idx = idx * batch_size
    end_idx = start_idx + data.size(0)
    test_data[start_idx:end_idx] = data
    test_labels[start_idx:end_idx] = label

print("Test Data Shape:", test_data.shape)
print("Test Data Type:", test_data.dtype)

iters_per_epoch = TRAIN_SIZE // batch_size
print("Iters per epoch:", iters_per_epoch)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = x.reshape(batch_size, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = MLP(in_features=784, hidden_features=256, num_classes=10).to("cuda")
# model = torch.compile(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# Training the model
def train(model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for i in range(iters_per_epoch):

        optimizer.zero_grad()
        data = train_data[i * batch_size : (i + 1) * batch_size].to("cuda")
        target = train_labels[i * batch_size : (i + 1) * batch_size].to("cuda")

        start = time.time()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end = time.time()
        running_loss += loss.item()
        if i % 100 == 99 or i == 0:
            print(f"Epoch: {epoch+1}, Iter: {i+1}, Loss: {loss}")
            print(f"Iteration Time: {(end - start) * 1e3:.4f} sec")
            running_loss = 0.0


# Evaluation function to report average batch accuracy using the loaded test data
def evaluate(model, test_data, test_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_batch_accuracy = torch.tensor(0.0, device=device)
    num_batches = 0

    with torch.no_grad():
        for i in range(len(test_data) // batch_size):
            data = test_data[i * batch_size : (i + 1) * batch_size].to(device)
            target = test_labels[i * batch_size : (i + 1) * batch_size].to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct_batch = (predicted == target).sum().item()
            total_batch = target.size(0)
            if total_batch != 0:  # Check to avoid division by zero
                batch_accuracy = correct_batch / total_batch
                total_batch_accuracy += batch_accuracy
                num_batches += 1

    avg_batch_accuracy = total_batch_accuracy / num_batches
    print(f"Average Batch Accuracy: {avg_batch_accuracy * 100:.2f}%")


# Main
if __name__ == "__main__":
    for epoch in range(epochs):
        train(model, criterion, optimizer, epoch)
        evaluate(model, test_data, test_labels)

    print("Finished Training")
