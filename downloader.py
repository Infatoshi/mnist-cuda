import os

import numpy as np
from torchvision import datasets, transforms

# Set the directory where you want to save the files
save_dir = "mnist_data"
os.makedirs(save_dir, exist_ok=True)

# Download and load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
mnist_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Convert to numpy arrays and normalize
X_train = mnist_train.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train = mnist_train.targets.numpy().astype(np.int32)
X_test = mnist_test.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_test = mnist_test.targets.numpy().astype(np.int32)

# Save the data as raw binary files
X_train.tofile(os.path.join(save_dir, "X_train.bin"))
y_train.tofile(os.path.join(save_dir, "y_train.bin"))
X_test.tofile(os.path.join(save_dir, "X_test.bin"))
y_test.tofile(os.path.join(save_dir, "y_test.bin"))

# Save metadata
with open(os.path.join(save_dir, "metadata.txt"), "w") as f:
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Test samples: {X_test.shape[0]}\n")
    f.write(f"Input dimensions: {X_train.shape[1]}\n")
    f.write(f"Number of classes: {len(np.unique(y_train))}\n")

print("MNIST dataset has been downloaded and saved in binary format.")
