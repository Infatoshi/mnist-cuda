import numpy as np
import torch
from torchvision import datasets, transforms

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        self.x = np.pad(x, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        n, c, h, w = self.x.shape
        out_h = (h - self.w.shape[2]) // self.stride + 1
        out_w = (w - self.w.shape[3]) // self.stride + 1
        out = np.zeros((n, self.w.shape[0], out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                out[:, :, i, j] = np.sum(self.x[:, np.newaxis, :, i*self.stride:i*self.stride+self.w.shape[2], j*self.stride:j*self.stride+self.w.shape[3]] * self.w[np.newaxis, :, :, :, :], axis=(2,3,4))

        return out

    def backward(self, dout):
        n, _, out_h, out_w = dout.shape
        dx = np.zeros_like(self.x)
        dw = np.zeros_like(self.w)

        for i in range(out_h):
            for j in range(out_w):
                x_slice = self.x[:, :, i*self.stride:i*self.stride+self.w.shape[2], j*self.stride:j*self.stride+self.w.shape[3]]
                for k in range(self.w.shape[0]):  # out_channels
                    dx[:, :, i*self.stride:i*self.stride+self.w.shape[2], j*self.stride:j*self.stride+self.w.shape[3]] += self.w[k, :, :, :] * dout[:, k, i, j][:, np.newaxis, np.newaxis, np.newaxis]
                    dw[k, :, :, :] += np.sum(x_slice * dout[:, k, i, j][:, np.newaxis, np.newaxis, np.newaxis], axis=0)

        return dx[:, :, self.padding:-self.padding, self.padding:-self.padding], dw

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * (self.x > 0)

class MaxPool:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        n, c, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        out = np.zeros((n, c, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                out[:, :, i, j] = np.max(x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size], axis=(2,3))

        return out

    def backward(self, dout):
        n, c, out_h, out_w = dout.shape
        dx = np.zeros_like(self.x)

        for i in range(out_h):
            for j in range(out_w):
                window = self.x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                mask = window == np.max(window, axis=(2,3))[:, :, np.newaxis, np.newaxis]
                dx[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += mask * dout[:, :, i:i+1, j:j+1]

        return dx

class Linear:
    def __init__(self, in_features, out_features):
        self.w = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)

    def forward(self, x):
        self.x = x
        return np.dot(self.w, x)

    def backward(self, dout):
        dx = np.dot(self.w.T, dout)
        dw = np.dot(dout, self.x.T)
        return dx, dw

class NeuralNet:
    def __init__(self):
        self.conv1 = ConvLayer(1, 32, 3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool(2, 2)
        self.fc1 = Linear(32 * 14 * 14, 128)
        self.relu2 = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = x.reshape(x.shape[0], -1).T
        x = self.fc1.forward(x)
        x = self.relu2.forward(x)
        x = self.fc2.forward(x)
        return x

    def backward(self, dout):
        dx, fc2_grad = self.fc2.backward(dout)
        dx = self.relu2.backward(dx)
        dx, fc1_grad = self.fc1.backward(dx)
        dx = dx.T.reshape(dx.shape[1], 32, 14, 14)
        dx = self.pool1.backward(dx)
        dx = self.relu1.backward(dx)
        dx, conv_grad = self.conv1.backward(dx)

        return [(conv_grad, ), (fc1_grad, ), (fc2_grad, )]

    def update_weights(self, grads, lr):
        self.conv1.w -= lr * grads[0][0]
        self.fc1.w -= lr * grads[1][0]
        self.fc2.w -= lr * grads[2][0]

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[1]
    p = softmax(y_pred)
    log_likelihood = -np.log(p[y_true, range(m)])
    loss = np.sum(log_likelihood) / m
    return loss

# Load and preprocess the data
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

X_train = mnist_train.data.numpy().reshape(-1, 1, 28, 28) / 255.0
y_train = mnist_train.targets.numpy()
X_test = mnist_test.data.numpy().reshape(-1, 1, 28, 28) / 255.0
y_test = mnist_test.targets.numpy()



# Training parameters
batch_size = 32
epochs = 5
lr = 1e-3  # Further reduced learning rate

model = NeuralNet()

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        # Forward pass
        y_pred = model.forward(batch_X)

        # Compute loss and gradients
        loss = cross_entropy_loss(y_pred, batch_y)
        dout = softmax(y_pred)
        dout[batch_y, range(len(batch_y))] -= 1
        dout /= len(batch_y)
 
        # Backward pass
        conv, fc1_grad, fc2_grad = model.backward(dout)

        # Update weights
        model.update_weights(conv, fc1_grad, fc2_grad, lr)

        if i % 64 == 0:
            print(f"Iter: {i//batch_size} Loss: {loss}")

    # Evaluate on test set
    y_pred = model.forward(X_test)
    test_loss = cross_entropy_loss(y_pred, y_test)
    accuracy = np.mean(np.argmax(y_pred, axis=0) == y_test)
    print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

print("Training completed!")