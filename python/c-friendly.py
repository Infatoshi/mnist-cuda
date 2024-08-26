import numpy as np
from torchvision import datasets, transforms


# Load and preprocess the data
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='/mnt/d/CUDA/cuda-learn/mnist-cuda/mnist_data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='/mnt/d/CUDA/cuda-learn/mnist-cuda/mnist_data', train=False, download=True, transform=transform)

X_train = mnist_train.data.numpy().reshape(-1, 1, 28, 28) / 255.0
y_train = mnist_train.targets.numpy()
X_test = mnist_test.data.numpy().reshape(-1, 1, 28, 28) / 255.0
y_test = mnist_test.targets.numpy()



# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Linear layer
def initialize_weights(input_size, output_size):
    # return np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

def linear_forward(x, weights):
    # return weights @ x # (output_size, input_size) @ (input_size, batch_size) = (output_size, batch_size
    return x @ weights # (batch_size, input_size) @ (input_size, output_size) = (batch_size, output_size)

# def linear_backward(grad_output, x, weights):
#     # print shapes
#     print('grad_output:', grad_output.shape)
#     print('x:', x.shape)
#     print('weights:', weights.shape)
#     # grad_output: (10, 8)
#     # x: (256, 8)
#     # weights: (10, 256)
#     grad_weights = grad_output @ x.T # (10, 8) @ (8, 256) = (10, 256)
#     grad_input = weights.T @ grad_output # (256, 10) @ (10, 8) = (256, 8)
#     return grad_input, grad_weights

def linear_backward(grad_output, x, weights):
    grad_weights = x.T @ grad_output
    grad_input = grad_output @ weights.T
    return grad_input, grad_weights

# Softmax and Cross-Entropy Loss
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # changed from axis=0
    return exp_x / np.sum(exp_x, axis=1, keepdims=True) # changed from axis=0

def cross_entropy_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    probabilities = softmax(y_pred)
    # Use np.arange(batch_size) for the first index, and y_true for the second index
    correct_log_probs = np.log(probabilities[np.arange(batch_size), y_true])
    loss = -np.sum(correct_log_probs) / batch_size
    return loss


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = initialize_weights(input_size, hidden_size)
        self.weights2 = initialize_weights(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        fc1_input = x.reshape(batch_size, -1)  # Flatten the input
        fc1_output = linear_forward(fc1_input, self.weights1)
        relu_output = relu(fc1_output)
        fc2_output = linear_forward(relu_output, self.weights2)
        return fc2_output, (fc1_input, fc1_output, relu_output)

    def backward(self, grad_output, cache):
        x, fc1_output, relu_output = cache

        grad_fc2, grad_weights2 = linear_backward(grad_output, relu_output, self.weights2)
        grad_relu = grad_fc2 * relu_derivative(fc1_output)
        grad_fc1, grad_weights1 = linear_backward(grad_relu, x, self.weights1)
        return grad_weights1, grad_weights2

    def update_weights(self, grad_weights1, grad_weights2, learning_rate):
        self.weights1 -= learning_rate * grad_weights1
        self.weights2 -= learning_rate * grad_weights2

def train(model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            # Forward pass
            y_pred, cache = model.forward(batch_X)

            # Compute loss and gradients
            loss = cross_entropy_loss(y_pred, batch_y)

            # Compute gradient
            softmax_probs = softmax(y_pred)
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[np.arange(len(batch_y)), batch_y] = 1
            # grad_output = (softmax_probs - y_true_one_hot) / len(batch_y)
            grad_output = (softmax_probs - y_true_one_hot)

            # Backward pass
            grad_weights1, grad_weights2 = model.backward(grad_output, cache)

            # Update weights
            model.update_weights(grad_weights1, grad_weights2, learning_rate)

            if (i//batch_size) % 100 == 0:
                print(f"Iteration: {i//batch_size} Loss: {loss:.4f}")

        # Evaluate on test set
        y_pred, _ = model.forward(X_test)
        test_loss = cross_entropy_loss(y_pred, y_test)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test)
        print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    print("Training completed!")

# Main execution
if __name__ == "__main__":
    # Assume X_train, y_train, X_test, y_test are loaded from somewhere
    
    input_size = 784  # 28x28 pixels
    hidden_size = 1024
    output_size = 10  # 10 digits
    
    model = NeuralNetwork(input_size, hidden_size, output_size)
    
    batch_size = 16
    epochs = 5
    learning_rate = 5e-3
    
    train(model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate)
