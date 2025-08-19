import numpy as np
import time

# Load binary data
X_train = np.fromfile("data/X_train.bin", dtype=np.float32).reshape(60000, 784)[:10000]
y_train = np.fromfile("data/y_train.bin", dtype=np.int32)[:10000]
X_test = np.fromfile("data/X_test.bin", dtype=np.float32).reshape(10000, 784)
y_test = np.fromfile("data/y_test.bin", dtype=np.int32)

# Apply MNIST normalization
mean, std = 0.1307, 0.3081
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Reshape to (N, 1, 28, 28) format
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Linear layer
def initialize_weights(input_size, output_size):
    scale = np.sqrt(2.0 / input_size) 
    return (np.random.rand(input_size, output_size) * 2.0 - 1.0) * scale

def initialize_bias(output_size):
    return np.zeros((1, output_size))

def linear_forward(x, weights, bias):
    return x @ weights + bias

def linear_backward(grad_output, x, weights):
    grad_weights = x.T @ grad_output
    grad_bias = np.sum(grad_output, axis=0, keepdims=True)
    grad_input = grad_output @ weights.T
    return grad_input, grad_weights, grad_bias

# Softmax and Cross-Entropy Loss
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    probabilities = softmax(y_pred)
    correct_log_probs = np.log(probabilities[np.arange(batch_size), y_true])
    loss = -np.sum(correct_log_probs) / batch_size
    return loss

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights using He initialization (random)
        self.weights1 = initialize_weights(input_size, hidden_size)
        self.bias1 = initialize_bias(hidden_size)
        self.weights2 = initialize_weights(hidden_size, output_size)
        self.bias2 = initialize_bias(output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        fc1_input = x.reshape(batch_size, -1)
        fc1_output = linear_forward(fc1_input, self.weights1, self.bias1)
        relu_output = relu(fc1_output)
        fc2_output = linear_forward(relu_output, self.weights2, self.bias2)
        return fc2_output, (fc1_input, fc1_output, relu_output)

    def backward(self, grad_output, cache):
        x, fc1_output, relu_output = cache

        grad_fc2, grad_weights2, grad_bias2 = linear_backward(grad_output, relu_output, self.weights2)
        grad_relu = grad_fc2 * relu_derivative(fc1_output)
        grad_fc1, grad_weights1, grad_bias1 = linear_backward(grad_relu, x, self.weights1)
        return grad_weights1, grad_bias1, grad_weights2, grad_bias2

    def update_weights(self, grad_weights1, grad_bias1, grad_weights2, grad_bias2, learning_rate):
        self.weights1 -= learning_rate * grad_weights1
        self.bias1 -= learning_rate * grad_bias1
        self.weights2 -= learning_rate * grad_weights2
        self.bias2 -= learning_rate * grad_bias2

def train_timed(model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate):
    # Initialize timing stats
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0,
        'total_time': 0.0
    }
    
    # Start total timing
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            # Data loading timing
            data_start = time.time()
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            data_end = time.time()
            timing_stats['data_loading'] += data_end - data_start
            
            # Forward pass timing
            forward_start = time.time()
            y_pred, cache = model.forward(batch_X)
            forward_end = time.time()
            timing_stats['forward'] += forward_end - forward_start
            
            # Loss computation timing
            loss_start = time.time()
            loss = cross_entropy_loss(y_pred, batch_y)
            epoch_loss += loss

            softmax_probs = softmax(y_pred)
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[np.arange(len(batch_y)), batch_y] = 1
            grad_output = (softmax_probs - y_true_one_hot) / len(batch_y)
            loss_end = time.time()
            timing_stats['loss_computation'] += loss_end - loss_start

            # Backward pass timing
            backward_start = time.time()
            grad_weights1, grad_bias1, grad_weights2, grad_bias2 = model.backward(grad_output, cache)
            backward_end = time.time()
            timing_stats['backward'] += backward_end - backward_start
            
            # Weight updates timing
            update_start = time.time()
            model.update_weights(grad_weights1, grad_bias1, grad_weights2, grad_bias2, learning_rate)
            update_end = time.time()
            timing_stats['weight_updates'] += update_end - update_start

        print(f"Epoch {epoch} loss: {epoch_loss / (len(X_train) // batch_size):.4f}")

    # End total timing
    total_end = time.time()
    timing_stats['total_time'] = total_end - total_start
    
    # Print detailed timing breakdown
    print("\n=== PYTHON NUMPY IMPLEMENTATION TIMING BREAKDOWN ===")
    print(f"Total training time: {timing_stats['total_time']:.1f} seconds\n")
    
    print("Detailed Breakdown:")
    print(f"  Data loading:     {timing_stats['data_loading']:6.3f}s ({100.0 * timing_stats['data_loading'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Forward pass:     {timing_stats['forward']:6.3f}s ({100.0 * timing_stats['forward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Loss computation: {timing_stats['loss_computation']:6.3f}s ({100.0 * timing_stats['loss_computation'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Backward pass:    {timing_stats['backward']:6.3f}s ({100.0 * timing_stats['backward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Weight updates:   {timing_stats['weight_updates']:6.3f}s ({100.0 * timing_stats['weight_updates'] / timing_stats['total_time']:5.1f}%)")
    
    print("Training completed!")

if __name__ == "__main__":
    # Use random seed for natural variance (no fixed seed)
    
    input_size = 784  # 28x28 pixels
    hidden_size = 256
    output_size = 10  # 10 digits
    
    model = NeuralNetwork(input_size, hidden_size, output_size)
    
    batch_size = 8
    epochs = 10
    learning_rate = 0.01
    
    train_timed(model, X_train, y_train, X_test, y_test, batch_size, epochs, learning_rate)