import numpy as np

def max_pool2d_backward_naive(prev_layer_grad, x, pool_size=2, stride=2):
    """
    Backward pass for a 2D max-pooling layer.
    
    Args:
    - prev_layer_grad: Gradient of the loss with respect to the outputs of the max pooling layer (shape: (N, C, H_out, W_out)).
    - x: Input to the max pooling layer during the forward pass (shape: (N, C, H_in, W_in)).
    - pool_size: Size of the pooling window (default 2).
    - stride: Stride of the pooling window (default 2).
    
    Returns:
    - dx: Gradient of the loss with respect to the inputs of the max pooling layer (shape: (N, C, H_in, W_in)).
    """
    N, C, H_in, W_in = x.shape
    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1
    
    # Initialize the gradient with respect to input with zeros
    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size

                    # Extract the current pooling region from the input
                    x_pool = x[n, c, h_start:h_end, w_start:w_end]
                    
                    # Determine the maximum value in the pooling region
                    max_val = np.max(x_pool)
                    
                    # Create a mask that is 1 at the position of the max value and 0 elsewhere
                    mask = (x_pool == max_val)
                    
                    # Propagate the gradient through the max-pooling operation
                    dx[n, c, h_start:h_end, w_start:w_end] += prev_layer_grad[n, c, h, w] * mask
                    
    return dx

# Example usage
if __name__ == "__main__":
    # Example input and simulated gradients from the next layer's backward pass
    x = np.array([[[[3., 1.], [2., 4.]]]], dtype=np.float32)
    
    # Output shape after a max pooling with pool_size=2 and stride=2
    prev_layer_grad = np.array([[[[1.]]]], dtype=np.float32)  # Shape should match the output from max pooling
    
    # No need to reshape prev_layer_grad since it already matches the expected shape
    # The function max_pool2d_backward_naive will handle the gradient propagation correctly
    dx = max_pool2d_backward_naive(prev_layer_grad, x, pool_size=2, stride=2)
    
    print("Input x:")
    print(x)
    print("Gradient from next layer (prev_layer_grad):")
    print(prev_layer_grad)
    print("Gradient with respect to input (dx):")
    print(dx)