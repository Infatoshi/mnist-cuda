#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

// Timing accumulator structure
typedef struct {
    double data_loading;
    double fwd_matmul1;
    double fwd_bias1;
    double fwd_relu;
    double fwd_matmul2;
    double fwd_bias2;
    double fwd_softmax;
    double cross_entropy;
    double bwd_output_grad;
    double bwd_matmul2;
    double bwd_bias2;
    double bwd_relu;
    double bwd_matmul1;
    double bwd_bias1;
    double weight_updates;
    double total_time;
} TimingStats;

// Helper function to get time difference in seconds
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

#define INPUT_SIZE 784
#define HIDDEN_SIZE 1024
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 10
#define LEARNING_RATE 0.01

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

typedef struct {
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
    float *grad_weights1;
    float *grad_weights2;
    float *grad_bias1;
    float *grad_bias2;
} NeuralNetwork;

// load batched img data
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// load batch labels
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// optimal uniform He init for weights
void initialize_weights(float *weights, int input_size, int output_size) {
    float scale = sqrtf(2.0f / input_size);
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

// basic init for biases
void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// normalize data using MNIST mean and std
void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

// GPU KERNELS

// CUDA kernel for matrix multiplication (A @ B)
__global__ void matmul_a_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// CUDA kernel for matrix multiplication (A @ B.T)
__global__ void matmul_a_bt_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[col * n + i];
        }
        C[row * k + col] = sum;
    }
}

// CUDA kernel for matrix multiplication (A.T @ B)
__global__ void matmul_at_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i) {
            sum += A[i * n + row] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// ReLU forward kernel
__global__ void relu_forward_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

// Add bias kernel
__global__ void bias_forward_kernel(float *x, float *bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;

    if (b < batch_size && i < size) {
        x[idx] += bias[i];
    }
}

// Softmax kernel (simplified version - runs on CPU for now)
__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    if (b < batch_size) {
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) {
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        for (int i = 0; i < size; ++i) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

// Zero gradients kernel
__global__ void zero_grad_kernel(float *grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

// Compute output gradients kernel
__global__ void compute_output_gradients_kernel(float *grad_output, float *output, int *labels, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
        
        // Divide by batch_size
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            grad_output[b * OUTPUT_SIZE + i] /= batch_size;
        }
    }
}

// ReLU backward kernel
__global__ void relu_backward_kernel(float *grad, float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] *= (x[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

// Bias backward kernel
__global__ void bias_backward_kernel(float *grad_bias, float *grad, int batch_size, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad[b * size + i];
        }
        grad_bias[i] = sum;
    }
}

// Weight update kernel
__global__ void weight_update_kernel(float *weights, float *grad_weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
    }
}

// Modified forward function with timing (GPU version of v3.c)
void forward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    dim3 block_size(32, 32);
    
    // Input to Hidden (X @ W1)
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_size1((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);
    matmul_a_b_kernel<<<grid_size1, block_size>>>(input, nn->weights1, hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul1 += get_time_diff(start, end);
    
    // Add bias1
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias1 += get_time_diff(start, end);
    
    // Apply ReLU
    clock_gettime(CLOCK_MONOTONIC, &start);
    relu_forward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_relu += get_time_diff(start, end);

    // Hidden to Output (Hidden @ W2)
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_size2((OUTPUT_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);
    matmul_a_b_kernel<<<grid_size2, block_size>>>(hidden, nn->weights2, output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul2 += get_time_diff(start, end);

    // Add bias2
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias2 += get_time_diff(start, end);
    
    // Apply softmax
    clock_gettime(CLOCK_MONOTONIC, &start);
    softmax_kernel<<<batch_size, 1>>>(output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_softmax += get_time_diff(start, end);
}

// Cross entropy loss (CPU - same as v3.c)
float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}

// Backward pass function with timing (GPU version of v3.c)
void backward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    dim3 block_size(32, 32);
    
    // Initialize gradients to zero
    zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256>>>(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad_kernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);

    // Allocate temp gradients on GPU
    float *grad_output, *dX2, *d_ReLU_out;
    CUDA_CHECK(cudaMalloc(&grad_output, batch_size * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dX2, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ReLU_out, batch_size * HIDDEN_SIZE * sizeof(float)));

    // Compute gradients for output layer
    clock_gettime(CLOCK_MONOTONIC, &start);
    compute_output_gradients_kernel<<<(batch_size + 255) / 256, 256>>>(grad_output, output, labels, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_output_grad += get_time_diff(start, end);

    // Update gradients for weights2 (W2.grad = hidden.T @ grad_output)
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_weights2((OUTPUT_SIZE + block_size.x - 1) / block_size.x, (HIDDEN_SIZE + block_size.y - 1) / block_size.y);
    matmul_at_b_kernel<<<grid_weights2, block_size>>>(hidden, grad_output, nn->grad_weights2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul2 += get_time_diff(start, end);

    // Update gradients for bias2
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->grad_bias2, grad_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias2 += get_time_diff(start, end);

    // Compute dX2 (gradient of loss w.r.t. input of second layer)
    // grad_output @ W2.T = dX2 -> (B, 10) @ (10, 256) = (B, 256)
    dim3 grid_hidden((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);
    matmul_a_bt_kernel<<<grid_hidden, block_size>>>(grad_output, nn->weights2, dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);

    // Compute d_ReLU_out (element-wise multiplication with ReLU derivative)
    clock_gettime(CLOCK_MONOTONIC, &start);
    relu_backward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(dX2, hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(d_ReLU_out, dX2, batch_size * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_relu += get_time_diff(start, end);
    
    // Update gradients for weights1 (W1.grad = input.T @ d_ReLU_out)
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_weights1((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (INPUT_SIZE + block_size.y - 1) / block_size.y);
    matmul_at_b_kernel<<<grid_weights1, block_size>>>(input, d_ReLU_out, nn->grad_weights1, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul1 += get_time_diff(start, end);

    // Update gradients for bias1
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_bias1, d_ReLU_out, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias1 += get_time_diff(start, end);

    // Free temp GPU memory
    CUDA_CHECK(cudaFree(grad_output));
    CUDA_CHECK(cudaFree(dX2));
    CUDA_CHECK(cudaFree(d_ReLU_out));
}

// gradient descent step with timing (GPU version)
void update_weights_timed(NeuralNetwork *nn, TimingStats *stats) {
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    weight_update_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256>>>(nn->weights1, nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    weight_update_kernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(nn->weights2, nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    weight_update_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(nn->bias1, nn->grad_bias1, HIDDEN_SIZE);
    weight_update_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->bias2, nn->grad_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->weight_updates += get_time_diff(start, end);
}

// Train function with comprehensive timing (GPU version of v3.c)
void train_timed(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *d_hidden, *d_output, *d_input_batch;
    int *d_labels_batch;

    // Allocate GPU memory for batch processing
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels_batch, BATCH_SIZE * sizeof(int)));

    // Host buffers for loss computation
    float *h_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;
    
    // Initialize timing stats
    TimingStats stats = {0};
    
    // Start total timing
    struct timespec total_start, total_end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            
            // Data loading timing 
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float *batch_input = &X_train[start_idx * INPUT_SIZE];
            int *batch_labels = &y_train[start_idx];
            
            // Copy data to GPU
            CUDA_CHECK(cudaMemcpy(d_input_batch, batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_labels_batch, batch_labels, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.data_loading += get_time_diff(step_start, step_end);
            
            // Forward pass with timing
            forward_timed(nn, d_input_batch, d_hidden, d_output, BATCH_SIZE, &stats);

            // Copy output back to CPU for loss computation
            CUDA_CHECK(cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            // Cross entropy loss timing  
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float loss = cross_entropy_loss(h_output, batch_labels, BATCH_SIZE);
            total_loss += loss;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.cross_entropy += get_time_diff(step_start, step_end);

            // Backward pass with timing
            backward_timed(nn, d_input_batch, d_hidden, d_output, d_labels_batch, BATCH_SIZE, &stats);
            
            // Weight update with timing
            update_weights_timed(nn, &stats);
        }
        
        printf("Epoch %d loss: %.4f\n", epoch, total_loss / num_batches);
    }
    
    // End total timing
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    stats.total_time = get_time_diff(total_start, total_end);
    
    // Print detailed timing breakdown
    printf("\n=== CUDA GPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
    printf("Total training time: %.1f seconds\n\n", stats.total_time);
    
    printf("Detailed Breakdown:\n");
    printf("  Data loading:     %6.3fs (%5.1f%%)\n", stats.data_loading, 100.0 * stats.data_loading / stats.total_time);
    double forward_pass = stats.fwd_matmul1 + stats.fwd_bias1 + stats.fwd_relu + stats.fwd_matmul2 + stats.fwd_bias2 + stats.fwd_softmax;
    printf("  Forward pass:     %6.3fs (%5.1f%%)\n", forward_pass, 100.0 * forward_pass / stats.total_time);
    printf("  Loss computation: %6.3fs (%5.1f%%)\n", stats.cross_entropy, 100.0 * stats.cross_entropy / stats.total_time);
    double backward_pass = stats.bwd_output_grad + stats.bwd_matmul2 + stats.bwd_bias2 + stats.bwd_relu + stats.bwd_matmul1 + stats.bwd_bias1;
    printf("  Backward pass:    %6.3fs (%5.1f%%)\n", backward_pass, 100.0 * backward_pass / stats.total_time);
    printf("  Weight updates:   %6.3fs (%5.1f%%)\n", stats.weight_updates, 100.0 * stats.weight_updates / stats.total_time);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_batch));
    CUDA_CHECK(cudaFree(d_labels_batch));
    free(h_output);
}

// Initialize weights using He initialization (same as v3.c)
void initialize_random_weights(NeuralNetwork *nn) {
    // Create host buffers
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Initialize on host
    initialize_weights(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
    initialize_weights(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    initialize_bias(h_bias1, HIDDEN_SIZE);
    initialize_bias(h_bias2, OUTPUT_SIZE);
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free host buffers
    free(h_weights1);
    free(h_weights2);
    free(h_bias1);
    free(h_bias2);
}

// Initialize neural network with GPU memory
void initialize_neural_network(NeuralNetwork *nn) {
    CUDA_CHECK(cudaMalloc(&nn->weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));

    initialize_random_weights(nn);
}

int main() {
    srand(time(NULL));  // Random seed for natural variance

    NeuralNetwork nn;
    initialize_neural_network(&nn);

    float *X_train = (float *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = (float *)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_SIZE * sizeof(int));

    load_data("./data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("./data/y_train.bin", y_train, TRAIN_SIZE);
    load_data("./data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    normalize_data(X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("./data/y_test.bin", y_test, TEST_SIZE);

    train_timed(&nn, X_train, y_train);

    CUDA_CHECK(cudaFree(nn.weights1));
    CUDA_CHECK(cudaFree(nn.weights2));
    CUDA_CHECK(cudaFree(nn.bias1));
    CUDA_CHECK(cudaFree(nn.bias2));
    CUDA_CHECK(cudaFree(nn.grad_weights1));
    CUDA_CHECK(cudaFree(nn.grad_weights2));
    CUDA_CHECK(cudaFree(nn.grad_bias1));
    CUDA_CHECK(cudaFree(nn.grad_bias2));
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}