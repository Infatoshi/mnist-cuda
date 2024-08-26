#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 3
#define LEARNING_RATE 0.001

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

typedef struct {
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
} NeuralNetwork;

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

__global__ void initialize_weights_kernel(float *weights, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init((unsigned long long)clock() + idx, 0, 0, &state);
        weights[idx] = curand_uniform(&state) * scale - (scale / 2.0f);
    }
}

__global__ void initialize_bias_kernel(float *bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        bias[idx] = 0.0f;
    }
}

__global__ void relu_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__device__ float atomicMaxFloat(float* address, float val)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void softmax_kernel(float *x, int size) {
    __shared__ float max_val;
    __shared__ float sum;

    // Find max value
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, x[i]);
    }
    atomicMaxFloat(&max_val, thread_max);
    __syncthreads();

    // Compute exp and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        x[i] = expf(x[i] - max_val);
        thread_sum += x[i];
    }
    atomicAdd(&sum, thread_sum);
    __syncthreads();

    // Normalize
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        x[i] = fmaxf(x[i] / sum, 1e-7f);
    }
}

__global__ void forward_kernel(float *weights1, float *bias1, float *weights2, float *bias2,
                               float *input, float *hidden, float *output,
                               int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < hidden_size) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += weights1[idx * input_size + j] * input[j];
        }
        hidden[idx] = fmaxf(0.0f, sum + bias1[idx]);
    }

    __syncthreads();

    if (idx < output_size) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += weights2[idx * hidden_size + j] * hidden[j];
        }
        output[idx] = sum + bias2[idx];
    }
}

__global__ void backward_kernel(float *weights2, float *input, float *hidden, float *output,
                                int label, float *grad_weights1, float *grad_weights2,
                                float *grad_bias1, float *grad_bias2,
                                int input_size, int hidden_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float grad_output[OUTPUT_SIZE];
    if (idx < output_size) {
        grad_output[idx] = output[idx];
        if (idx == label) grad_output[idx] -= 1.0f;
    }
    __syncthreads();

    if (idx < output_size * hidden_size) {
        int i = idx / hidden_size;
        int j = idx % hidden_size;
        grad_weights2[idx] = grad_output[i] * hidden[j];
    }

    if (idx < output_size) {
        grad_bias2[idx] = grad_output[idx];
    }

    __syncthreads();

    if (idx < hidden_size) {
        float grad_hidden = 0.0f;
        for (int j = 0; j < output_size; j++) {
            grad_hidden += grad_output[j] * weights2[j * hidden_size + idx];
        }
        grad_hidden *= (hidden[idx] > 0);

        for (int j = 0; j < input_size; j++) {
            grad_weights1[idx * input_size + j] = grad_hidden * input[j];
        }
        grad_bias1[idx] = grad_hidden;
    }
}

__global__ void update_weights_kernel(float *weights, float *grad_weights, int size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad_weights[idx];
    }
}

void train(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    float *d_input, *d_hidden, *d_output;
    float *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;
    int *d_label;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_label, sizeof(int)));

    // Copy weights and biases to device
    CUDA_CHECK(cudaMemcpy(d_weights1, nn->weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights2, nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias1, nn->bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias2, nn->bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;


        for (int i = 0; i < TRAIN_SIZE; i++) {
            clock_t start = clock();
            double cpu_time_used;

            // Copy input and label to device
            CUDA_CHECK(cudaMemcpy(d_input, &X_train[i * INPUT_SIZE], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_label, &y_train[i], sizeof(int), cudaMemcpyHostToDevice));

            // Forward pass
            grid_size = (max(HIDDEN_SIZE, OUTPUT_SIZE) + block_size - 1) / block_size;
            forward_kernel<<<grid_size, block_size>>>(d_weights1, d_bias1, d_weights2, d_bias2,
                                                      d_input, d_hidden, d_output,
                                                      INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

            // Softmax
            softmax_kernel<<<1, OUTPUT_SIZE>>>(d_output, OUTPUT_SIZE);

            // Compute loss and accuracy
            float output[OUTPUT_SIZE];
            CUDA_CHECK(cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            float loss = -logf(fmaxf(output[y_train[i]], 1e-7f));
            total_loss += loss;

            int predicted = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[predicted]) {
                    predicted = j;
                }
            }
            if (predicted == y_train[i]) {
                correct++;
            }

            // Backward pass
            grid_size = (max(HIDDEN_SIZE * INPUT_SIZE, OUTPUT_SIZE * HIDDEN_SIZE) + block_size - 1) / block_size;
            backward_kernel<<<grid_size, block_size>>>(d_weights2, d_input, d_hidden, d_output,
                                                       y_train[i], d_grad_weights1, d_grad_weights2,
                                                       d_grad_bias1, d_grad_bias2,
                                                       INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

            // Update weights
            grid_size = (HIDDEN_SIZE * INPUT_SIZE + block_size - 1) / block_size;
            update_weights_kernel<<<grid_size, block_size>>>(d_weights1, d_grad_weights1, HIDDEN_SIZE * INPUT_SIZE, LEARNING_RATE);
            
            grid_size = (OUTPUT_SIZE * HIDDEN_SIZE + block_size - 1) / block_size;
            update_weights_kernel<<<grid_size, block_size>>>(d_weights2, d_grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE, LEARNING_RATE);
            
            grid_size = (HIDDEN_SIZE + block_size - 1) / block_size;
            update_weights_kernel<<<grid_size, block_size>>>(d_bias1, d_grad_bias1, HIDDEN_SIZE, LEARNING_RATE);
            
            grid_size = (OUTPUT_SIZE + block_size - 1) / block_size;
            update_weights_kernel<<<grid_size, block_size>>>(d_bias2, d_grad_bias2, OUTPUT_SIZE, LEARNING_RATE);

            clock_t end = clock();
            cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;

            if ((i + 1) % 1000 == 0 || (epoch == 0 && i == 0)) {
                printf("Epoch %d/%d, Step %d/%d, Loss: %.4f, Accuracy: %.2f%%, Time: %.2f\n", 
                       epoch + 1, EPOCHS, i + 1, TRAIN_SIZE, total_loss / (i + 1), 
                       100.0f * correct / (i + 1), cpu_time_used);
                cpu_time_used = 0.0;
            }
        }

        printf("Epoch %d/%d completed, Loss: %.4f, Accuracy: %.2f%%\n", 
               epoch + 1, EPOCHS, total_loss / TRAIN_SIZE, 100.0f * correct / TRAIN_SIZE);
    }

    // print out the first 10 images along with predicted labels in the terminal
    for (int i = 0; i < 16; i++) {
        // Copy input and label to device
        CUDA_CHECK(cudaMemcpy(d_input, &X_train[i * INPUT_SIZE], INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_label, &y_train[i], sizeof(int), cudaMemcpyHostToDevice));

        // Forward pass
        grid_size = (max(HIDDEN_SIZE, OUTPUT_SIZE) + block_size - 1) / block_size;
        forward_kernel<<<grid_size, block_size>>>(d_weights1, d_bias1, d_weights2, d_bias2,
                                                  d_input, d_hidden, d_output,
                                                  INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

        // Softmax
        softmax_kernel<<<1, OUTPUT_SIZE>>>(d_output, OUTPUT_SIZE);

        // Copy output back to host
        float output[OUTPUT_SIZE];
        CUDA_CHECK(cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

        // Print image
        printf("Image %d:\n", i);
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                if (X_train[i * INPUT_SIZE + j * 28 + k] > 0.0f) {
                    printf("X");
                } else {
                    printf(" ");
                }
            }
            printf("\n");
        }

        // Print predicted label
        int predicted = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[predicted]) {
                predicted = j;
            }
        }
        printf("Predicted label: %d\n", predicted);
    }

    // Copy weights and biases back to host
    CUDA_CHECK(cudaMemcpy(nn->weights1, d_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn->weights2, d_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn->bias1, d_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn->bias2, d_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_weights1);
    cudaFree(d_weights2);
    cudaFree(d_bias1);
    cudaFree(d_bias2);
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_grad_weights1);
    cudaFree(d_grad_weights2);
    cudaFree(d_grad_bias1);
    cudaFree(d_grad_bias2);
    cudaFree(d_label);
}

int main() {
    srand(time(NULL));

    NeuralNetwork nn;
    nn.weights1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn.weights2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn.bias1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    nn.bias2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    CUDA_CHECK(cudaMalloc(&d_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias2, OUTPUT_SIZE * sizeof(float)));

    int block_size = 256;
    int grid_size;

    grid_size = (HIDDEN_SIZE * INPUT_SIZE + block_size - 1) / block_size;
    initialize_weights_kernel<<<grid_size, block_size>>>(d_weights1, HIDDEN_SIZE * INPUT_SIZE, sqrtf(2.0f / INPUT_SIZE));
    
    grid_size = (OUTPUT_SIZE * HIDDEN_SIZE + block_size - 1) / block_size;
    initialize_weights_kernel<<<grid_size, block_size>>>(d_weights2, OUTPUT_SIZE * HIDDEN_SIZE, sqrtf(2.0f / HIDDEN_SIZE));
    
    grid_size = (HIDDEN_SIZE + block_size - 1) / block_size;
    initialize_bias_kernel<<<grid_size, block_size>>>(d_bias1, HIDDEN_SIZE);
    
    grid_size = (OUTPUT_SIZE + block_size - 1) / block_size;
    initialize_bias_kernel<<<grid_size, block_size>>>(d_bias2, OUTPUT_SIZE);

    CUDA_CHECK(cudaMemcpy(nn.weights1, d_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn.weights2, d_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn.bias1, d_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn.bias2, d_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_weights1);
    cudaFree(d_weights2);
    cudaFree(d_bias1);
    cudaFree(d_bias2);

    float *X_train = (float*)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int*)malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = (float*)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int*)malloc(TEST_SIZE * sizeof(int));

    load_data("../mnist_data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("../mnist_data/y_train.bin", y_train, TRAIN_SIZE);
    load_data("../mnist_data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("../mnist_data/y_test.bin", y_test, TEST_SIZE);

    // Print first image in the terminal
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (X_train[0 * INPUT_SIZE + i * 28 + j] > 0.0f) {
                printf("X");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }

    printf("First 10 training labels: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", y_train[i]);
    }
    printf("\n");

    train(&nn, X_train, y_train);

    free(nn.weights1);
    free(nn.weights2);
    free(nn.bias1);
    free(nn.bias2);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}