#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 4

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void init_random(float *data, int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

__global__ void relu_derivative(float *grad, float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] *= (x[idx] > 0) ? 1.0f : 0.0f;
    }
}


__global__ void backward_pass_naive(float *input, float *hidden, float *output, int *labels,
                                    float *weights1, float *weights2,
                                    float *grad_weights1, float *grad_weights2,
                                    float *grad_bias1, float *grad_bias2,
                                    int input_size, int hidden_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    __shared__ float grad_output[OUTPUT_SIZE];

    if (idx < output_size && batch_idx < batch_size) {
        grad_output[idx] = output[batch_idx * output_size + idx];
        if (idx == labels[batch_idx]) {
            grad_output[idx] -= 1.0f;
        }
    }

    __syncthreads();

    if (idx < hidden_size && batch_idx < batch_size) {
        float grad_hidden = 0.0f;
        for (int i = 0; i < output_size; i++) {
            grad_hidden += grad_output[i] * weights2[i * hidden_size + idx];
        }
        grad_hidden *= (hidden[batch_idx * hidden_size + idx] > 0) ? 1.0f : 0.0f;  // ReLU derivative

        for (int i = 0; i < input_size; i++) {
            atomicAdd(&grad_weights1[idx * input_size + i], grad_hidden * input[batch_idx * input_size + i]);
        }
        atomicAdd(&grad_bias1[idx], grad_hidden);
    }

    if (idx < output_size * hidden_size && batch_idx < batch_size) {
        int i = idx / hidden_size;
        int j = idx % hidden_size;
        atomicAdd(&grad_weights2[idx], grad_output[i] * hidden[batch_idx * hidden_size + j]);
    }

    if (idx < output_size && batch_idx < batch_size) {
        atomicAdd(&grad_bias2[idx], grad_output[idx]);
    }
}

__global__ void compute_output_gradient(float *output, int *labels, float *grad_output, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx < output_size && batch_idx < batch_size) {
        int index = batch_idx * output_size + idx;
        grad_output[index] = output[index];
        if (idx == labels[batch_idx]) {
            grad_output[index] -= 1.0f;
        }
    }
}

__global__ void compute_hidden_gradient(float *grad_hidden, float *grad_output, float *weights2, float *hidden,
                                        int hidden_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx < hidden_size && batch_idx < batch_size) {
        float grad = 0.0f;
        for (int i = 0; i < output_size; i++) {
            grad += grad_output[batch_idx * output_size + i] * weights2[i * hidden_size + idx];
        }
        grad_hidden[batch_idx * hidden_size + idx] = grad * ((hidden[batch_idx * hidden_size + idx] > 0) ? 1.0f : 0.0f);
    }
}

__global__ void compute_bias_gradient(float *grad_bias, float *grad, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += grad[i * size + idx];
        }
        grad_bias[idx] = sum;
    }
}

// 3/4 working
void backward_pass_cublas(cublasHandle_t handle, float *d_input, float *d_hidden, float *d_output, int *d_labels,
                          float *d_weights1, float *d_weights2,
                          float *d_grad_weights1, float *d_grad_weights2,
                          float *d_grad_bias1, float *d_grad_bias2,
                          float *d_grad_output, float *d_grad_hidden, float *d_ones,
                          int input_size, int hidden_size, int output_size, int batch_size) {
    float alpha = 1.0f, beta = 0.0f;

    // Compute output gradient
    dim3 block_size(256);
    dim3 grid_size((output_size + block_size.x - 1) / block_size.x, batch_size);
    compute_output_gradient<<<grid_size, block_size>>>(d_output, d_labels, d_grad_output, output_size, batch_size);

    // Compute dW2 = dLoss @ x2.T = (10, B) @ (B, 256) = (10, 256)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                hidden_size, output_size, batch_size, // (M K N)
                &alpha,
                d_hidden, hidden_size,
                d_grad_output, output_size,
                &beta,
                d_grad_weights2, hidden_size);

    // Compute hidden gradient
    grid_size.x = (hidden_size + block_size.x - 1) / block_size.x;
    compute_hidden_gradient<<<grid_size, block_size>>>(d_grad_hidden, d_grad_output, d_weights2, d_hidden,
                                                       hidden_size, output_size, batch_size);

    // Compute dW1 = dRelu @ x1.T = (256, B) @ (B, 784) = (256, 784)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                input_size, hidden_size, batch_size,
                &alpha,
                d_input, input_size,
                d_grad_hidden, hidden_size,
                &beta,
                d_grad_weights1, input_size);

    // Compute bias gradients
    compute_bias_gradient<<<(output_size + 255) / 256, 256>>>(d_grad_bias2, d_grad_output, output_size, batch_size);
    compute_bias_gradient<<<(hidden_size + 255) / 256, 256>>>(d_grad_bias1, d_grad_hidden, hidden_size, batch_size);
}

void print_comparison(const char* name, float* arr1, float* arr2, int size) {
    float max_diff = 0.0f;
    printf("%s:\n", name);
    printf("First 10 values:\n");
    for (int i = 0; i < 10 && i < size; i++) {
        printf("%.6f vs %.6f\n", arr1[i], arr2[i]);
        max_diff = fmaxf(max_diff, fabsf(arr1[i] - arr2[i]));
    }
    for (int i = 10; i < size; i++) {
        max_diff = fmaxf(max_diff, fabsf(arr1[i] - arr2[i]));
    }
    printf("Max difference: %.6f\n\n", max_diff);
}

int main() {
    // Allocate host memory
    float *h_input, *h_hidden, *h_output, *h_weights1, *h_weights2;
    int *h_labels;
    float *h_grad_weights1_naive, *h_grad_weights2_naive, *h_grad_bias1_naive, *h_grad_bias2_naive;
    float *h_grad_weights1_cublas, *h_grad_weights2_cublas, *h_grad_bias1_cublas, *h_grad_bias2_cublas;

    cudaMallocHost(&h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMallocHost(&h_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMallocHost(&h_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMallocHost(&h_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMallocHost(&h_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMallocHost(&h_labels, BATCH_SIZE * sizeof(int));
    cudaMallocHost(&h_grad_weights1_naive, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMallocHost(&h_grad_weights2_naive, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMallocHost(&h_grad_bias1_naive, HIDDEN_SIZE * sizeof(float));
    cudaMallocHost(&h_grad_bias2_naive, OUTPUT_SIZE * sizeof(float));
    cudaMallocHost(&h_grad_weights1_cublas, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMallocHost(&h_grad_weights2_cublas, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMallocHost(&h_grad_bias1_cublas, HIDDEN_SIZE * sizeof(float));
    cudaMallocHost(&h_grad_bias2_cublas, OUTPUT_SIZE * sizeof(float));

    // Allocate device memory
    float *d_input, *d_hidden, *d_output, *d_weights1, *d_weights2;
    int *d_labels;
    float *d_grad_weights1_naive, *d_grad_weights2_naive, *d_grad_bias1_naive, *d_grad_bias2_naive;
    float *d_grad_weights1_cublas, *d_grad_weights2_cublas, *d_grad_bias1_cublas, *d_grad_bias2_cublas;
    float *d_grad_output, *d_grad_hidden, *d_ones;

    CUDA_CHECK(cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights1_naive, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights2_naive, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias1_naive, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias2_naive, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights1_cublas, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights2_cublas, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias1_cublas, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias2_cublas, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones, BATCH_SIZE * sizeof(float)));

    // Initialize random data
    int threads = 256;
    int blocks;
    unsigned long long seed = time(NULL);

    blocks = (BATCH_SIZE * INPUT_SIZE + threads - 1) / threads;
    init_random<<<blocks, threads>>>(d_input, BATCH_SIZE * INPUT_SIZE, seed);

    blocks = (BATCH_SIZE * HIDDEN_SIZE + threads - 1) / threads;
    init_random<<<blocks, threads>>>(d_hidden, BATCH_SIZE * HIDDEN_SIZE, seed);

    blocks = (BATCH_SIZE * OUTPUT_SIZE + threads - 1) / threads;
    init_random<<<blocks, threads>>>(d_output, BATCH_SIZE * OUTPUT_SIZE, seed);

    blocks = (HIDDEN_SIZE * INPUT_SIZE + threads - 1) / threads;
    init_random<<<blocks, threads>>>(d_weights1, HIDDEN_SIZE * INPUT_SIZE, seed);

    blocks = (OUTPUT_SIZE * HIDDEN_SIZE + threads - 1) / threads;
    init_random<<<blocks, threads>>>(d_weights2, OUTPUT_SIZE * HIDDEN_SIZE, seed);

    // Initialize labels with random values between 0 and OUTPUT_SIZE - 1
    for (int i = 0; i < BATCH_SIZE; i++) {
        h_labels[i] = rand() % OUTPUT_SIZE;
    }
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // Initialize d_ones with all 1's
    CUDA_CHECK(cudaMemset(d_ones, 1, BATCH_SIZE * sizeof(float)));

    // Allocate host memory for grad_output
    float *h_grad_output_naive, *h_grad_output_cublas;
    cudaMallocHost(&h_grad_output_naive, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMallocHost(&h_grad_output_cublas, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    // Allocate device memory for grad_output_naive
    float *d_grad_output_naive;
    CUDA_CHECK(cudaMalloc(&d_grad_output_naive, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Perform naive backward pass
    dim3 block_size(256);
    dim3 grid_size((max(HIDDEN_SIZE, OUTPUT_SIZE) + block_size.x - 1) / block_size.x, BATCH_SIZE);
    backward_pass_naive<<<grid_size, block_size>>>(d_input, d_hidden, d_output, d_labels,
                                                   d_weights1, d_weights2,
                                                   d_grad_weights1_naive, d_grad_weights2_naive,
                                                   d_grad_bias1_naive, d_grad_bias2_naive,
                                                   INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);

    // Compute grad_output for naive approach
    compute_output_gradient<<<grid_size, block_size>>>(d_output, d_labels, d_grad_output_naive, OUTPUT_SIZE, BATCH_SIZE);

    // Perform cuBLAS backward pass
    cublasHandle_t handle;
    cublasCreate(&handle);
    backward_pass_cublas(handle, d_input, d_hidden, d_output, d_labels,
                         d_weights1, d_weights2,
                         d_grad_weights1_cublas, d_grad_weights2_cublas,
                         d_grad_bias1_cublas, d_grad_bias2_cublas,
                         d_grad_output, d_grad_hidden, d_ones,
                         INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);
    cublasDestroy(handle);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_grad_weights1_naive, d_grad_weights1_naive, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_weights2_naive, d_grad_weights2_naive, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_bias1_naive, d_grad_bias1_naive, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_bias2_naive, d_grad_bias2_naive, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_grad_weights1_cublas, d_grad_weights1_cublas, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_weights2_cublas, d_grad_weights2_cublas, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_bias1_cublas, d_grad_bias1_cublas, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_bias2_cublas, d_grad_bias2_cublas, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_grad_output_naive, d_grad_output_naive, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_output_cublas, d_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare and print results
    print_comparison("grad_output", h_grad_output_naive, h_grad_output_cublas, BATCH_SIZE * OUTPUT_SIZE);
    print_comparison("grad_weights2", h_grad_weights2_naive, h_grad_weights2_cublas, OUTPUT_SIZE * HIDDEN_SIZE);

    // print indices of x > 1e-2 here (h_grad_weights2_naive, h_grad_weights2_cublas):
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i += 16) {
        if (fabsf(h_grad_weights2_naive[i] - h_grad_weights2_cublas[i]) > 1e-3) {   
            printf("Index %d: %.6f vs %.6f\n", i, h_grad_weights2_naive[i], h_grad_weights2_cublas[i]);
        }
    }

    print_comparison("grad_bias2", h_grad_bias2_naive, h_grad_bias2_cublas, OUTPUT_SIZE);
    print_comparison("grad_bias2", h_grad_bias2_naive, h_grad_bias2_cublas, OUTPUT_SIZE);
    print_comparison("grad_weights1", h_grad_weights1_naive, h_grad_weights1_cublas, HIDDEN_SIZE * INPUT_SIZE);
    print_comparison("grad_bias1", h_grad_bias1_naive, h_grad_bias1_cublas, HIDDEN_SIZE);

    // Free memory
    cudaFreeHost(h_input);
    cudaFreeHost(h_hidden);
    cudaFreeHost(h_output);
    cudaFreeHost(h_weights1);
    cudaFreeHost(h_weights2);
    cudaFreeHost(h_labels);
    cudaFreeHost(h_grad_weights1_naive);
    cudaFreeHost(h_grad_weights2_naive);
    cudaFreeHost(h_grad_bias1_naive);
    cudaFreeHost(h_grad_bias2_naive);
    cudaFreeHost(h_grad_weights1_cublas);
    cudaFreeHost(h_grad_weights2_cublas);
    cudaFreeHost(h_grad_bias1_cublas);
    cudaFreeHost(h_grad_bias2_cublas);
    cudaFreeHost(h_grad_output_naive);
    cudaFreeHost(h_grad_output_cublas);

    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_weights1);
    cudaFree(d_weights2);
    cudaFree(d_labels);
    cudaFree(d_grad_weights1_naive);
    cudaFree(d_grad_weights2_naive);
    cudaFree(d_grad_bias1_naive);
    cudaFree(d_grad_bias2_naive);
    cudaFree(d_grad_weights1_cublas);
    cudaFree(d_grad_weights2_cublas);
    cudaFree(d_grad_bias1_cublas);
    cudaFree(d_grad_bias2_cublas);
    cudaFree(d_grad_output);
    cudaFree(d_grad_hidden);
    cudaFree(d_ones);
    cudaFree(d_grad_output_naive);

    return 0;
}
