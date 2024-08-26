#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__device__ float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
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

__global__ void compute_hidden_gradient(float *hidden, float *weights2, float *grad_output, float *grad_hidden,
                                        int hidden_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx < hidden_size && batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < output_size; i++) {
            sum += grad_output[batch_idx * output_size + i] * weights2[i * hidden_size + idx];
        }
        int hidden_index = batch_idx * hidden_size + idx;
        grad_hidden[hidden_index] = sum * relu_derivative(hidden[hidden_index]);
    }
}

__global__ void compute_weight_gradients(float *input, float *grad_hidden, float *grad_weights1,
                                         int input_size, int hidden_size, int batch_size) {
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int input_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (hidden_idx < hidden_size && input_idx < input_size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad_hidden[b * hidden_size + hidden_idx] * input[b * input_size + input_idx];
        }
        grad_weights1[hidden_idx * input_size + input_idx] = sum;
    }
}

__global__ void compute_bias_gradients(float *grad_hidden, float *grad_bias1, int hidden_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < hidden_size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad_hidden[b * hidden_size + idx];
        }
        grad_bias1[idx] = sum;
    }
}

__device__ float relu(float x) {
    return fmaxf(x, 0.0f);
}

__global__ void add_bias_and_relu(float *data, float *bias, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx < size && batch_idx < batch_size) {
        int index = batch_idx * size + idx;
        data[index] = relu(data[index] + bias[idx]);
    }
}

__global__ void add_bias(float *data, float *bias, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx < size && batch_idx < batch_size) {
        int index = batch_idx * size + idx;
        data[index] += bias[idx];
    }
}


__global__ void matmul_forward_naive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void add_bias_naive(float *input, float *bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        input[row * cols + col] += bias[col];
    }
}

__global__ void apply_relu_naive(float *input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        input[idx] = relu(input[idx]);
    }
}


void forward_pass_naive(float *input, float *weights1, float *bias1, float *hidden,
                        float *weights2, float *bias2, float *output,
                        int input_size, int hidden_size, int output_size, int batch_size) {
    // Define grid and block dimensions
    dim3 block_dim(32, 32);
    dim3 grid_dim_1((hidden_size + block_dim.x - 1) / block_dim.x, (batch_size + block_dim.y - 1) / block_dim.y);
    dim3 grid_dim_2((output_size + block_dim.x - 1) / block_dim.x, (batch_size + block_dim.y - 1) / block_dim.y);


    // // print inputs
    // float *h_input = (float*)malloc(batch_size * input_size * sizeof(float));
    // CHECK_CUDA(cudaMemcpy(h_input, input, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "input to naive: " << std::endl;
    // for (int i = 0; i < batch_size * input_size; i++) {
    //     printf("%f ", h_input[i]);
    //     if ((i+1) % input_size == 0) {
    //         printf("\n");
    //     }
    // }
    // // copy back to device
    // CHECK_CUDA(cudaMemcpy(input, h_input, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));

    // First layer: input to hidden
    matmul_forward_naive<<<grid_dim_1, block_dim>>>(input, weights1, hidden, batch_size, hidden_size, input_size);

    // print "hidden" values
    // copy hidden to host
    // float *h_hidden = (float*)malloc(batch_size * hidden_size * sizeof(float));
    // CHECK_CUDA(cudaMemcpy(h_hidden, hidden, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "naive hidden values (no bias): " << std::endl;
    // for (int i = 0; i < batch_size * hidden_size; i++) {
    //     printf("%f ", h_hidden[i]);
    //     if ((i+1) % hidden_size == 0) {
    //         printf("\n");
    //     }
    // }
    // // copy back to device
    // CHECK_CUDA(cudaMemcpy(hidden, h_hidden, batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));


    add_bias_naive<<<grid_dim_1, block_dim>>>(hidden, bias1, batch_size, hidden_size);
    apply_relu_naive<<<(batch_size * hidden_size + 255) / 256, 256>>>(hidden, batch_size * hidden_size);

    // Second layer: hidden to output
    matmul_forward_naive<<<grid_dim_2, block_dim>>>(hidden, weights2, output, batch_size, output_size, hidden_size);
    add_bias_naive<<<grid_dim_2, block_dim>>>(output, bias2, batch_size, output_size);
}

void backward_pass_naive(float *input, float *hidden, float *output, int *labels,
                         float *weights1, float *weights2,
                         float *grad_weights1, float *grad_weights2,
                         float *grad_bias1, float *grad_bias2,
                         int input_size, int hidden_size, int output_size, int batch_size) {
                            
    float *d_grad_output, *d_grad_hidden;
    CHECK_CUDA(cudaMalloc(&d_grad_output, batch_size * output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_hidden, batch_size * hidden_size * sizeof(float)));

    dim3 block(256);
    dim3 grid_output((output_size + block.x - 1) / block.x, batch_size);
    dim3 grid_hidden((hidden_size + block.x - 1) / block.x, batch_size);

    compute_output_gradient<<<grid_output, block>>>(output, labels, d_grad_output, output_size, batch_size);
    compute_hidden_gradient<<<grid_hidden, block>>>(hidden, weights2, d_grad_output, d_grad_hidden, hidden_size, output_size, batch_size);

    dim3 block_weights(16, 16);
    dim3 grid_weights((hidden_size + block_weights.x - 1) / block_weights.x,
                      (input_size + block_weights.y - 1) / block_weights.y);
    compute_weight_gradients<<<grid_weights, block_weights>>>(input, d_grad_hidden, grad_weights1, input_size, hidden_size, batch_size);

    compute_bias_gradients<<<(hidden_size + 255) / 256, 256>>>(d_grad_hidden, grad_bias1, hidden_size, batch_size);

    // For grad_weights2 and grad_bias2, we can reuse the existing kernels with different dimensions
    dim3 grid_weights2((output_size + block_weights.x - 1) / block_weights.x,
                       (hidden_size + block_weights.y - 1) / block_weights.y);
    compute_weight_gradients<<<grid_weights2, block_weights>>>(hidden, d_grad_output, grad_weights2, hidden_size, output_size, batch_size);

    compute_bias_gradients<<<(output_size + 255) / 256, 256>>>(d_grad_output, grad_bias2, output_size, batch_size);

    CHECK_CUDA(cudaFree(d_grad_output));
    CHECK_CUDA(cudaFree(d_grad_hidden));
}

void backward_pass_cublas(cublasHandle_t handle,
                          float *input, float *hidden, float *output, int *labels,
                          float *weights1, float *weights2,
                          float *grad_weights1, float *grad_weights2,
                          float *grad_bias1, float *grad_bias2,
                          int input_size, int hidden_size, int output_size, int batch_size) {
    float *d_grad_output, *d_grad_hidden;
    CHECK_CUDA(cudaMalloc(&d_grad_output, batch_size * output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_hidden, batch_size * hidden_size * sizeof(float)));

    dim3 block(256);
    dim3 grid_output((output_size + block.x - 1) / block.x, batch_size);
    dim3 grid_hidden((hidden_size + block.x - 1) / block.x, batch_size);

    compute_output_gradient<<<grid_output, block>>>(output, labels, d_grad_output, output_size, batch_size);

    // Compute grad_hidden using cuBLAS
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             hidden_size, batch_size, output_size,
                             &alpha, weights2, output_size,
                             d_grad_output, output_size,
                             &beta, d_grad_hidden, hidden_size));

    // Apply ReLU derivative
    dim3 block_relu(256);
    dim3 grid_relu((batch_size * hidden_size + block_relu.x - 1) / block_relu.x);
    compute_hidden_gradient<<<grid_hidden, block>>>(hidden, weights2, d_grad_output, d_grad_hidden, hidden_size, output_size, batch_size);

    // Compute grad_weights1 using cuBLAS
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             input_size, hidden_size, batch_size,
                             &alpha, input, input_size,
                             d_grad_hidden, hidden_size,
                             &beta, grad_weights1, input_size));

    // Compute grad_bias1
    compute_bias_gradients<<<(hidden_size + 255) / 256, 256>>>(d_grad_hidden, grad_bias1, hidden_size, batch_size);

    // Compute grad_weights2 using cuBLAS
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             hidden_size, output_size, batch_size,
                             &alpha, hidden, hidden_size,
                             d_grad_output, output_size,
                             &beta, grad_weights2, hidden_size));

    // Compute grad_bias2
    compute_bias_gradients<<<(output_size + 255) / 256, 256>>>(d_grad_output, grad_bias2, output_size, batch_size);

    CHECK_CUDA(cudaFree(d_grad_output));
    CHECK_CUDA(cudaFree(d_grad_hidden));
}

void compare_results(float *output1, float *output2, int size) {
    float eps = 1e-5f;
    int max_diff_idx = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(output1[i] - output2[i]);
        if (diff > eps) {
            printf("Results differ at index %d: %f vs %f\n", i, output1[i], output2[i]);
            max_diff_idx = i;
            break;
        }
    }
}


int main() {
    const int batch_size = 2;
    const int input_size = 4;
    const int hidden_size = 4;
    const int output_size = 1;

    size_t input_bytes = batch_size * input_size * sizeof(float);
    size_t hidden_bytes = batch_size * hidden_size * sizeof(float);
    size_t output_bytes = batch_size * output_size * sizeof(float);
    size_t weights1_bytes = input_size * hidden_size * sizeof(float);
    size_t weights2_bytes = hidden_size * output_size * sizeof(float);
    size_t bias1_bytes = hidden_size * sizeof(float);
    size_t bias2_bytes = output_size * sizeof(float);

    float *d_input, *d_hidden, *d_output, *d_output_cublas;
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    float *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;
    float *d_grad_weights1_cublas, *d_grad_weights2_cublas, *d_grad_bias1_cublas, *d_grad_bias2_cublas;
    int *d_labels;

    // Allocate memory
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_hidden, hidden_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_output_cublas, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights1, weights1_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights2, weights2_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias1, bias1_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias2, bias2_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_weights1, weights1_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_weights2, weights2_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_bias1, bias1_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_bias2, bias2_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_weights1_cublas, weights1_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_weights2_cublas, weights2_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_bias1_cublas, bias1_bytes));
    CHECK_CUDA(cudaMalloc(&d_grad_bias2_cublas, bias2_bytes));
    CHECK_CUDA(cudaMalloc(&d_labels, batch_size * sizeof(int)));

    // Initialize data
    float h_input[batch_size * input_size] = {1.0f, 2.0f, 3.0f, 4.0f,
                                              2.0f, 4.0f, 6.0f, 8.0f};
    float h_weights1[input_size * hidden_size] = {1.0f, 2.0f, 3.0f, 4.0f,
                                                  2.0f, 4.0f, 6.0f, 8.0f,
                                                  3.0f, 6.0f, 9.0f, 12.0f,
                                                  4.0f, 8.0f, 12.0f, 16.0f};
    float h_bias1[hidden_size] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_weights2[hidden_size * output_size] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_bias2[output_size] = {1.0f};
    int h_labels[batch_size] = {0, 0}; // Assuming binary classification

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights1, h_weights1, weights1_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias1, h_bias1, bias1_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights2, h_weights2, weights2_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias2, h_bias2, bias2_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, h_labels, batch_size * sizeof(int), cudaMemcpyHostToDevice));

    // Forward pass (naive)
    forward_pass_naive(d_input, d_weights1, d_bias1, d_hidden,
                       d_weights2, d_bias2, d_output,
                       input_size, hidden_size, output_size, batch_size);

    // Forward pass (cuBLAS)
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    // forward_pass_cublas(handle, d_input, d_weights1, d_bias1, d_hidden,
    //                     d_weights2, d_bias2, d_output_cublas,
    //                     input_size, hidden_size, output_size, batch_size);

    // Backward pass (naive)
    backward_pass_naive(d_input, d_hidden, d_output, d_labels,
                        d_weights1, d_weights2,
                        d_grad_weights1, d_grad_weights2,
                        d_grad_bias1, d_grad_bias2,
                        input_size, hidden_size, output_size, batch_size);

    // Backward pass (cuBLAS)
    backward_pass_cublas(handle, d_input, d_hidden, d_output_cublas, d_labels,
                         d_weights1, d_weights2,
                         d_grad_weights1_cublas, d_grad_weights2_cublas,
                         d_grad_bias1_cublas, d_grad_bias2_cublas,
                         input_size, hidden_size, output_size, batch_size);

    CHECK_CUBLAS(cublasDestroy(handle));

    // Compare results
    float *h_output = (float*)malloc(output_bytes);
    float *h_output_cublas = (float*)malloc(output_bytes);
    float *h_grad_weights1 = (float*)malloc(weights1_bytes);
    float *h_grad_weights1_cublas = (float*)malloc(weights1_bytes);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_cublas, d_output_cublas, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_grad_weights1, d_grad_weights1, weights1_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_grad_weights1_cublas, d_grad_weights1_cublas, weights1_bytes, cudaMemcpyDeviceToHost));

    printf("Comparing forward pass results:\n");
    compare_results(h_output, h_output_cublas, batch_size * output_size);

    printf("Comparing backward pass results (grad_weights1):\n");
    compare_results(h_grad_weights1, h_grad_weights1_cublas, input_size * hidden_size);

    // Free memory
    free(h_output);
    free(h_output_cublas);
    free(h_grad_weights1);
    free(h_grad_weights1_cublas);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_output_cublas));
    CHECK_CUDA(cudaFree(d_weights1));
    CHECK_CUDA(cudaFree(d_weights2));
    CHECK_CUDA(cudaFree(d_bias1));
    CHECK_CUDA(cudaFree(d_bias2));
    CHECK_CUDA(cudaFree(d_grad_weights1));
    CHECK_CUDA(cudaFree(d_grad_weights2));
    CHECK_CUDA(cudaFree(d_grad_bias1));
    CHECK_CUDA(cudaFree(d_grad_bias2));
    CHECK_CUDA(cudaFree(d_grad_weights1_cublas));
    CHECK_CUDA(cudaFree(d_grad_weights2_cublas));
    CHECK_CUDA(cudaFree(d_grad_bias1_cublas));
    CHECK_CUDA(cudaFree(d_grad_bias2_cublas));
    CHECK_CUDA(cudaFree(d_labels));

    return 0;
}