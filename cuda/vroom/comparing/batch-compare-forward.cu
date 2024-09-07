#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <iostream>

using namespace std;

#define CHECK_CURAND(call) { \
    curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND error in %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

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

// __device__ float relu(float x) {
//     return fmaxf(x, 0.0f);
// }

// __global__ void matmul_forward_naive(float *A, float *B, float *C, int M, int N, int K) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row < M && col < N) {
//         float sum = 0.0f;
//         for (int i = 0; i < K; i++) {
//             sum += A[row * K + i] * B[i * N + col];
//         }
//         C[row * N + col] = sum;
//     }
// }

// __global__ void forward_pass(float *input, float *weights1, float *bias1, float *hidden,
//                              float *weights2, float *bias2, float *output,
//                              int input_size, int hidden_size, int output_size, int batch_size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int batch_idx = blockIdx.y;

//     if (idx < hidden_size && batch_idx < batch_size) {
//         float sum = 0.0f;
//         for (int i = 0; i < input_size; i++) {
//             sum += weights1[idx * input_size + i] * input[batch_idx * input_size + i];
//         }
//         float hidden_val = relu(sum + bias1[idx]);
//         hidden[batch_idx * hidden_size + idx] = hidden_val;
//     }

//     __syncthreads();

//     if (idx < output_size && batch_idx < batch_size) {
//         float sum = 0.0f;
//         for (int i = 0; i < hidden_size; i++) {
//             sum += weights2[idx * hidden_size + i] * hidden[batch_idx * hidden_size + i];
//         }
//         float output_val = sum + bias2[idx];
//         output[batch_idx * output_size + idx] = output_val;
//     }
// }


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

void cublasMatmul(cublasHandle_t handle, float *d_A, float *d_B, float *d_C, int M, int K, int N) {
    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                        &alpha, d_B, N, d_A, K, &beta, d_C, N));

}

__global__ void forward_pass(float *input, float *weights1, float *bias1, float *hidden,
                             float *weights2, float *bias2, float *output,
                             int input_size, int hidden_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (idx < hidden_size && batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += weights1[idx * input_size + i] * input[batch_idx * input_size + i];
        }
        hidden[batch_idx * hidden_size + idx] = relu(sum + bias1[idx]);
    }

    __syncthreads();

    if (idx < output_size && batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            sum += weights2[idx * hidden_size + i] * hidden[batch_idx * hidden_size + i];
        }
        output[batch_idx * output_size + idx] = sum + bias2[idx];
    }
}

void forward_pass_wrapper(float *d_input, float *d_weights1, float *d_bias1, float *d_hidden,
                          float *d_weights2, float *d_bias2, float *d_output,
                          int input_size, int hidden_size, int output_size, int batch_size) {
    dim3 block_dim(256);
    dim3 grid_dim((max(hidden_size, output_size) + block_dim.x - 1) / block_dim.x, batch_size);

    forward_pass<<<grid_dim, block_dim>>>(d_input, d_weights1, d_bias1, d_hidden,
                                          d_weights2, d_bias2, d_output,
                                          input_size, hidden_size, output_size, batch_size);

    // Print hidden layer values
    float *h_hidden = (float*)malloc(batch_size * hidden_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_hidden, d_hidden, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Forward pass hidden layer values:" << std::endl;
    for (int i = 0; i < batch_size * hidden_size; i++) {
        printf("%f ", h_hidden[i]);
        if ((i+1) % hidden_size == 0) printf("\n");
    }
    free(h_hidden);

    // Print output values
    float *h_output = (float*)malloc(batch_size * output_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Forward pass output values:" << std::endl;
    for (int i = 0; i < batch_size * output_size; i++) {
        printf("%f ", h_output[i]);
        if ((i+1) % output_size == 0) printf("\n");
    }
    free(h_output);
}

void forward_pass_naive(float *input, float *weights1, float *bias1, float *hidden,
                        float *weights2, float *bias2, float *output,
                        int input_size, int hidden_size, int output_size, int batch_size) {
    dim3 block_dim(32, 32);
    dim3 grid_dim_1((hidden_size + block_dim.x - 1) / block_dim.x, (batch_size + block_dim.y - 1) / block_dim.y);
    dim3 grid_dim_2((output_size + block_dim.x - 1) / block_dim.x, (batch_size + block_dim.y - 1) / block_dim.y);

    // First layer: input to hidden
    matmul_forward_naive<<<grid_dim_1, block_dim>>>(input, weights1, hidden, batch_size, hidden_size, input_size);
    
    // Print hidden values after matmul
    float *h_hidden = (float*)malloc(batch_size * hidden_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_hidden, hidden, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Naive hidden values after matmul:" << std::endl;
    for (int i = 0; i < batch_size * hidden_size; i++) {
        printf("%f ", h_hidden[i]);
        if ((i+1) % hidden_size == 0) printf("\n");
    }
    free(h_hidden);

    add_bias_naive<<<grid_dim_1, block_dim>>>(hidden, bias1, batch_size, hidden_size);
    
    // Print hidden values after adding bias
    h_hidden = (float*)malloc(batch_size * hidden_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_hidden, hidden, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Naive hidden values after adding bias:" << std::endl;
    for (int i = 0; i < batch_size * hidden_size; i++) {
        printf("%f ", h_hidden[i]);
        if ((i+1) % hidden_size == 0) printf("\n");
    }
    free(h_hidden);

    apply_relu_naive<<<(batch_size * hidden_size + 255) / 256, 256>>>(hidden, batch_size * hidden_size);
    
    // Print hidden values after ReLU
    h_hidden = (float*)malloc(batch_size * hidden_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_hidden, hidden, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Naive hidden values after ReLU:" << std::endl;
    for (int i = 0; i < batch_size * hidden_size; i++) {
        printf("%f ", h_hidden[i]);
        if ((i+1) % hidden_size == 0) printf("\n");
    }
    free(h_hidden);

    // Second layer: hidden to output
    matmul_forward_naive<<<grid_dim_2, block_dim>>>(hidden, weights2, output, batch_size, output_size, hidden_size);
    add_bias_naive<<<grid_dim_2, block_dim>>>(output, bias2, batch_size, output_size);

    // Print final output
    float *h_output = (float*)malloc(batch_size * output_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Naive final output:" << std::endl;
    for (int i = 0; i < batch_size * output_size; i++) {
        printf("%f ", h_output[i]);
        if ((i+1) % output_size == 0) printf("\n");
    }
    std::cout << std::endl << std::endl;
    free(h_output);
}

void forward_pass_cublas(cublasHandle_t handle, float *input, float *weights1, float *bias1, float *hidden,
                         float *weights2, float *bias2, float *output,
                         int input_size, int hidden_size, int output_size, int batch_size) {
    float alpha = 1.0f, beta = 0.0f;

    // First layer: input to hidden
    cublasMatmul(handle, input, weights1, hidden, batch_size, input_size, hidden_size);
    
    // Print hidden values after matmul
    float *h_hidden = (float*)malloc(batch_size * hidden_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_hidden, hidden, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "cuBLAS hidden values after matmul:" << std::endl;
    for (int i = 0; i < batch_size * hidden_size; i++) {
        printf("%f ", h_hidden[i]);
        if ((i+1) % hidden_size == 0) printf("\n");
    }
    free(h_hidden);

    dim3 block(256);
    dim3 grid((hidden_size + block.x - 1) / block.x, batch_size);
    add_bias_and_relu<<<grid, block>>>(hidden, bias1, hidden_size, batch_size);
    
    // Print hidden values after bias and ReLU
    h_hidden = (float*)malloc(batch_size * hidden_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_hidden, hidden, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "cuBLAS hidden values after bias and ReLU:" << std::endl;
    for (int i = 0; i < batch_size * hidden_size; i++) {
        printf("%f ", h_hidden[i]);
        if ((i+1) % hidden_size == 0) printf("\n");
    }
    free(h_hidden);

    // Second layer: hidden to output
    cublasMatmul(handle, hidden, weights2, output, batch_size, hidden_size, output_size);
    
    grid = dim3((output_size + block.x - 1) / block.x, batch_size);
    add_bias<<<grid, block>>>(output, bias2, output_size, batch_size);

    // Print final output
    float *h_output = (float*)malloc(batch_size * output_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "cuBLAS final output:" << std::endl;
    for (int i = 0; i < batch_size * output_size; i++) {
        printf("%f ", h_output[i]);
        if ((i+1) % output_size == 0) printf("\n");
    }
    free(h_output);
}

void compare_results(float *output1, float *output2, int size) {
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(output1[i] - output2[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
            std::cout << "max_diff_idx: " << max_diff_idx << std::endl;
        }
    }
}

__global__ void scale_array(float *arr, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = (arr[idx] - 0.5f) * scale;
    }
}

int main() {
    const int batch_size = 2; // M -> batch_size
    const int input_size = 4; // K -> 784
    const int hidden_size = 4; // N -> 256
    const int output_size = 1; // O

    size_t input_bytes = batch_size * input_size * sizeof(float); // M * K
    size_t hidden_bytes = batch_size * hidden_size * sizeof(float); // M * N
    size_t output_bytes = batch_size * output_size * sizeof(float); // M * O
    size_t weights1_bytes = input_size * hidden_size * sizeof(float); // K * N
    size_t weights2_bytes = hidden_size * output_size * sizeof(float); // N * O
    size_t bias1_bytes = hidden_size * sizeof(float);
    size_t bias2_bytes = output_size * sizeof(float);

    float *d_input, *d_weights1, *d_bias1, *d_hidden, *d_weights2, *d_bias2, *d_output, *d_output_cublas;

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights1, weights1_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias1, bias1_bytes));
    CHECK_CUDA(cudaMalloc(&d_hidden, hidden_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights2, weights2_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias2, bias2_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_output_cublas, output_bytes));

    float h_input[batch_size * input_size] = {1.0f, -2.0f, 3.0f, -4.0f,
                                              2.0f, -4.0f, 6.0f, -8.0f};

    float h_weights1[input_size * hidden_size] = {-1.0f, 2.0f, 3.0f, 4.0f,
                                                  2.0f, -4.0f, 6.0f, 8.0f,
                                                  3.0f, 6.0f, -9.0f, 12.0f,
                                                  4.0f, 8.0f, 12.0f, -16.0f};

    float h_bias1[hidden_size] = {-1.0f, -2.0f, -3.0f, -4.0f};

    float h_weights2[hidden_size * output_size] = {1.0f, 2.0f, 3.0f, 4.0f};

    float h_bias2[output_size] = {-1.0f};

    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_weights1, h_weights1, weights1_bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_bias1, h_bias1, bias1_bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_weights2, h_weights2, weights2_bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_bias2, h_bias2, bias2_bytes, cudaMemcpyHostToDevice));


    dim3 block(256);
    dim3 grid((max(hidden_size, output_size) + block.x - 1) / block.x, batch_size);
    forward_pass_naive(d_input, d_weights1, d_bias1, d_hidden,
                                d_weights2, d_bias2, d_output,
                                input_size, hidden_size, output_size, batch_size);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    forward_pass_cublas(handle, d_input, d_weights1, d_bias1, d_hidden,
                        d_weights2, d_bias2, d_output_cublas,
                        input_size, hidden_size, output_size, batch_size);
    CHECK_CUBLAS(cublasDestroy(handle));

    float *h_output = (float*)malloc(output_bytes);
    float *h_output_cublas = (float*)malloc(output_bytes);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_cublas, d_output_cublas, output_bytes, cudaMemcpyDeviceToHost));
    // In main()
    forward_pass_wrapper(d_input, d_weights1, d_bias1, d_hidden,
                        d_weights2, d_bias2, d_output,
                        input_size, hidden_size, output_size, batch_size);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    compare_results(h_output, h_output_cublas, batch_size * output_size);

    free(h_output);
    free(h_output_cublas);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weights1));
    CHECK_CUDA(cudaFree(d_bias1));
    CHECK_CUDA(cudaFree(d_hidden));
    CHECK_CUDA(cudaFree(d_weights2));
    CHECK_CUDA(cudaFree(d_bias2));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_output_cublas));

    return 0;
}