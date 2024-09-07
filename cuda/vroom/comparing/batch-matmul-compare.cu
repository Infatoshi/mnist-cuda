#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
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

// Naive matrix multiplication kernel
__global__ void naiveMatmulKernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Function to initialize a matrix with random values
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        // mat[i] = static_cast<float>(rand()) / RAND_MAX;
        // set to i and static cast to float
        mat[i] = static_cast<float>(i) * 0.05;
    }
}

// Function to compare two matrices
bool compareMatrices(float* A, float* B, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        std::cout << "A[" << i << "] = " << A[i] << " B[" << i << "] = " << B[i] << std::endl;
        if (fabs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 4; // batchsize
    const int K = 4; // input size
    const int N = 6; // hidden size

    // (batch_size, input_size) x (input_size, hidden_size) = (batch_size, hidden_size) = (4, 6)


    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A, *h_B, *h_C_naive, *h_C_cublas;
    float *d_A, *d_B, *d_C_naive, *d_C_cublas;

    // Allocate host memory
    h_A = (float*)malloc(bytes_A);
    h_B = (float*)malloc(bytes_B);
    h_C_naive = (float*)malloc(bytes_C);
    h_C_cublas = (float*)malloc(bytes_C);

    // Initialize matrices
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C_naive, bytes_C));
    CHECK_CUDA(cudaMalloc(&d_C_cublas, bytes_C));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // Naive kernel
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    naiveMatmulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C_naive, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f;
    float beta = 0.0f;


    // w @ x -> 
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                            &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N));

    CHECK_CUBLAS(cublasDestroy(handle));

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_C_naive, d_C_naive, bytes_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C_cublas, bytes_C, cudaMemcpyDeviceToHost));

    // Compare results
    bool results_match = compareMatrices(h_C_naive, h_C_cublas, M * N);
    if (results_match) {
        printf("Naive and cuBLAS results match!\n");
    } else {
        printf("Naive and cuBLAS results do not match!\n");
    }

    // print all results
    std::cout << "naive\n";
    for (int i = 0; i < M * K; i++) {
        // std::cout << "naive idx " << i << " = " << h_C_naive[i] << std::endl;
        // std::cout << "cublas idx " << i << " = " << h_C_cublas[i] << std::endl;
        std::cout << h_C_naive[i];
        if (i % M == 0) {
            std::cout << "\n";
        }
        
    }

    std::cout << "\n\n";
    std::cout << "cublas\n";
    for (int i = 0; i < M * K; i++) {
        std::cout << h_C_cublas[i];
        if (i % M == 0) {
            std::cout << "\n";
        }
    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_cublas);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C_naive));
    CHECK_CUDA(cudaFree(d_C_cublas));

    return 0;
}