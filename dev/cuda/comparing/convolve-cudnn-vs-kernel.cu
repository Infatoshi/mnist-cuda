// compare outputs from CPU, custom kernel and cuDNN
#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cmath> // for fabs

using namespace std;

// CUDA error checking macro
#define CUDA_CHECK(error) \
    if (error != cudaSuccess) { \
        cout << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(-1); \
    }

#define CUDNN_CHECK(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        cout << "cuDNN error: " << cudnnGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(-1); \
    }

const int B = 1;    // Batch size
const int C = 1;    // Number of input channels
const int H = 5;    // Height of input
const int W = 5;    // Width of input
const int K = 1;    // Number of output channels
const int KH = 3;   // Height of kernel
const int KW = 3;   // Width of kernel

// Initialize input and kernel with some values
float h_input[B*C*H*W] = {
    1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0,
    16.0, 17.0, 18.0, 19.0, 20.0,
    21.0, 22.0, 23.0, 24.0, 100.0
};

float h_kernel[K*C*KH*KW] = {
    1.0, 0.0, -1.0,
    1.0, 0.0, -1.0,
    1.0, 0.0, -1.0
};

float h_output_custom[B*K*H*W] = {0}; // Output buffer for custom kernel
float h_output_cudnn[B*K*H*W] = {0};  // Output buffer for cuDNN

// Custom convolution kernel
__global__ void custom_conv2d_kernel(float* input, float* output, float* kernel, int B, int C, int H, int W, int K, int KH, int KW) 
{
    int b = blockIdx.x;
    int k = blockIdx.y;
    int h = threadIdx.x;
    int w = threadIdx.y;
    
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int ih = h + kh;
                int iw = w + kw;
                if (ih < H && iw < W) {
                    sum += input[b * C * H * W + c * H * W + ih * W + iw] * kernel[k * C * KH * KW + c * KH * KW + kh * KW + kw];
                }
            }
        }
    }
    output[b * K * H * W + k * H * W + h * W + w] = sum;
};

void custom_conv2d_kernel_cpu(float* input, float* output, float* kernel, int B, int C, int H, int W, int K, int KH, int KW) 
{
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H-KH+1; ++h) {
                for (int w = 0; w < W-KW+1; ++w) {
                    float sum = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < KH; ++kh) {
                            for (int kw = 0; kw < KW; ++kw) {
                                int ih = h + kh;
                                int iw = w + kw;
                                sum += input[b * C * H * W + c * H * W + ih * W + iw] * kernel[k * C * KH * KW + c * KH * KW + kh * KW + kw];
                            }
                        }
                    }
                    output[b * K * (H-KH+1) * (W-KW+1) + k * (H-KH+1) * (W-KW+1) + h * (W-KW+1) + w] = sum;
                }
            }
        }
    }
}
// Function to print the tensor in a PyTorch-like style
void print_tensor(const float* tensor, int B, int K, int H, int W) {
    for (int b = 0; b < B; ++b) {
        std::cout << "Batch " << b << ": \n";
        for (int k = 0; k < K; ++k) {
            std::cout << "Channel " << k << ": \n";
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    std::cout << tensor[b * K * H * W + k * H * W + h * W + w] << " ";
                }
                std::cout << "\n";
            }
        }
    }
}

int main() {
    // Device buffers
    float *d_input, *d_kernel, *d_output_custom, *d_output_cudnn;
    cudaMalloc(&d_input, B*C*H*W*sizeof(float));
    cudaMalloc(&d_kernel, K*C*KH*KW*sizeof(float));
    cudaMalloc(&d_output_custom, B*K*H*W*sizeof(float));
    cudaMalloc(&d_output_cudnn, B*K*H*W*sizeof(float));

    // Copy input and kernel to device
    cudaMemcpy(d_input, h_input, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, K*C*KH*KW*sizeof(float), cudaMemcpyHostToDevice);

    

    // Kernel launch configuration
    dim3 blockDim(H, W);
    dim3 gridDim(B, K);
    custom_conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_output_custom, d_kernel, B, C, H, W, K, KH, KW);

    // Copy custom kernel output back to host
    cudaMemcpy(h_output_custom, d_output_custom, B*K*H*W*sizeof(float), cudaMemcpyDeviceToHost);

    // cuDNN setup
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, KH, KW);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    int n, c, h, w;
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &n, &c, &h, &w);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &n, &c, &h, &w);

    size_t workspaceSize;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize);

    void* workspace;
    cudaMalloc(&workspace, workspaceSize);

    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_kernel, convDesc, algo, workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));

    cudaMemcpy(h_output_cudnn, d_output_cudnn, B*K*H*W*sizeof(float), cudaMemcpyDeviceToHost);

    // Compare outputs
    bool equal = true;
    for (int i = 0; i < B*K*H*W; ++i) {
        if (fabs(h_output_custom[i] - h_output_cudnn[i]) > 1e-2) {
            cout << "Mismatch at index " << i << ": " << h_output_custom[i] << " != " << h_output_cudnn[i] << endl;
            equal = false;
            break;
        }
    }

    if (equal) {
        cout << "Outputs are the same!" << endl;
    } else {
        cout << "Outputs are different!" << endl;
    }


    // print output for cpu kernel
    custom_conv2d_kernel_cpu(h_input, h_output_custom, h_kernel, B, C, H, W, K, KH, KW);
    cout << "Custom Kernel Output (CPU):\n";
    print_tensor(h_output_custom, B, K, H-KH+1, W-KW+1);
    // Print the outputs for debugging
    std::cout << "Custom Kernel Output (GPU):\n";
    print_tensor(h_output_custom, B, K, H-KH+1, W-KW+1);

    std::cout << "cuDNN Output:\n";
    print_tensor(h_output_cudnn, B, K, H-KH+1, W-KW+1);

    // Cleanup cuDNN resources
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output_custom);
    cudaFree(d_output_cudnn);
    cudaFree(workspace);

    return 0;
}
