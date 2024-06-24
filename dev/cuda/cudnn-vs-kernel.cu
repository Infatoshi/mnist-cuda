#include "dataloader/dataloader.cuh"
#include <iostream>
#include <cudnn.h>
#include <chrono>
#include <cmath>

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

// Custom CUDA kernel for "valid" 2D convolution -> output_size = input_size - kernel_size + 1
__global__ void custom_conv2d_kernel(float* input, float* output, float* kernel, int B, int C, int H, int W, int K, int KH, int KW) {
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
                    sum += input[b * C * H * W + c * H * W + ih * W + iw] * kernel[k * C * KH * KW + c * KH * KW + (KH - 1 - kh) * KW + (KW - 1 - kw)];
                }
            }
        }
    }
    output[b * K * (H-KH+1) * (W-KW+1) + k * (H-KH+1) * (W-KW+1) + h * (W-KW+1) + w] = sum;
}

// Function to compare outputs
bool compare_outputs(const float* output1, const float* output2, int size, float tolerance = 1e-2) {
    for (int i = 0; i < size; ++i) {
        if (fabs(output1[i] - output2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void print_tensor_shape(const string& name, int n, int c, int h, int w) {
    cout << name << " shape: (" << n << ", " << c << ", " << h << ", " << w << ")" << endl;
}

void print_first_element(const string& name, const float* data, int c, int h, int w) {
    cout << name << " first element:" << endl;
    for (int i = 0; i < c; ++i) {
        cout << "- Channel " << i << ":" << endl;
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < w; ++k) {
                cout << data[i * h * w + j * w + k] << " ";
            }
            cout << endl;
        }
    }
}

int main() {
    const int batch_size = 128;
    const int num_epochs = 5;
    const int data_size = 28 * 28; // MNIST image dimensions

    vector<float> train_data;
    vector<int> train_labels;

    // Load and normalize data
    read_mnist_image_file("../../data/MNIST/raw/train-images-idx3-ubyte", train_data);
    read_mnist_label_file("../../data/MNIST/raw/train-labels-idx1-ubyte", train_labels);
    normalize_data(train_data);

    // Allocate memory on GPU
    float* d_data;
    int* d_labels;
    allocate_gpu_memory(d_data, d_labels, train_data, train_labels);

    // Initialize cuDNN
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t output_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

    // Set tensor descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 28, 28));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 32, 1, 5, 5));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Find output dimensions
    int n, c, h, w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &c, &h, &w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    // Print the output shape
    print_tensor_shape("cuDNN Convolution Output", n, c, h, w);
    
    // Allocate memory for the filter and output on GPU
    float* d_filter;
    float* d_output_cudnn;
    float* d_output_custom;

    CUDA_CHECK(cudaMalloc(&d_filter, 32 * 1 * 5 * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_cudnn, n * c * h * w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_custom, n * c * h * w * sizeof(float)));

    // Host memory to compare outputs
    float* h_output_cudnn = new float[n * c * h * w];
    float* h_output_custom = new float[n * c * h * w];

    // Random initialization of filters for demonstration purposes
    vector<float> h_filter(32 * 1 * 5 * 5);
    srand(42);
    for (float& el : h_filter) { el = static_cast<float>(rand()) / RAND_MAX; }
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter.data(), 32 * 1 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice));

    int iters_per_epoch = 60000 / batch_size;
    cout << "Iters per epoch: " << iters_per_epoch << endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (int i = 0; i < iters_per_epoch; ++i) {
            float* batch_data;
            int* batch_labels;

            // Retrieve batch
            get_batch(d_data, d_labels, batch_data, batch_labels, batch_size, i, data_size);

            // Timing cuDNN Convolution
            auto start_cudnn = std::chrono::high_resolution_clock::now();
            // Perform cuDNN convolution
            const float alpha = 1.0f, beta = 0.0f;
            CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, input_desc, batch_data, filter_desc, d_filter, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, output_desc, d_output_cudnn));
            auto end_cudnn = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_cudnn = end_cudnn - start_cudnn;

            // Timing custom CUDA Convolution
            auto start_custom = std::chrono::high_resolution_clock::now();
            dim3 blockDim(h, w);
            dim3 gridDim(n, c);
            custom_conv2d_kernel<<<gridDim, blockDim>>>(batch_data, d_output_custom, d_filter, batch_size, 1, 28, 28, 32, 5, 5);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto end_custom = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> time_custom = end_custom - start_custom;

            // Copy results back to host
            CUDA_CHECK(cudaMemcpy(h_output_cudnn, d_output_cudnn, n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_output_custom, d_output_custom, n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost));

            // Compare the results
            bool outputs_match = compare_outputs(h_output_cudnn, h_output_custom, n * c * h * w);

            // Print the output shape
            // print_tensor_shape("Custom CUDA Convolution Output", n, c, h, w);

            // Print the first element of the batch
            if (i == 0 && epoch == 0) {  // Print only for the first batch of the first epoch for brevity
                print_first_element("cuDNN Convolution First Element", h_output_cudnn, c, h, w);
                print_first_element("Custom CUDA Convolution First Element", h_output_custom, c, h, w);
            }

            if (i % 100 == 99) {
                cout << "Epoch: " << epoch + 1 << ", Iter: " << i + 1 << endl;
                cout << "cuDNN Conv2D time: " << time_cudnn.count() << " seconds" << endl;
                cout << "Custom CUDA Conv2D time: " << time_custom.count() << " seconds" << endl;
                cout << "Outputs match: " << (outputs_match ? "Yes" : "No") << endl;
            }
        }
    }

    cout << "Finished Training" << endl;

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output_cudnn));
    CUDA_CHECK(cudaFree(d_output_custom));

    // Free host memory
    delete[] h_output_cudnn;
    delete[] h_output_custom;

    // Destroy cuDNN descriptors and handle
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));

    return 0;
}