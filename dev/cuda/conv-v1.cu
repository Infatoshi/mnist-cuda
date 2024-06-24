#include "dataloader/dataloader.cuh"
#include <iostream>
#include <cudnn.h>

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

    // Find output dimensions (this part isn't necessary for the convolution operation, but it's good to know the output dimensions for memory allocation purposes)
    // edit, you actually need this to set the output tensor descriptor (dont need to print out the shape though)
    int n, c, h, w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &c, &h, &w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    // // Calculate the array length of the output tensor
    // int output_array_length = n * c * h * w;
    // cout << "Output tensor array length: " << output_array_length << endl;
    // batchsize * channels * height * width -> 128 * 32 * 24 * 24

    // Allocate memory for the filter and output on GPU
    float* d_filter;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_filter, 32 * 1 * 5 * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * c * h * w * sizeof(float)));

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

            // Perform convolution
            const float alpha = 1.0f, beta = 0.0f;
            CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, input_desc, batch_data, filter_desc, d_filter, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &beta, output_desc, d_output));

            // Timing data movement
            double time_taken = time_to_move_data(train_data.data(), batch_data, batch_size * data_size);
            if (i % 100 == 99) {
                cout << "Epoch: " << epoch + 1 << ", Iter: " << i + 1 << endl;
                cout << "Time taken to move data to GPU: " << time_taken << " seconds" << endl;
            }
        }
    }

    cout << "Finished Training" << endl;

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output));

    // Destroy cuDNN descriptors and handle
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));

    return 0;
}