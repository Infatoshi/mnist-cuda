#include "./dataloader/dataloader.cuh"
#include <iostream>

using namespace std;

// CUDA error checking macro
#define CUDA_CHECK(error) \
    if (error != cudaSuccess) { \
        cout << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << endl; \
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

    int iters_per_epoch = 60000 / batch_size;
    cout << "Iters per epoch: " << iters_per_epoch << endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (int i = 0; i < iters_per_epoch; ++i) {
            float* batch_data;
            int* batch_labels;

            // Retrieve batch
            get_batch(d_data, d_labels, batch_data, batch_labels, batch_size, i, data_size);

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

    return 0;
}