#include "dataloader.cuh"
#include <vector>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// CUDA error checking macro
#define CUDA_CHECK(error) \
    if (error != cudaSuccess) { \
        cout << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(-1); \
    }

void read_mnist_image_file(const char* filename, vector<float>& data) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int number_of_rows = 0;
        int number_of_columns = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number);
        
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = __builtin_bswap32(number_of_images);

        file.read((char*)&number_of_rows, sizeof(number_of_rows));
        number_of_rows = __builtin_bswap32(number_of_rows);

        file.read((char*)&number_of_columns, sizeof(number_of_columns));
        number_of_columns = __builtin_bswap32(number_of_columns);

        int image_size = number_of_rows * number_of_columns;

        data.resize(number_of_images * image_size);

        for (int i = 0; i < number_of_images; ++i) {
            for (int j = 0; j < image_size; ++j) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                data[i * image_size + j] = static_cast<float>(temp) / 255.0f;
            }
        }

        file.close();
    } else {
        cerr << "Could not open the image file!" << endl;
        exit(1);
    }
}

void read_mnist_label_file(const char* filename, vector<int>& labels) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number);
        
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = __builtin_bswap32(number_of_labels);

        labels.resize(number_of_labels);

        for (int i = 0; i < number_of_labels; ++i) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labels[i] = static_cast<int>(temp);
        }

        file.close();
    } else {
        cerr << "Could not open the label file!" << endl;
        exit(1);
    }
}

void normalize_data(vector<float>& data) {
    float mean = 0.1307f;
    float std = 0.3081f;
    for (auto& val : data) {
        val = (val - mean) / std;
    }
}

void allocate_gpu_memory(float*& d_data, int*& d_labels, const vector<float>& data, const vector<int>& labels) {
    size_t data_size = data.size() * sizeof(float);
    size_t labels_size = labels.size() * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_data, data_size));
    CUDA_CHECK(cudaMalloc(&d_labels, labels_size));

    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, labels.data(), labels_size, cudaMemcpyHostToDevice));
}

void get_batch(float* d_data, int* d_labels, float*& batch_data, int*& batch_labels, int batch_size, int batch_index, int data_size) {
    int offset = batch_index * batch_size;
    size_t batch_data_size = batch_size * data_size * sizeof(float);
    size_t batch_labels_size = batch_size * sizeof(int);

    batch_data = d_data + offset * data_size;
    batch_labels = d_labels + offset;
}

double time_to_move_data(float* h_data, float* d_data, int size) {
    auto start = chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    auto end = chrono::high_resolution_clock::now();
    // copy back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    // print h_data shape to make sure it has the amount of elements we expect
    // cout << "h_data shape: " << size << endl;
    chrono::duration<double> diff = end - start;

    return diff.count();
}