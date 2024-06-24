#ifndef DATALOADER_CUH
#define DATALOADER_CUH

#include <vector>
#include <cuda_runtime.h>

// Function declarations
void read_mnist_image_file(const char* filename, std::vector<float>& data);
void read_mnist_label_file(const char* filename, std::vector<int>& labels);
void normalize_data(std::vector<float>& data);
void allocate_gpu_memory(float*& d_data, int*& d_labels, const std::vector<float>& data, const std::vector<int>& labels);
void get_batch(float* d_data, int* d_labels, float*& batch_data, int*& batch_labels, int batch_size, int batch_index, int data_size);
double time_to_move_data(float* h_data, float* d_data, int size);

#endif // DATALOADER_CUH