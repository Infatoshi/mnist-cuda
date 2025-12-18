#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Timing structure
typedef struct {
    double memory_transfers;  // H2D only (data at start of batch)
    double gpu_compute;       // Forward + Loss + Backward + Update (all GPU)
    double total_time;
} TimingStats;

#define INPUT_SIZE 784
#define HIDDEN_SIZE 1024
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000
#define BATCH_SIZE 32
#define EPOCHS 10
#define LEARNING_RATE 0.01

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error), error); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

typedef struct {
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    float *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;
    float *d_fc1_output, *d_fc2_output, *d_grad_hidden, *d_grad_output;

    // PERSISTENT BUFFERS - NO MORE MALLOC/FREE PER BATCH
    float *d_input_batch;
    int *d_labels;           // Labels on GPU for loss computation
    float *d_loss;           // Per-sample loss for reduction

    cublasHandle_t cublas_handle;
} NeuralNetworkCUDA;

void load_data(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen data"); exit(EXIT_FAILURE); }
    fread(data, sizeof(float), size, f);
    fclose(f);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen labels"); exit(EXIT_FAILURE); }
    fread(labels, sizeof(int), size, f);
    fclose(f);
}

void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

void initialize_weights_host(float *weights, int rows, int cols) {
    float scale = sqrtf(2.0f / rows);
    for (int i = 0; i < rows * cols; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

void initialize_bias_host(float *bias, int size) {
    memset(bias, 0, size * sizeof(float));
}

__global__ void bias_add_kernel(float *x, float *bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        x[idx] += bias[bias_idx];
    }
}

__global__ void relu_kernel(float *x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void relu_backward_kernel(float *grad, float *x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grad[idx] *= (x[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

__global__ void bias_backward_kernel(float *grad_output, float *grad_bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        atomicAdd(&grad_bias[bias_idx], grad_output[idx]);
    }
}

// GPU-side softmax + cross-entropy loss + backward gradient
// Eliminates D2H logits transfer and H2D gradient transfer
__global__ void softmax_cross_entropy_backward_kernel(
    float *logits,           // Input: raw logits [batch x classes]
    int *labels,             // Input: ground truth labels [batch]
    float *grad_output,      // Output: gradients [batch x classes]
    float *loss_per_sample,  // Output: loss per sample [batch]
    int batch_size,
    int num_classes
) {
    int b = blockIdx.x;  // One block per sample
    if (b >= batch_size) return;

    // Use shared memory for this sample's logits
    extern __shared__ float shared[];
    float *sample_logits = shared;

    int tid = threadIdx.x;

    // Load logits to shared memory
    if (tid < num_classes) {
        sample_logits[tid] = logits[b * num_classes + tid];
    }
    __syncthreads();

    // Find max for numerical stability (single thread for small num_classes)
    __shared__ float max_logit;
    __shared__ float sum_exp;

    if (tid == 0) {
        max_logit = sample_logits[0];
        for (int i = 1; i < num_classes; i++) {
            if (sample_logits[i] > max_logit) max_logit = sample_logits[i];
        }
    }
    __syncthreads();

    // Compute exp(logit - max)
    if (tid < num_classes) {
        sample_logits[tid] = expf(sample_logits[tid] - max_logit);
    }
    __syncthreads();

    // Compute sum of exponentials
    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += sample_logits[i];
        }
    }
    __syncthreads();

    // Compute softmax, gradient, and loss
    if (tid < num_classes) {
        float prob = sample_logits[tid] / sum_exp;
        int label = labels[b];

        // Gradient: (prob - one_hot) / batch_size
        float grad = prob;
        if (tid == label) {
            grad -= 1.0f;
        }
        grad /= (float)batch_size;

        grad_output[b * num_classes + tid] = grad;

        // Loss contribution (only for correct class)
        if (tid == label) {
            loss_per_sample[b] = -logf(fmaxf(prob, 1e-7f));
        }
    }
}

// FORWARD PASS ONLY - separate function
void forward_pass_only(NeuralNetworkCUDA *nn, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Forward matmul 1: input * weights1
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           HIDDEN_SIZE, batch_size, INPUT_SIZE,
                           &alpha, nn->d_weights1, HIDDEN_SIZE,
                           nn->d_input_batch, INPUT_SIZE, &beta,
                           nn->d_fc1_output, HIDDEN_SIZE));

    // Forward bias add 1
    int total_hidden = batch_size * HIDDEN_SIZE;
    int grid_hidden = (total_hidden + 255) / 256;
    bias_add_kernel<<<grid_hidden, 256>>>(nn->d_fc1_output, nn->d_bias1, batch_size, HIDDEN_SIZE);

    // Forward ReLU
    relu_kernel<<<grid_hidden, 256>>>(nn->d_fc1_output, total_hidden);

    // Forward matmul 2: hidden * weights2
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           OUTPUT_SIZE, batch_size, HIDDEN_SIZE,
                           &alpha, nn->d_weights2, OUTPUT_SIZE,
                           nn->d_fc1_output, HIDDEN_SIZE, &beta,
                           nn->d_fc2_output, OUTPUT_SIZE));

    // Forward bias add 2 (no sync needed - loss computed on GPU)
    int total_out = batch_size * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    bias_add_kernel<<<grid_out, 256>>>(nn->d_fc2_output, nn->d_bias2, batch_size, OUTPUT_SIZE);
}

// BACKWARD PASS ONLY - separate function  
void backward_pass_only(NeuralNetworkCUDA *nn, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Zero gradients (async)
    CUDA_CHECK(cudaMemset(nn->d_grad_weights1, 0, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_weights2, 0, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_bias1, 0, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_bias2, 0, OUTPUT_SIZE * sizeof(float)));

    // Backward matmul 2a: weights2 gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           OUTPUT_SIZE, HIDDEN_SIZE, batch_size,
                           &alpha, nn->d_grad_output, OUTPUT_SIZE,
                           nn->d_fc1_output, HIDDEN_SIZE, &beta,
                           nn->d_grad_weights2, OUTPUT_SIZE));

    // Backward bias2 gradients
    int total_out = batch_size * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    bias_backward_kernel<<<grid_out, 256>>>(nn->d_grad_output, nn->d_grad_bias2, batch_size, OUTPUT_SIZE);

    // Backward matmul 2b: hidden gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           HIDDEN_SIZE, batch_size, OUTPUT_SIZE,
                           &alpha, nn->d_weights2, OUTPUT_SIZE,
                           nn->d_grad_output, OUTPUT_SIZE, &beta,
                           nn->d_grad_hidden, HIDDEN_SIZE));

    // Backward ReLU
    int total_hidden = batch_size * HIDDEN_SIZE;
    int grid_hidden = (total_hidden + 255) / 256;
    relu_backward_kernel<<<grid_hidden, 256>>>(nn->d_grad_hidden, nn->d_fc1_output, total_hidden);

    // Backward matmul 1a: weights1 gradients
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           HIDDEN_SIZE, INPUT_SIZE, batch_size,
                           &alpha, nn->d_grad_hidden, HIDDEN_SIZE,
                           nn->d_input_batch, INPUT_SIZE, &beta,
                           nn->d_grad_weights1, HIDDEN_SIZE));

    // Backward bias1 gradients
    bias_backward_kernel<<<grid_hidden, 256>>>(nn->d_grad_hidden, nn->d_grad_bias1, batch_size, HIDDEN_SIZE);
}

// WEIGHT UPDATES ONLY - separate function
void update_weights_only(NeuralNetworkCUDA *nn, float lr) {
    float neg_lr = -lr;
    
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE * HIDDEN_SIZE,
                           &neg_lr, nn->d_grad_weights1, 1, nn->d_weights1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE * OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_weights2, 1, nn->d_weights2, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE,
                           &neg_lr, nn->d_grad_bias1, 1, nn->d_bias1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_bias2, 1, nn->d_bias2, 1));
    
    // Final sync only at end of entire batch (required before next iteration)
    CUDA_CHECK(cudaDeviceSynchronize());
}

// GPU-side loss computation function
// Returns average loss after computing softmax, cross-entropy, and gradients on GPU
float compute_loss_on_gpu(NeuralNetworkCUDA *nn, int batch_size) {
    // Launch softmax + cross-entropy + backward kernel
    int shared_mem = OUTPUT_SIZE * sizeof(float);
    softmax_cross_entropy_backward_kernel<<<batch_size, 32, shared_mem>>>(
        nn->d_fc2_output, nn->d_labels, nn->d_grad_output, nn->d_loss,
        batch_size, OUTPUT_SIZE);

    // Copy per-sample losses back and reduce on CPU (small transfer: 32 floats)
    float h_loss[BATCH_SIZE];
    CUDA_CHECK(cudaMemcpy(h_loss, nn->d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        total_loss += h_loss[i];
    }
    return total_loss / batch_size;
}

void initialize_random_weights_cuda(NeuralNetworkCUDA *nn) {
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    initialize_weights_host(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights1);

    float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    initialize_weights_host(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights2);

    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    initialize_bias_host(h_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias1);

    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    initialize_bias_host(h_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias2);
}

void initialize_nn_cuda(NeuralNetworkCUDA *nn) {
    // Network weights and gradients
    CUDA_CHECK(cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc1_output, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc2_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // PERSISTENT BUFFERS - ALLOCATED ONCE, REUSED FOR ALL BATCHES
    CUDA_CHECK(cudaMalloc(&nn->d_input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_labels, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&nn->d_loss, BATCH_SIZE * sizeof(float)));

    CUBLAS_CHECK(cublasCreate(&nn->cublas_handle));
    initialize_random_weights_cuda(nn);
}

void free_nn_cuda(NeuralNetworkCUDA *nn) {
    CUDA_CHECK(cudaFree(nn->d_weights1));
    CUDA_CHECK(cudaFree(nn->d_weights2));
    CUDA_CHECK(cudaFree(nn->d_bias1));
    CUDA_CHECK(cudaFree(nn->d_bias2));
    CUDA_CHECK(cudaFree(nn->d_grad_weights1));
    CUDA_CHECK(cudaFree(nn->d_grad_weights2));
    CUDA_CHECK(cudaFree(nn->d_grad_bias1));
    CUDA_CHECK(cudaFree(nn->d_grad_bias2));
    CUDA_CHECK(cudaFree(nn->d_fc1_output));
    CUDA_CHECK(cudaFree(nn->d_fc2_output));
    CUDA_CHECK(cudaFree(nn->d_grad_hidden));
    CUDA_CHECK(cudaFree(nn->d_grad_output));
    
    // Free persistent buffers
    CUDA_CHECK(cudaFree(nn->d_input_batch));
    CUDA_CHECK(cudaFree(nn->d_labels));
    CUDA_CHECK(cudaFree(nn->d_loss));

    CUBLAS_CHECK(cublasDestroy(nn->cublas_handle));
}

int main() {
    srand(12345); // Fixed seed for debugging

    float *train_data = (float *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *train_labels = (int *)malloc(TRAIN_SIZE * sizeof(int));
    load_data("./data/X_train.bin", train_data, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(train_data, TRAIN_SIZE * INPUT_SIZE);
    load_labels("./data/y_train.bin", train_labels, TRAIN_SIZE);

    NeuralNetworkCUDA nn;
    initialize_nn_cuda(&nn);

    int num_batches = TRAIN_SIZE / BATCH_SIZE;
    
    // Initialize CORRECTED timing stats
    TimingStats stats = {0};
    
    struct timespec start, end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        for (int batch = 0; batch < num_batches; batch++) {
            float *batch_input = train_data + batch * BATCH_SIZE * INPUT_SIZE;
            int *batch_labels = train_labels + batch * BATCH_SIZE;

            // === H2D Transfer: input data + labels ===
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaMemcpy(nn.d_input_batch, batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(nn.d_labels, batch_labels, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.memory_transfers += get_time_diff(step_start, step_end);

            // === ALL GPU COMPUTATION ===
            clock_gettime(CLOCK_MONOTONIC, &step_start);

            // Forward pass
            forward_pass_only(&nn, BATCH_SIZE);

            // Loss + backward gradient (GPU-side softmax + cross-entropy)
            float batch_loss = compute_loss_on_gpu(&nn, BATCH_SIZE);
            total_loss += batch_loss;

            // Backward pass
            backward_pass_only(&nn, BATCH_SIZE);

            // Weight updates
            update_weights_only(&nn, LEARNING_RATE);

            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.gpu_compute += get_time_diff(step_start, step_end);
        }
        printf("Epoch %d loss: %.4f\n", epoch, total_loss / num_batches);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    stats.total_time = get_time_diff(start, end);
    
    printf("\n=== CUBLAS GPU IMPLEMENTATION (ALL COMPUTATION ON GPU) ===\n");
    printf("Total training time: %.3f seconds\n\n", stats.total_time);

    printf("Timing Breakdown:\n");
    printf("  H2D transfers:  %6.3fs (%5.1f%%)\n", stats.memory_transfers, 100.0 * stats.memory_transfers / stats.total_time);
    printf("  GPU compute:    %6.3fs (%5.1f%%)\n", stats.gpu_compute, 100.0 * stats.gpu_compute / stats.total_time);

    free_nn_cuda(&nn);
    free(train_data);
    free(train_labels);

    return 0;
}
