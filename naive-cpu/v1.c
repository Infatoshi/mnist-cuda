#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 1
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 4
#define EPOCHS 10
#define LEARNING_RATE 0.001

typedef struct {
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
    float *grad_weights1;
    float *grad_weights2;
    float *grad_bias1;
    float *grad_bias2;
} NeuralNetwork;


// load batched img data
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// load batch labels
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// kaiming init func for weights
void initialize_weights(float *weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

// basic init for biases
void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// Modify softmax to work with batches
void softmax(float *x, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        float max = x[b * size];
        for (int i = 1; i < size; i++) {
            if (x[b * size + i] > max) max = x[b * size + i];
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[b * size + i] = expf(x[b * size + i] - max);
            sum += x[b * size + i];
        }
        for (int i = 0; i < size; i++) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

void matmul_a_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

// Matrix multiplication A @ B.T
void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[j * n + l];
            }
        }
    }
}

// Matrix multiplication A.T @ B
void matmul_at_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < m; l++) {
                C[i * k + j] += A[l * n + i] * B[l * k + j];
            }
        }
    }
}

// ReLU forward
void relu_forward(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

// Add bias
void bias_forward(float *x, float *bias, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            x[b * size + i] += bias[i];
        }
    }
}

// Modified forward function
void forward(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size) {
    // Input to Hidden (X @ W1)
    matmul_a_b(input, nn->weights1, hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    
    // Add bias1
    bias_forward(hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    
    // Apply ReLU
    relu_forward(hidden, batch_size * HIDDEN_SIZE);

    // Hidden to Output (Hidden @ W2)
    matmul_a_b(hidden, nn->weights2, output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);

    // Add bias2
    bias_forward(output, nn->bias2, batch_size, OUTPUT_SIZE);
    
    // Apply softmax
    softmax(output, batch_size, OUTPUT_SIZE);
}

// Modify cross_entropy_loss to work with batches
float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}


// Zero out gradients
void zero_grad(float *grad, int size) {
    memset(grad, 0, size * sizeof(float));
}

// ReLU backward
void relu_backward(float *grad, float *x, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] *= (x[i] > 0);
    }
}

// Bias backward
void bias_backward(float *grad_bias, float *grad, int batch_size, int size) {
    for (int i = 0; i < size; i++) {
        grad_bias[i] = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_bias[i] += grad[b * size + i];
        }
    }
}

// Compute gradients for output layer
void compute_output_gradients(float *grad_output, float *output, int *labels, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
}

// Update gradients for weights and biases
void update_gradients(float *grad_weights, float *grad_bias, float *grad_layer, float *prev_layer, int batch_size, int prev_size, int curr_size) {
    for (int i = 0; i < curr_size; i++) {
        for (int j = 0; j < prev_size; j++) {
            for (int b = 0; b < batch_size; b++) {
                grad_weights[i * prev_size + j] += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
            }
        }
        for (int b = 0; b < batch_size; b++) {
            grad_bias[i] += grad_layer[b * curr_size + i];
        }
    }
}

// Backward pass function
void backward(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size) {
    
    // Initialize gradients to zero
    zero_grad(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad(nn->grad_bias2, OUTPUT_SIZE);

    // Compute gradients for output layer
    float *grad_output = malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    compute_output_gradients(grad_output, output, labels, batch_size);

    // Update gradients for weights2 (W2.grad = grad_output.T @ hidden)
    matmul_at_b(hidden, grad_output, nn->grad_weights2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);

    // Update gradients for bias2
    bias_backward(nn->grad_bias2, grad_output, batch_size, OUTPUT_SIZE);

    // Compute dX2 (gradient of loss w.r.t. input of second layer)
    float *dX2 = malloc(batch_size * HIDDEN_SIZE * sizeof(float));

    // grad_output @ W2.T = dX2 -> (B, 10) @ (10, 256) = (B, 256)
    matmul_a_bt(grad_output, nn->weights2, dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);

    // Compute d_ReLU_out (element-wise multiplication with ReLU derivative)
    float *d_ReLU_out = malloc(batch_size * HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < batch_size * HIDDEN_SIZE; i++) {
        d_ReLU_out[i] = dX2[i] * (hidden[i] > 0);
    }
    // retains its shape since its just a point-wise operation
    // Update gradients for weights1 (W1.grad = d_ReLU_out.T @ input)
    matmul_at_b(input, d_ReLU_out, nn->grad_weights1, batch_size, INPUT_SIZE, HIDDEN_SIZE);

    // Update gradients for bias1
    bias_backward(nn->grad_bias1, d_ReLU_out, batch_size, HIDDEN_SIZE);

    // Free allocated memory
    free(grad_output);
    free(dX2);
    free(d_ReLU_out);
}

// gradient descent step
void update_weights(NeuralNetwork *nn) {
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        nn->weights1[i] -= LEARNING_RATE * nn->grad_weights1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        nn->weights2[i] -= LEARNING_RATE * nn->grad_weights2[i];
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->bias1[i] -= LEARNING_RATE * nn->grad_bias1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        nn->bias2[i] -= LEARNING_RATE * nn->grad_bias2[i];
    }
}

// Modify train function to work with batches
void train(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *hidden = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            
            forward(nn, &X_train[start_idx * INPUT_SIZE], hidden, output, BATCH_SIZE);

            float loss = cross_entropy_loss(output, &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;

            for (int i = 0; i < BATCH_SIZE; i++) {
                int predicted = 0;
                for (int j = 1; j < OUTPUT_SIZE; j++) {
                    if (output[i * OUTPUT_SIZE + j] > output[i * OUTPUT_SIZE + predicted]) {
                        predicted = j;
                    }
                }
                if (predicted == y_train[start_idx + i]) {
                    correct++;
                }
            }

            backward(nn, &X_train[start_idx * INPUT_SIZE], hidden, output, &y_train[start_idx], BATCH_SIZE);
            update_weights(nn);

            if ((batch + 1) % 100 == 0 || (epoch == 0 && batch == 0)) {
                printf("Epoch %d/%d, Iter %d/%d, Loss: %.4f, Accuracy: %.2f%%\n", 
                       epoch + 1, EPOCHS, batch + 1, num_batches, total_loss / (batch + 1), 
                       100.0f * correct / ((batch + 1) * BATCH_SIZE));
            }
        }
        
        printf("Epoch %d/%d completed, Loss: %.4f, Accuracy: %.2f%%\n", 
            epoch + 1, EPOCHS, total_loss / num_batches, 100.0f * correct / TRAIN_SIZE);
    }
    
    free(hidden);
    free(output);
}

// Modify the initialize function to allocate memory for gradients
void initialize_neural_network(NeuralNetwork *nn) {
    nn->weights1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn->weights2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias2 = malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_weights1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn->grad_weights2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->grad_bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->grad_bias2 = malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(nn->weights1, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(nn->weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    initialize_bias(nn->bias1, HIDDEN_SIZE);
    initialize_bias(nn->bias2, OUTPUT_SIZE);
}

int main() {
    srand(time(NULL));

    NeuralNetwork nn;
    initialize_neural_network(&nn);

    float *X_train = malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = malloc(TEST_SIZE * sizeof(int));

    load_data("../mnist_data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("../mnist_data/y_train.bin", y_train, TRAIN_SIZE);
    load_data("../mnist_data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("../mnist_data/y_test.bin", y_test, TEST_SIZE);


    // print first image in the terminal
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (X_train[0 * INPUT_SIZE + i * 28 + j] > 0.0f) {
                printf("X");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }

    printf("First 10 training labels: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", y_train[i]);
    }
    printf("\n");

    train(&nn, X_train, y_train);

    free(nn.weights1);
    free(nn.weights2);
    free(nn.bias1);
    free(nn.bias2);
    free(nn.grad_weights1);
    free(nn.grad_weights2);
    free(nn.grad_bias1);
    free(nn.grad_bias2);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
