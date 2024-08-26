#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 4
#define EPOCHS 3
#define LEARNING_RATE 0.001

typedef struct {
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
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

// Modify relu to work with batches
void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
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

// Modify forward to work with batches
void forward(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size) {
    // Input to Hidden (X @ W1 + b1)
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            hidden[b * HIDDEN_SIZE + i] = 0.0f;
            for (int j = 0; j < INPUT_SIZE; j++) {
                hidden[b * HIDDEN_SIZE + i] += input[b * INPUT_SIZE + j] * nn->weights1[i * INPUT_SIZE + j];
            }
            hidden[b * HIDDEN_SIZE + i] += nn->bias1[i];
        }
    }
    relu(hidden, batch_size * HIDDEN_SIZE);

    // Hidden to Output (Hidden @ W2 + b2)
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            output[b * OUTPUT_SIZE + i] = 0.0f;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                output[b * OUTPUT_SIZE + i] += hidden[b * HIDDEN_SIZE + j] * nn->weights2[i * HIDDEN_SIZE + j];
            }
            output[b * OUTPUT_SIZE + i] += nn->bias2[i];
        }
    }
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

// Modify backward to work with batches
void backward(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size,
              float *grad_weights1, float *grad_weights2, float *grad_bias1, float *grad_bias2) {
    
    // Initialize gradients to zero
    memset(grad_weights1, 0, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    memset(grad_weights2, 0, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    memset(grad_bias1, 0, HIDDEN_SIZE * sizeof(float));
    memset(grad_bias2, 0, OUTPUT_SIZE * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        float grad_output[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            grad_output[i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[labels[b]] -= 1.0f;

        // Output to Hidden gradients
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                grad_weights2[i * HIDDEN_SIZE + j] += grad_output[i] * hidden[b * HIDDEN_SIZE + j];
            }
            grad_bias2[i] += grad_output[i];
        }

        float grad_hidden[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            grad_hidden[i] = 0.0f;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                grad_hidden[i] += grad_output[j] * nn->weights2[j * HIDDEN_SIZE + i];
            }
            grad_hidden[i] *= (hidden[b * HIDDEN_SIZE + i] > 0);
        }

        // Hidden to Input gradients
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                grad_weights1[i * INPUT_SIZE + j] += grad_hidden[i] * input[b * INPUT_SIZE + j];
            }
            grad_bias1[i] += grad_hidden[i];
        }
    }
}

// gradient descent step
void update_weights(NeuralNetwork *nn, float *grad_weights1, float *grad_weights2, 
                    float *grad_bias1, float *grad_bias2) {
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        nn->weights1[i] -= LEARNING_RATE * grad_weights1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        nn->weights2[i] -= LEARNING_RATE * grad_weights2[i];
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->bias1[i] -= LEARNING_RATE * grad_bias1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        nn->bias2[i] -= LEARNING_RATE * grad_bias2[i];
    }
}

// Modify train function to work with batches
void train(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *hidden = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    float *grad_weights1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float *grad_weights2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *grad_bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    float *grad_bias2 = malloc(OUTPUT_SIZE * sizeof(float));

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

            backward(nn, &X_train[start_idx * INPUT_SIZE], hidden, output, &y_train[start_idx], BATCH_SIZE,
                     grad_weights1, grad_weights2, grad_bias1, grad_bias2);
            update_weights(nn, grad_weights1, grad_weights2, grad_bias1, grad_bias2);

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
    free(grad_weights1);
    free(grad_weights2);
    free(grad_bias1);
    free(grad_bias2);
}


int main() {
    srand(time(NULL));

    NeuralNetwork nn;
    nn.weights1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn.weights2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn.bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn.bias2 = malloc(OUTPUT_SIZE * sizeof(float));

    initialize_weights(nn.weights1, HIDDEN_SIZE * INPUT_SIZE);
    initialize_weights(nn.weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    initialize_bias(nn.bias1, HIDDEN_SIZE);
    initialize_bias(nn.bias2, OUTPUT_SIZE);

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
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
