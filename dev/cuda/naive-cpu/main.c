#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 5
#define LEARNING_RATE 0.01

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

// relu forward
void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

// softmax for converting logits to probs & cross_entropy_loss internal
void softmax(float *x, int size) {
    float max = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max) max = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(x[i] / sum, 1e-7f);  // Add small epsilon
    }
}

// fwd function for nn 
void forward(NeuralNetwork *nn, float *input, float *hidden, float *output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = 0.0f;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += nn->weights1[i * INPUT_SIZE + j] * input[j];
        }
        hidden[i] += nn->bias1[i];
    }
    relu(hidden, HIDDEN_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += nn->weights2[i * HIDDEN_SIZE + j] * hidden[j];
        }
        output[i] += nn->bias2[i];
    }
    softmax(output, OUTPUT_SIZE);
}

// cross_entropy_loss 
float cross_entropy_loss(float *output, int label) {
    return -logf(fmaxf(output[label], 1e-7f));  // Add small epsilon
}

// bwkd fn for nn (no accumulation)
void backward(NeuralNetwork *nn, float *input, float *hidden, float *output, int label, 
              float *grad_weights1, float *grad_weights2, float *grad_bias1, float *grad_bias2) {
    float grad_output[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        grad_output[i] = output[i];
    }
    grad_output[label] -= 1.0f;

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            grad_weights2[i * HIDDEN_SIZE + j] = grad_output[i] * hidden[j];
        }
        grad_bias2[i] = grad_output[i];
    }

    float grad_hidden[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        grad_hidden[i] = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            grad_hidden[i] += grad_output[j] * nn->weights2[j * HIDDEN_SIZE + i];
        }
        grad_hidden[i] *= (hidden[i] > 0);
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            grad_weights1[i * INPUT_SIZE + j] = grad_hidden[i] * input[j];
        }
        grad_bias1[i] = grad_hidden[i];
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

// train function calling each of our smaller pieces
void train(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *hidden = malloc(HIDDEN_SIZE * sizeof(float));
    float *output = malloc(OUTPUT_SIZE * sizeof(float));
    float *grad_weights1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float *grad_weights2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *grad_bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    float *grad_bias2 = malloc(OUTPUT_SIZE * sizeof(float));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        
        for (int i = 0; i < TRAIN_SIZE; i++) {
            
            clock_t start, end;
            double cpu_time_used;
            start = clock();
            forward(nn, &X_train[i * INPUT_SIZE], hidden, output);

            int predicted = 0;
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[predicted]) {
                    predicted = j;
                }
            }
            if (predicted == y_train[i]) {
                correct++;
            }

            float loss = cross_entropy_loss(output, y_train[i]);
            total_loss += loss;

            backward(nn, &X_train[i * INPUT_SIZE], hidden, output, y_train[i], 
                     grad_weights1, grad_weights2, grad_bias1, grad_bias2);
            update_weights(nn, grad_weights1, grad_weights2, grad_bias1, grad_bias2);

            end = clock();
            cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
 
            if ((i + 1) % 1000 == 0 || (epoch == 0 && i == 0)) {
                
                printf("Epoch %d/%d, Step %d/%d, Loss: %.4f, Accuracy: %.2f%%\n, Time taken: %f seconds\n", 
                       epoch + 1, EPOCHS, i + 1, TRAIN_SIZE, total_loss / (i + 1), 
                       100.0f * correct / (i + 1), cpu_time_used);
                cpu_time_used = 0;
            }
            
        }

        
        
        printf("Epoch %d/%d completed, Loss: %.4f, Accuracy: %.2f%%\n", 
            epoch + 1, EPOCHS, total_loss / TRAIN_SIZE, 100.0f * correct / TRAIN_SIZE);
        
    }
    
    // free memory
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
