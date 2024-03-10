#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

#define OUTPUT_SIZE 10

float ReLU(float x)
{
    if (x > 0)
    {
        return x;
    }
    else
    {
        return 0;
    }
}

float ReLU_dash(float x)
{
    if (x < 0)
    {
        return 0;
    }
    else if (x > 0)
    {
        return 1;
    }
    else
    {
        // https://stackoverflow.com/questions/42042561/relu-derivative-in-backpropagation
        // Try x == 0 ReLU = 0
        return 0;
    }
}

// Calculate activation for n th output node
float softmax(int n, int layer, float *weighted_inputs)
{
    double denominator = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        denominator += exp(weighted_inputs[layer * OUTPUT_SIZE + i]);
    }
    return exp(weighted_inputs[layer * OUTPUT_SIZE + n]) / denominator;
}

float kaiming_init(int input_dim, unsigned int *seed)
{
    return ((float)rand_r(seed) / RAND_MAX) * sqrt(2.0 / (double)input_dim);
}

typedef struct _NeuralNetwork
{
    // Deal with a batch of images concurrently
    int matrix_layers;

    int total_pixels;
    int rows;
    int cols;
    int num_hidden_layers;
    int num_neurons_per_layer;
    float alpha; // Learning Rate

    int total_layers;        // input layer + hidden layers + output layer
    int *layer_sizes;        // {784, ..., 10}
    float **weights;         // For each layer store edges to each of the next layer nodes: weights[layer][1D matrix]
    float **biases;          // Store bias for each node in each layer: biases[layer-1][nodes]
    float **weighted_inputs; // Z: weighted_inputs[layer-1][nodes]
    float **activation;      // activation[layer-1][nodes]
    float **errors;          // errors[layer-1][nodes]

    float **prod_error_activation; // Need for each weight: prod_error_activation[layer][1D matrix]
    float **error_accumulation;    // Need for each node besides input: error_accumulation[layer-1][nodes]
} NeuralNetwork;

NeuralNetwork *init_NeuralNetwork(int total_pixels, int nl, int nh, float alpha, int nb, unsigned int *seed)
{
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->matrix_layers = nb;
    nn->total_pixels = total_pixels;
    nn->num_hidden_layers = nl;
    nn->num_neurons_per_layer = nh;
    nn->alpha = alpha;

    nn->total_layers = nl + 2;
    nn->layer_sizes = malloc(sizeof(int) * nn->total_layers);
    nn->layer_sizes[0] = total_pixels; //  input layer
    for (int i = 1; i <= nl; i++)
    {
        nn->layer_sizes[i] = nh; // hidden layers
    }
    nn->layer_sizes[nn->total_layers - 1] = OUTPUT_SIZE; // output layer

    nn->weights = malloc(sizeof(float *) * (nl + 1));
    nn->prod_error_activation = malloc(sizeof(float *) * (nl + 1));
    for (int i = 0; i < nl + 1; i++)
    {
        int cur_num_neurons = nn->layer_sizes[i];
        int next_num_neurons = nn->layer_sizes[i + 1];
        nn->weights[i] = malloc(sizeof(float) * cur_num_neurons * next_num_neurons);
        nn->prod_error_activation[i] = malloc(sizeof(float) * cur_num_neurons * next_num_neurons);
        for (int j = 0; j < cur_num_neurons; j++)
        {
            for (int k = 0; k < next_num_neurons; k++)
            {
                nn->weights[i][j * next_num_neurons + k] = kaiming_init(total_pixels, seed); // Initialize the weights of the NN
                nn->prod_error_activation[i][j * next_num_neurons + k] = 0.0;                // Init the error to zero
            }
        }
    }

    // For each batch we need separate layers for calculations
    nn->activation = malloc(sizeof(float *) * (nl + 1));
    nn->weighted_inputs = malloc(sizeof(float *) * (nl + 1));
    nn->errors = malloc(sizeof(float *) * (nl + 1));
    for (int i = 0; i < (nl + 1); i++)
    {
        nn->weighted_inputs[i] = malloc(sizeof(float) * nn->matrix_layers * nn->layer_sizes[i + 1]);
        nn->activation[i] = malloc(sizeof(float *) * nn->matrix_layers * nn->layer_sizes[i + 1]);
        nn->errors[i] = malloc(sizeof(float *) * nn->matrix_layers * nn->layer_sizes[i + 1]);
        for (int j = 0; j < nn->matrix_layers; j++)
        {
            for (int k = 0; k < nn->layer_sizes[i + 1]; k++)
            {
                nn->weighted_inputs[i][j * nn->layer_sizes[i + 1] + k] = 0.0;
            }
        }
    }

    // Need all but the input layer
    nn->biases = malloc(sizeof(float *) * (nl + 1));
    nn->error_accumulation = malloc(sizeof(float *) * (nl + 1));
    for (int i = 0; i < nl + 1; i++)
    {
        nn->biases[i] = malloc(sizeof(float) * nn->layer_sizes[i + 1]);
        nn->error_accumulation[i] = malloc(sizeof(float) * nn->layer_sizes[i + 1]);
        for (int j = 0; j < nn->layer_sizes[i + 1]; j++)
        {
            nn->biases[i][j] = kaiming_init(nn->layer_sizes[i + 1], seed);
            nn->error_accumulation[i][j] = 0.0;
        }
    }

    return nn;
}

void reset_accumulation(NeuralNetwork *nn)
{
    for (int i = 0; i < nn->total_layers - 1; i++)
    {
        int cur_num_neurons = nn->layer_sizes[i];
        int next_num_neurons = nn->layer_sizes[i + 1];

        for (int k = 0; k < next_num_neurons; k++)
        {
            for (int j = 0; j < cur_num_neurons; j++)
            {
                nn->prod_error_activation[i][j * next_num_neurons + k] = 0.0; // Reset to zero
            }
            nn->error_accumulation[i][k] = 0.0; // Reset
        }
    }
}

void free_NN(NeuralNetwork *nn)
{
    for (int i = 0; i < nn->total_layers - 1; i++)
    {
        free(nn->weights[i]);
        free(nn->prod_error_activation[i]);
    }
    free(nn->weights);
    free(nn->prod_error_activation);

    for (int i = 0; i < nn->total_layers - 1; i++)
    {
        free(nn->activation[i]);
        free(nn->weighted_inputs[i]);
        free(nn->errors[i]);
    }
    free(nn->activation);
    free(nn->weighted_inputs);
    free(nn->errors);

    for (int i = 0; i < nn->total_layers - 1; i++)
    {
        free(nn->biases[i]);
        free(nn->error_accumulation[i]);
    }
    free(nn->biases);
    free(nn->error_accumulation);
    free(nn->layer_sizes);

    free(nn);
}

void train_network(NeuralNetwork *nn, float *image_matrix, int nb, float *label_matrix)
{
    /* -------- Feed Forward -------- */
    for (int l = 1; l < nn->total_layers; l++)
    {
        if (l == 1)
        {
            // input layer to hidden layer
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        nb, nn->layer_sizes[l], nn->layer_sizes[l - 1], 1.0, image_matrix, nn->layer_sizes[l - 1], nn->weights[l - 1], nn->layer_sizes[l], 0.0, nn->weighted_inputs[l - 1], nn->layer_sizes[l]);
        }
        else
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        nb, nn->layer_sizes[l], nn->layer_sizes[l - 1], 1.0, nn->activation[l - 2], nn->layer_sizes[l - 1], nn->weights[l - 1], nn->layer_sizes[l], 0.0, nn->weighted_inputs[l - 1], nn->layer_sizes[l]);
        }
        // Add Bias
        for (int b = 0; b < nb; b++)
        {
            cblas_saxpy(nn->layer_sizes[l], 1.0, nn->biases[l - 1], 1, &nn->weighted_inputs[l - 1][b * nn->layer_sizes[l]], 1);
        }

        for (int b = 0; b < nb; b++)
        {
            for (int i = 0; i < nn->layer_sizes[l]; i++)
            {
                if (l != nn->total_layers - 1)
                {
                    // Hidden Layers
                    nn->activation[l - 1][b * nn->layer_sizes[l] + i] = ReLU(nn->weighted_inputs[l - 1][b * nn->layer_sizes[l] + i]);
                }
                else
                {
                    // Output Layer
                    nn->activation[l - 1][b * nn->layer_sizes[l] + i] = softmax(i, b, nn->weighted_inputs[l - 1]);
                }
            }
        }
    }

    /* -------- Output Error -------- */
    // Calculate output layer
    // https://towardsdatascience.com/deriving-backpropagation-with-cross-entropy-loss-d24811edeaf9
    // https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax/
    // https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1

    cblas_scopy(OUTPUT_SIZE * nb, nn->activation[nn->total_layers - 2], 1, nn->errors[nn->total_layers - 2], 1);
    cblas_saxpy(OUTPUT_SIZE * nb, -1.0, label_matrix, 1, nn->errors[nn->total_layers - 2], 1);

    for (int b = 0; b < nb; b++)
    {
        cblas_saxpy(OUTPUT_SIZE, 1.0, &nn->errors[nn->total_layers - 2][b * OUTPUT_SIZE], 1, nn->error_accumulation[nn->total_layers - 2], 1);
    }

    // If hidden layers == 0, use the input layer
    if (nn->num_hidden_layers == 0)
    {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    nn->layer_sizes[nn->total_layers - 2], nn->layer_sizes[nn->total_layers - 1], nb, 1.0, image_matrix, nn->layer_sizes[nn->total_layers - 2], nn->errors[nn->total_layers - 2], nn->layer_sizes[nn->total_layers - 1], 0.0, nn->prod_error_activation[nn->total_layers - 2], nn->layer_sizes[nn->total_layers - 1]);
    }
    else
    {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    nn->layer_sizes[nn->total_layers - 2], nn->layer_sizes[nn->total_layers - 1], nb, 1.0, nn->activation[nn->total_layers - 3], nn->layer_sizes[nn->total_layers - 2], nn->errors[nn->total_layers - 2], nn->layer_sizes[nn->total_layers - 1], 0.0, nn->prod_error_activation[nn->total_layers - 2], nn->layer_sizes[nn->total_layers - 1]);
    }

    // Back Propagation
    for (int l = nn->total_layers - 2; l >= 1; l--)
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nb, nn->layer_sizes[l], nn->layer_sizes[l + 1], 1.0, nn->errors[l], nn->layer_sizes[l + 1], nn->weights[l], nn->layer_sizes[l + 1], 0.0, nn->errors[l - 1], nn->layer_sizes[l]);
        for (int b = 0; b < nb; b++)
        {
            for (int i = 0; i < nn->layer_sizes[l]; i++)
            {
                nn->errors[l - 1][b * nn->layer_sizes[l] + i] *= ReLU_dash(nn->weighted_inputs[l - 1][b * nn->layer_sizes[l] + i]);
            }
            cblas_saxpy(nn->layer_sizes[l], 1.0, &nn->errors[l - 1][b * nn->layer_sizes[l]], 1, nn->error_accumulation[l - 1], 1);
        }
        if (l == 1)
        {
            // Use the input layer
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        nn->layer_sizes[l - 1], nn->layer_sizes[l], nb, 1.0, image_matrix, nn->layer_sizes[l - 1], nn->errors[l - 1], nn->layer_sizes[l], 0.0, nn->prod_error_activation[l - 1], nn->layer_sizes[l]);
        }
        else
        {
            // Use prev hidden layer
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        nn->layer_sizes[l - 1], nn->layer_sizes[l], nb, 1.0, nn->activation[l - 2], nn->layer_sizes[l - 1], nn->errors[l - 1], nn->layer_sizes[l], 0.0, nn->prod_error_activation[l - 1], nn->layer_sizes[l]);
        }
    }
    return;
}

void gradient_descent(NeuralNetwork *nn, int m)
{
    for (int l = 0; l < nn->total_layers - 1; l++)
    {
        cblas_saxpy(nn->layer_sizes[l] * nn->layer_sizes[l + 1], -1.0 * (nn->alpha / (float)m), nn->prod_error_activation[l], 1, nn->weights[l], 1);
        cblas_saxpy(nn->layer_sizes[l + 1], -1.0 * (nn->alpha / (float)m), nn->error_accumulation[l], 1, nn->biases[l], 1);
    }
}

int classify_image(NeuralNetwork *nn, const float *input_layer)
{
    /* -------- Feed Forward -------- */
    for (int l = 1; l < nn->total_layers; l++)
    {
        if (l == 1)
        {
            // input layer to hidden layer
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        1, nn->layer_sizes[l], nn->layer_sizes[l - 1], 1.0, input_layer, nn->layer_sizes[l - 1], nn->weights[l - 1], nn->layer_sizes[l], 0.0, nn->weighted_inputs[l - 1], nn->layer_sizes[l]);
        }
        else
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        1, nn->layer_sizes[l], nn->layer_sizes[l - 1], 1.0, nn->activation[l - 2], nn->layer_sizes[l - 1], nn->weights[l - 1], nn->layer_sizes[l], 0.0, nn->weighted_inputs[l - 1], nn->layer_sizes[l]);
        }
        // Add Bias
        cblas_saxpy(nn->layer_sizes[l], 1.0, nn->biases[l - 1], 1, nn->weighted_inputs[l - 1], 1);
        for (int i = 0; i < nn->layer_sizes[l]; i++)
        {
            if (l != nn->total_layers - 1)
            {
                // Hidden Layers
                nn->activation[l - 1][i] = ReLU(nn->weighted_inputs[l - 1][i]);
            }
            else
            {
                // Output Layer
                nn->activation[l - 1][i] = softmax(i, 0, nn->weighted_inputs[l - 1]);
            }
        }
    }

    // printf("Guess:\n");
    float max_activation = nn->activation[nn->total_layers - 2][0];
    int max_index = 0;
    // printf("%d: %lf\n", i, nn->activation[nn->total_layers - 2][0]);
    for (int i = 1; i < nn->layer_sizes[nn->total_layers - 1]; i++)
    {
        if (max_activation < nn->activation[nn->total_layers - 2][i])
        {
            max_activation = nn->activation[nn->total_layers - 2][i];
            max_index = i;
        }
        // printf("%d: %lf\n", i, nn->activation[nn->total_layers - 2][i]);
    }
    return max_index;
}

// Calculate the cost at the test stage (on one image)
float calculate_cost_test(NeuralNetwork *nn, int label)
{
    return (float)(-log((double)nn->activation[nn->total_layers - 2][label]));
}

#endif // NEURAL_NETWORK_H