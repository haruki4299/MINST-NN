#include "read_input.h"
#include "neural_network.h"

int main(int argc, char **argv)
{
    unsigned int seed = 42;

    // Read Command Line Arguments
    if (argc < 6)
    {
        printf("Not enough arguments. Usage: ./%s nl nh ne nb alpha\n", argv[0]);
        exit(1);
    }

    int nl, nh, ne, nb;
    float alpha;
    nl = atoi(argv[1]);    // number of dense (fully connected) linear layers
    nh = atoi(argv[2]);    // number of units in each of the hidden layers
    ne = atoi(argv[3]);    // number of training epochs
    nb = atoi(argv[4]);    // number of training samples per batch
    alpha = atof(argv[5]); // learning rate

    // Step1: Input File Prep
    FILE *train_image_file = fopen("datafiles/train-images-idx3-ubyte", "rb");
    FILE *train_label_file = fopen("datafiles/train-labels-idx1-ubyte", "rb");

    FILE *test_image_file = fopen("datafiles/t10k-images-idx3-ubyte", "rb");
    FILE *test_label_file = fopen("datafiles/t10k-labels-idx1-ubyte", "rb");
    if (train_image_file == NULL || train_label_file == NULL || test_image_file == NULL || test_label_file == NULL)
    {
        printf("Error opening files.\n");
        return 0;
    }

    IMGs *train_imgs = init_IMGs();
    read_images(train_imgs, train_image_file, train_label_file);

    IMGs *test_imgs = init_IMGs();
    read_images(test_imgs, test_image_file, test_label_file);

    NeuralNetwork *nn = init_NeuralNetwork(train_imgs->size, nl, nh, alpha, nb, &seed);
    float *image_matrix = malloc(sizeof(float) * nb * train_imgs->size); // Create a matrix for passing into function
    float *label_matrix = malloc(sizeof(float) * nb * OUTPUT_SIZE);

    int total_rounds_per_epoch = 60000 / nb;

    // Start the timer

    for (int i = 0; i < ne; i++)
    {
        for (int r = 0; r < total_rounds_per_epoch; r++)
        {
            return_next_images(train_imgs, nb, image_matrix, label_matrix, OUTPUT_SIZE);
            train_network(nn, image_matrix, nb, label_matrix);
            gradient_descent(nn, nb);
            reset_accumulation(nn);
        }
        nn->alpha *= 0.95;

        // TESTING
        int total = test_imgs->total_images;
        int correct = 0;
        float loss = 0;

        // Start the timer
        float *image_data;
        int label;
        for (int k = 0; k < test_imgs->total_images; k++)
        {
            return_next_image(test_imgs, &image_data, &label);
            int guess = classify_image(nn, image_data);
            if (guess == label)
            {
                correct++;
            }
            loss += calculate_cost_test(nn, label);
        }

        printf("Epoch %d: Training Loss (Cost): %f Accuracy: %lf\n", i, loss, (double)correct / (double)total);
    }

    free_NN(nn);

    free_IMGs(train_imgs);
    free_IMGs(test_imgs);

    free(label_matrix);
    free(image_matrix);

    fclose(train_image_file);
    fclose(train_label_file);
}