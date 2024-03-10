#ifndef READ_INPUT_H
#define READ_INPUT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// https://stackoverflow.com/questions/32786493/reversing-byte-order-in-c
uint32_t reverse_bytes(uint32_t bytes)
{
    uint32_t aux = 0;
    uint8_t byte;
    int i;

    for (i = 0; i < 32; i += 8)
    {
        byte = (bytes >> i) & 0xff;
        aux |= byte << (32 - 8 - i);
    }
    return aux;
}

// Struct to store and output images
typedef struct _IMGs
{
    int total_images;
    int current_index; // Keep track of where we are
    int *indices;      // List of randomized (shuffled indices)
    int rows;
    int cols;
    int size;       // For the image
    float **images; // Read all the images up front store image pixels in 1D  Normalize 0 to 255 to 0 to 1
    int *labels;
} IMGs;

IMGs *init_IMGs()
{
    IMGs *imgs = malloc(sizeof(IMGs));
    imgs->total_images = 0;
    imgs->current_index = 0;
    imgs->indices = NULL;
    imgs->rows = 0;
    imgs->cols = 0;
    imgs->size = 0;
    imgs->images = NULL;
    imgs->labels = NULL;
    return imgs;
}

void free_IMGs(IMGs *imgs)
{
    free(imgs->indices);
    free(imgs->labels);
    for (int i = 0; i < imgs->total_images; i++)
    {
        free(imgs->images[i]);
    }
    free(imgs->images);
    free(imgs);
}

// https://stackoverflow.com/questions/6127503/shuffle-array-in-c
void shuffle_indices(IMGs *imgs)
{
    unsigned int seed = time(NULL); // Set Seed Based on Time
    int n = imgs->total_images;
    for (int i = 0; i < n; i++)
    {
        // Generate a random index within [i, n - i]
        int j = i + rand_r(&seed) % (n - i);
        int temp = imgs->indices[i];
        imgs->indices[i] = imgs->indices[j];
        imgs->indices[j] = temp;
    }
    imgs->current_index = 0;
}

void read_images(IMGs *imgs, FILE *images_file, FILE *labels_file)
{ // Just read the image file for the parameters
    uint32_t temp_magic_number, temp_total_images, temp_num_rows, temp_num_cols;
    fread(&temp_magic_number, sizeof(uint32_t), 1, images_file);
    temp_magic_number = reverse_bytes(temp_magic_number);
    fread(&temp_total_images, sizeof(uint32_t), 1, images_file);
    temp_total_images = reverse_bytes(temp_total_images);
    fread(&temp_num_rows, sizeof(uint32_t), 1, images_file);
    temp_num_rows = reverse_bytes(temp_num_rows);
    fread(&temp_num_cols, sizeof(uint32_t), 1, images_file);
    temp_num_cols = reverse_bytes(temp_num_cols);

    int magic_number = (int)temp_magic_number;
    int total_images = (int)temp_total_images;
    int num_rows = (int)temp_num_rows;
    int num_cols = (int)temp_num_cols;

    imgs->rows = num_rows;
    imgs->cols = num_cols;
    imgs->size = num_rows * num_cols;
    imgs->total_images = total_images;

    imgs->indices = malloc(sizeof(int) * imgs->total_images);
    for (int i = 0; i < imgs->total_images; i++)
    {
        imgs->indices[i] = i;
    }
    shuffle_indices(imgs);

    // Allocate memory to store the image and label
    imgs->images = malloc(sizeof(float *) * imgs->total_images);
    for (int i = 0; i < imgs->total_images; i++)
    {
        imgs->images[i] = malloc(sizeof(float) * num_rows * num_cols);
    }

    imgs->labels = malloc(sizeof(int) * imgs->total_images);

    // READ the labels and images
    for (int i = 0; i < imgs->total_images; i++)
    {
        for (int j = 0; j < imgs->size; j++)
        {
            uint8_t pixel;
            fread(&pixel, sizeof(uint8_t), 1, images_file);
            imgs->images[i][j] = (float)pixel / 255.0;
        }
    }

    long label_offset = sizeof(uint32_t) * 2;   // Skip the headers
    fseek(labels_file, label_offset, SEEK_SET); // Move pointer to start of first image
    for (int i = 0; i < imgs->total_images; i++)
    {
        uint8_t label;
        fread(&label, sizeof(uint8_t), 1, labels_file);
        imgs->labels[i] = (int)label;
    }
}

void return_next_image(IMGs *imgs, float **image, int *label)
{
    if (imgs->current_index == imgs->total_images)
    {
        imgs->current_index = 0;
        shuffle_indices(imgs);
    }
    int index = imgs->indices[imgs->current_index++];
    *image = imgs->images[index];
    *label = imgs->labels[index];
}

void return_next_images(IMGs *imgs, int nb, float *image_matrix, float *label_matrix, int label_matrix_dim)
{
    for (int i = 0; i < nb; i++)
    {
        if (imgs->current_index == imgs->total_images)
        {
            imgs->current_index = 0;
            shuffle_indices(imgs);
        }
        int index = imgs->indices[imgs->current_index++];
        for (int j = 0; j < imgs->size; j++)
        {
            image_matrix[i * imgs->size + j] = imgs->images[index][j];
        }
        for (int j = 0; j < label_matrix_dim; j++)
        {
            if (j == imgs->labels[index])
            {
                label_matrix[i * label_matrix_dim + j] = 1.0;
            }
            else
            {
                label_matrix[i * label_matrix_dim + j] = 0.0;
            }
        }
    }
}

void write_nth_image(IMGs *imgs, float *image_matrix, int n)
{
    // Open a text file for writing
    FILE *output_file = fopen("image.pgm", "w");
    if (output_file == NULL)
    {
        printf("Error opening file for writing.\n");
        return;
    }

    fprintf(output_file, "P2\n");
    fprintf(output_file, "%d %d\n", imgs->rows, imgs->cols);
    fprintf(output_file, "255\n");

    // Write the pixel values to the text file
    for (int i = 0; i < imgs->rows; i++)
    {
        for (int j = 0; j < imgs->cols; j++)
        {
            fprintf(output_file, "%d ", (int)(image_matrix[(n - 1) * imgs->size + imgs->rows * i + j] * 255));
        }
        fprintf(output_file, "\n");
    }

    // Close the text file
    fclose(output_file);
}

#endif // READ_INPUT_H