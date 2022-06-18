#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../common.h"
#include "process_file.h"

/*
 * Main method of the program
 *
 * Arguments - Runtime arguments
 */
int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int files_n;
    char *files_path[10];
    double *results;
    double *matrix;
    int input = 0;

    while (input != -1)
    {
        input = getopt(argc, argv, "f:");
        if (input == 'f')
            files_paths[files_n++] = optarg;
    }

    for (int file_index = 0; i < files_n; i++)
    {
        FILE *file = fopen(files_paths[file_index], "rb");

        if (file == NULL)
        {
            perror("File could not be read");
            exit(1);
        }

        int matrix_count, matrix_order;
        if (fread(&matrix_count, sizeof(int), 1, file) != 1)
        {
            perror("Error reading number of matrices!");
            exit(1);
        }
        if (fread(&matrix_order, sizeof(int), 1, file) != 1)
        {
            perror("Error reading matrices order!");
            exit(1);
        }

        results = malloc(sizeof(double) * matrix_count);
        matrix = malloc(sizeof(double) * matrix_count * matrix_order * matrix_order);

        int num_read = fread(matrix, sizeof(double), matrix_order * matrix_order * matrix_count, file);
        if (num_read != (matrix_order * matrix_order * matrix_count))
        {
            perror("Error reading values from matrix!\n");
            exit(2);
        }

        dim3 grid, block;
        grid.x = matrix_count;
        grid.y = 1;
        grid.z = 1;
        block.x = matrix_order;
        block.y = 1;
        block.z = 1;

        process_file << grid, block >> (matrix[file_index], results[file_index], matrix_count, matrix_order);

        printf("\n\nProcessing file: %s\n", argv[file_index + 1]);
        printf("Number of matrices to be read: %d\n", matrix_count);
        printf("Matrices order: %d\n\n", matrix_order);

        for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++)
        {
            printf("Processing matrix %d\n", matrix_index + 1);
            printf("The determinant is %.3e\n", results[matrix_index]);
        }
        free(matrix);
        free(results);
    }
}