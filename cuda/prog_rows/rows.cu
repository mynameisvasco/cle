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

    // declare storage variables
    int files_n;
    char *files_path[10];
    double *results;
    double *results_gpu;
    double *matrix;
    double *matrix_gpu;
    int input = 0;

    // parse command line arguments
    while (input != -1)
    {
        input = getopt(argc, argv, "f:");
        if (input == 'f')
            files_paths[files_n++] = optarg;
    }

    // process each file individually
    for (int file_index = 0; i < files_n; i++)
    {
        FILE *file = fopen(files_paths[file_index], "rb");

        if (file == NULL)
        {
            perror("File could not be read");
            exit(1);
        }

        // get the number o f matrices to process and their order
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

        // allocate space on the Host for the matrices
        matrix = (double *)malloc(sizeof(double) * matrix_count * matrix_order * matrix_order);

        // read the remaining contents of the file (all the matrices coefficients)
        int num_read = fread(matrix, sizeof(double), matrix_order * matrix_order * matrix_count, file);
        if (num_read != (matrix_order * matrix_order * matrix_count))
        {
            perror("Error reading values from matrix!\n");
            exit(2);
        }

        // prepare launching grid
        dim3 grid, block;
        grid.x = matrix_count;
        grid.y = 1;
        grid.z = 1;
        block.x = matrix_order;
        block.y = 1;
        block.z = 1;

        // allocate space for Host results
        results = (double *)malloc(sizeof(double) * matrix_count * matrix_order);

        // initialize Host results with 1
        memset(results, 1, sizeof(double) * matrix_count * matrix_order);

        // allocate space on the Device for the matrices and the results
        cudaMalloc(results_gpu, sizeof(double) * matrix_count * matrix_order);
        cudaMalloc(matrix_gpu, sizeof(double) * matrix_count * matrix_order * matrix_order);

        // copy matrices and results to the Device
        cudaMemcpy(matrix, matrix_gpu, sizeof(double) * matrix_count * matrix_order * matrix_order, cudaMemcpyHostToDevice);
        cudaMemcpy(results, results_gpu, sizeof(double) * matrix_count * matrix_order, cudaMemcpyHostToDevice);

        // free matrices on Host
        free(matrix);

        // process file on GPU
        process_file<<<grid, block>>>(matrix, results, matrix_order);

        // wait for processing to finish
        cudaDeviceSynchronize();

        // free matrices on Device
        cudaFree(matrix_gpu);

        // copy results to Host
        cudaMemcpy(results, results_gpu, sizeof(double) * matrix_count * matrix_order, cudaMemcpyDeviceToHost);

        // free results on Device
        cudaFree(results_gpu);

        // print results
        printf("\n\nProcessing file: %s\n", argv[file_index + 1]);
        printf("Number of matrices to be read: %d\n", matrix_count);
        printf("Matrices order: %d\n\n", matrix_order);

        for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++)
        {
            printf("Processing matrix %d\n", matrix_index + 1);
            double determinant = 1.0;
            for (int res = 0; res < matrix_order; res++)
            {
                determinant *= results[matrix_index * matrix_order + res];
            }
            printf("The determinant is %.3e\n", res);
        }

        free(results);
    }
}