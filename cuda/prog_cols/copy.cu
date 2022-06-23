#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../common.h"
#include "process_file.h"

struct timespec start, finish;

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

    // start timer
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

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
        results = (double *)malloc(sizeof(double) * matrix_count);

        // initialize Host results with 1
        for (int i = 0; i < matrix_count; i++)
        {
            results[i] = 1;
        }

        // allocate space on the Device for the matrices and the results
        cudaMalloc(results_gpu, sizeof(double) * matrix_count);
        cudaMalloc(matrix_gpu, sizeof(double) * matrix_count * matrix_order * matrix_order);

        // copy matrices and results to the Device
        cudaMemcpy(matrix_gpu, matrix, sizeof(double) * matrix_count * matrix_order * matrix_order, cudaMemcpyHostToDevice);
        cudaMemcpy(results_gpu, results, sizeof(double) * matrix_count, cudaMemcpyHostToDevice);

        // free matrices on Host
        free(matrix);

        // process file on GPU
        process_matrices<<<grid, block>>>(matrix, results, matrix_order);

        // wait for processing to finish
        cudaDeviceSynchronize();

        // free matrices on Device
        cudaFree(matrix_gpu);

        // copy results to Host
        cudaMemcpy(results, results_gpu, sizeof(double) * matrix_count, cudaMemcpyDeviceToHost);

        // free results on Device
        cudaFree(results_gpu);

        // print results
        printf("\n\nProcessing file: %s\n", argv[file_index + 1]);
        printf("Number of matrices to be read: %d\n", matrix_count);
        printf("Matrices order: %d\n\n", matrix_order);

        for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++)
        {
            printf("det(Matrix %d) = %.3e\n", matrix_index + 1, results[matrix_index]);
        }
        free(results);
    }

    // stop timer
    clock_gettime(CLOCK_MONOTONIC_RAW, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Elapsed time = %.5f s\n", elapsed);
}

__global__ static void process_matrices(double *matrix, double *results, int order)
{
    // Identifier of the matrix
    int matrix_id = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

    // Identifier of the row to be processed
    int row_id = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

    // Start of the matrix
    double *mat = &matrix[matrix_id * order * order];

    // Start of the row
    double *row = &mat[row_id * order];
    double *ii = row + row_id;
    unsigned int signal_reversion = 0;
    // If position (i,i) is equal to 0 then look for a column different from 0 and swap

    for (int i = 0; i < order; i++)
    {
        if (row_id > i)
        {
            continue;
        }

        if (*ii == 0)
        {
            for (int j = i + 1; j < order; j++)
            {
                if (mat[order * i + j] != 0)
                {
                    for (int k = 0; j < order; k++)
                    {
                        double tmp = mat[order * k + i];
                        mat[order * k + i] = matrix[order * k + j];
                        mat[order * k + j] = tmp;
                    }

                    signal_reversion = !signal_reversion;
                    break;
                }
            }
        }

        // Apply the Guassian transformation
        for (int j = order - 1; j > row_id - 1; j--)
        {
            for (int k = row_id + 1; k < order; k++)
            {
                mat[order * k + j] = mat[order * k + j] - ((mat[order * k + row_id] / *ii) * mat[order * row_id + j]);
            }
        }

        __syncthreads();
    }

    if (threadIdx.x % order == 0)
    {
        for (int i = 0; i < order; i++)
        {
            results[matrix_id] *= mat[i * order + i];
        }
    }
}