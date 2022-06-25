#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "common.h"

__global__ static void process_matrices(double *matrix, double *results, int order);

/*
 * Main method of the program
 *
 * Arguments - Runtime arguments
 */
int main(int argc, char **argv)
{
    struct timespec start, finish;
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // declare storage variables
    int files_n = 0;
    char *files_paths[10] = {0};
    double *results;
    double *results_gpu;
    double *matrix;
    double *matrix_gpu;
    int input = 0;
    double elapsed, total_elapsed = 0.0;

    // parse command line arguments
    while (input != -1)
    {
        input = getopt(argc, argv, "f:");

        if (input == 'f')
        {
            files_paths[files_n++] = optarg;
        }
    }

    // start timer
    for (int file_index = 0; file_index < files_n; file_index++)
    {
        FILE *file = fopen(files_paths[file_index], "rb");

        if (file == NULL)
        {
            perror("File could not be read");
            exit(1);
        }

        // get the number of matrices to process and their order
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

        // allocate space for Host results
        results = (double *)malloc(sizeof(double) * matrix_count);

        // initialize Host results with 1
        for (int i = 0; i < matrix_count; i++)
        {
            results[i] = 1.0;
        }

        // allocate space on the Device for the matrices and the results
        CHECK(cudaMalloc(&results_gpu, sizeof(double) * matrix_count));
        CHECK(cudaMalloc(&matrix_gpu, sizeof(double) * matrix_count * matrix_order * matrix_order));

        for (int i = 8; i >= 0; i--)
        {
            // copy matrices and results to the Device
            CHECK(cudaMemcpy(matrix_gpu, matrix, sizeof(double) * matrix_count * matrix_order * matrix_order, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(results_gpu, results, sizeof(double) * matrix_count, cudaMemcpyHostToDevice));

            // prepare launching grid
            dim3 grid, block;
            grid.x = 1;
            grid.y = matrix_count;
            grid.z = 1;
            block.x = 1 << i;
            block.y = 1 << 8 - i;
            block.z = 1;

            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
            // process file on GPU
            process_matrices<<<grid, block>>>(matrix_gpu, results_gpu, matrix_order);

            // wait for processing to finish
            CHECK(cudaDeviceSynchronize());

            // stop timer
            clock_gettime(CLOCK_MONOTONIC_RAW, &finish);
            elapsed = (finish.tv_sec - start.tv_sec);
            elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
            total_elapsed += elapsed;
            printf("(%d,%d) - (%d,%d) = %.10f s\n", grid.x, grid.y, block.x, block.y, elapsed);
        }

        // free matrices on Device
        CHECK(cudaFree(matrix_gpu));

        // copy results to Host
        results = (double *)malloc(sizeof(double) * 128);
        CHECK(cudaMemcpy(results, results_gpu, sizeof(double) * matrix_count, cudaMemcpyDeviceToHost));

        // free results on Device
        CHECK(cudaFree(results_gpu));

        // free matrices on Host
        free(matrix);

        // print results
        // printf("\n\nProcessing file: %s\n", files_paths[file_index]);
        // printf("Number of matrices to be read: %d\n", matrix_count);
        // printf("Matrices order: %d\n", matrix_order);
        // printf("Partial %d - Elapsed time = %.5f s\n\n", file_index, elapsed);

        // for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++)
        // {
        //     printf("det(Matrix %d) = %.3e\n", matrix_index + 1, results[matrix_index]);
        // }

        free(results);
    }
    // print total time spent in processing
    printf("Elapsed time = %.10f s\n", total_elapsed);
}

__global__ static void process_matrices(double *matrix, double *results, int order)
{
    // Identifier of the matrix
    int matrix_id = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

    // Identifier of the column to be processed
    int col_id = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

    // Start of the matrix
    int mat = matrix_id * order * order;

    // Start of col
    int col = mat + col_id;

    // apply transformation
    for (int i = 0; i < order; i++)
    {
        int current_col = mat + i;

        if (col_id < i)
        {
            continue;
        }

        if (col_id == i)
        {
            results[matrix_id] *= matrix[current_col + i * order];
            continue;
        }

        for (int j = i + 1; j < order; j++)
        {
            matrix[col + j * order] -= matrix[current_col + j * order] * matrix[col + i * order] / matrix[current_col + i * order];
        }

        __syncthreads();
    }
}