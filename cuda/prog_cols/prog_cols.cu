#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "matrix.h"
#include "common.h"

void setup_device();

int process_cmd(char **files_paths, int argc, char **argv);

int read_matrices(double *matrix, char **files_paths, int files_n);

__global__ static void process_matrices(double *matrix, double *results, int order);

struct timespec start, finish;

int main(int argc, char **argv)
{
    setup_device();
    char *files_paths[10];
    int files_n = process_cmd(files_paths, argc, argv);
    double *matrices_host = (double *)malloc(sizeof(double) * 128 * 32 * 32);
    double *results_host = (double *)malloc(sizeof(double) * 128);

    for (int i = 0; i < 128; i++)
    {
        results_host[i] = 1;
    }

    double *matrices_device;
    double *results_device;
    int matrices_count = read_matrices(matrices_host, files_paths, files_n);
    CHECK(cudaMalloc(&matrices_device, sizeof(double) * 128 * 32 * 32));
    CHECK(cudaMemcpy(matrices_device, matrices_host, sizeof(double) * 128 * 32 * 32, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&results_device, sizeof(double) * 128));
    CHECK(cudaMemcpy(results_device, results_host, sizeof(double) * 128, cudaMemcpyHostToDevice));
    free(matrices_host);

    dim3 grid, block;
    grid.x = 128;
    grid.y = 1;
    grid.z = 1;
    block.x = 32;
    block.y = 1;
    block.z = 1;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    double elapsed;
    process_matrices<<<grid, block>>>(matrices_device, results_device, 32);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    results_host = (double *)malloc(sizeof(double) * 128);
    CHECK(cudaMemcpy(results_host, results_device, sizeof(double) * 128, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 128; i++)
    {
        printf("det(Matrix %d) = %.3e\n", i + 1, results_host[i]);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Elapsed time = %.5f s\n", elapsed);
    free(results_host);
}

void setup_device()
{
    int dev = 0;
    int i;

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}

int process_cmd(char **files_paths, int argc, char **argv)
{
    int files_n = 0;
    int input = 0;

    while (input != -1)
    {
        input = getopt(argc, argv, "f:");
        if (input == 'f')
        {
            files_paths[files_n++] = optarg;
        }
    }

    return files_n;
}

int read_matrices(double *matrices, char **files_paths, int files_n)
{
    for (int file_index = 0; file_index < files_n; file_index++)
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

        int num_read = fread(matrices, sizeof(double), matrix_count * matrix_order * matrix_order, file);

        if (num_read != matrix_count * matrix_order * matrix_order)
        {
            perror("Error reading values from matrix!\n");
            exit(2);
        }
    }
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