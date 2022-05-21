#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include "matrix.h"

#define N 1

struct timespec start, finish;
static int files_n = 0;
static char *files_paths[10];

int main(int argc, char **argv)
{
    // clock_gettime(CLOCK_MONOTONIC, &start);
    double elapsed, determinant = 1;
    int input = 0, rank, size, nNorm, n, index, cont = 0;
    int matrix_count, matrix_order, signal_reversion = 0;
    int i, j, k;
    int *send_data = NULL, *recv_data;
    double *matrix;

    for (i = 0; i < argc; i++)
    {
        if (!strcmp(argv[i], "-i") && i != argc - 1)
        {
            files_paths[input++] = argv[i + 1];
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size <= 1 || ((N % size) == 0))
    {
        printf("size = %d\n", size);
        if (rank == 0)
            printf("Wrong number of processes! It must be a submultiple of %d different from 1.\n", N);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    nNorm = N + (((N % size) == 0) ? 0 : size - (N % size));

    for (int file_index = 0; file_index < files_n; file_index++)
    {

        FILE *file = fopen(files_paths[file_index], "rb");

        if (file == NULL)
        {
            perror("File could not be read");
            exit(1);
        }

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

        while (cont++ < matrix_count)
        {
            if (rank == 0)
            {
                printf("\n\nProcessing file: %s\n", argv[file_index + 1]);
                printf("Number of matrices to be read: %d\n", matrix_count);
                printf("Matrices order: %d\n\n", matrix_order);

                matrix = malloc(sizeof(double) * matrix_order * matrix_order);
                if (fread(matrix, sizeof(double), matrix_order * matrix_order, file) != (matrix_order * matrix_order))
                {
                    perror("Error reading values from matrix!\n");
                    exit(2);
                }
            }
            fclose(file);

            n = nNorm / size;

            for (i = 0; i < matrix_order; i++)
            {
                index = matrix_order * i + i;

                if (matrix[index] == 0)
                {
                    for (j = i + 1; j < matrix_order; j++)
                    {
                        if (matrix[index] != 0)
                        {
                            swap_columns(matrix, &i, &j, matrix_order);
                            signal_reversion = !signal_reversion;
                            break;
                        }
                    }
                }

                for (j = matrix_order - 1; j > i - 1; j--)
                {
                    for (k = i + 1; k < matrix_order; k++)
                    {
                        transformation(&matrix[matrix_order * k + j], matrix[matrix_order * k + i], matrix[matrix_order * i + i], matrix[matrix_order * i + j]);
                    }
                }

                if (matrix[matrix_order * i + i] == 0)
                    return 0;

                determinant *= matrix[matrix_order * i + i];
            }

            if (rank == 0)
            {
                free(matrix);

                printf("Processing matrix %d\n", cont);
                printf("The determinant is %.3e\n", determinant);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Elapsed time = %.5f s\n", elapsed);

    MPI_Finalize();

    return EXIT_SUCCESS;
}