#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>

struct timespec start, finish;

int main(int argc, char **argv)
{
    clock_gettime(CLOCK_MONOTONIC, &start);
    double elapsed;
    int input = 0, rank, size;
    int *send_data = NULL, *recv_data;

    MPI_Init(&argc, &argv);
    MPI_Comm_Rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_Size(MPI_COMM_WORLD, &size)

        while (input != -1)
    {
        input = getopt(argc, argv, "t:i:");
        if (input == 't')
            workers_n = atoi(optarg);
        else if (input == 'i')
            files_paths[files_n++] = optarg;
    }

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

        printf("\n\nProcessing file: %s\n", argv[file_index + 1]);
        printf("Number of matrices to be read: %d\n", matrix_count);
        printf("Matrices order: %d\n\n", matrix_order);

        for (int matrix_id = 0; matrix_id < matrix_count; matrix_id++)
        {
            if ((num_read = fread(matrix, sizeof(double), matrix_order * matrix_order, file)) != (matrix_order * matrix_order))
            {
                perror("Error reading values from matrix!\n");
                exit(2);
            }

            int signal_reversion = 0;
            double determinant = 1;

            for (int i = 0; i < matrix_order; i++)
            {
                int index = matrix_order * i + i;

                if (matrix_values[index] == 0)
                {
                    for (int j = i + 1; j < matrix_order; j++)
                    {
                        if (matrix[index] != 0)
                        {
                            swap_columns(matrix, &i, &j, matrix_order);
                            signal_reversion = !signal_reversion;
                            break;
                        }
                    }
                }

                for (int j = matrix_order - 1; j > i - 1; j--)
                {
                    for (int k = i + 1; k < matrix_order; k++)
                    {
                        transformation(&matrix[matrix_order * k + j], matrix[matrix_order * k + i], matrix[matrix_order * i + i], matrix[matrix_order * i + j]);
                    }
                }

                if (matrix[matrix_order * i + i] == 0)
                    return 0;

                determinant *= matrix[matrix_order * i + i];
            }
            printf("Processing matrix %d\n", matrix_id);
            printf("The determinant is %.3e\n", determinant);
        }

        fclose(file);
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Elapsed time = %.5f s\n", elapsed);
    free_memory();
    return 0;
}