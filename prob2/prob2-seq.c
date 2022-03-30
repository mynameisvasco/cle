#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int SIZE = 0;

// Swaps the values of columns X and Y
void swap_columns(double (*matrix)[SIZE], int *x, int *y)
{
    for (int i = 0; i < SIZE; i++)
    {
        double tmp = matrix[i][*x];
        matrix[i][*x] = matrix[i][*y];
        matrix[i][*y] = tmp;
    }
}

// Looks for matrix coefficients on the upper triangule with value different from 0
void process_matrix(double (*matrix)[SIZE], int *signal_reversion)
{
    for (int i = 0; i < SIZE - 1; i++)
    {
        if (matrix[i][i] == 0)
        {

            for (int j = i + 1; j < SIZE; j++)
            {
                if (matrix[i][j] != 0)
                {
                    swap_columns(matrix, &i, &j);
                    *signal_reversion = !(*signal_reversion);
                }
            }
        }
    }
}

// Reads the binary file where the matrices are stored and decodes the matrices ad additional parameters
void process_file(char *file_path)
{
    FILE *file = fopen(file_path, "rb");

    if (file == NULL)
    {
        printf("FIle %s was not found!\n", file_path);
    }

    // reads the number of matrixes and the matrixes order (int = 4 bytes)
    int matrix_count, matrix_order;
    int signal_reversion = 0;
    fread(&matrix_count, sizeof(int), 1, file);
    fread(&matrix_order, sizeof(int), 1, file);

    printf("Number of matrices to be read: %d\n", matrix_count);
    printf("Matrices order: %d\n", matrix_order);

    double matrix[matrix_order][matrix_order];

    SIZE = matrix_order;

    while (matrix_count-- > 0)
    {
        for (int i = 0; i < matrix_order; i++)
        {
            fread(&matrix[i], sizeof(double), matrix_order, file);
        }

        process_matrix(matrix, &signal_reversion);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Wrong number of arguments\n, Usage: ./a.out mat1.bin mat2.bin ...\n");
        exit(-1);
    }

    clock_t begin = clock();

    for (int i = 1; i < argc; i++)
    {
        process_file(argv[i]);
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed time = %.3f s\n", time_spent);
    return 0;
}