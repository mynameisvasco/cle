#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int SIZE = 3;

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

void transformation(double *kj, double ki, double ii, double ij)
{
    *kj = *kj - ((ki / ii) * ij);
}

// Looks for matrix coefficients on the upper triangule with value different from 0
// and transforms the matrix on an upper trigular matrix
double process_matrix(double (*matrix)[SIZE], int *signal_reversion)
{
    double determinant = 1;
    for (int i = 0; i < SIZE; i++)
    {
        if (matrix[i][i] == 0)
        {

            for (int j = i + 1; j < SIZE; j++)
            {
                if (matrix[j][i] != 0)
                {
                    swap_columns(matrix, &i, &j);
                    *signal_reversion = !(*signal_reversion);
                    break;
                }
            }
        }
        for (int j = SIZE - 1; j > i - 1; j--)
        {
            for (int k = i + 1; k < SIZE; k++)
            {
                transformation(&matrix[k][j], matrix[k][i], matrix[i][i], matrix[i][j]);
            }
        }
        if (matrix[i][i] == 0)
            return 0;
        determinant *= matrix[i][i];
    }
    return determinant;
}

// Reads the binary file where the matrices are stored and decodes the matrices ad additional parameters
void process_file(char *file_path)
{
    FILE *file = fopen(file_path, "rb");

    if (file == NULL)
    {
        printf("File %s was not found!\n", file_path);
    }

    // reads the number of matrixes and the matrixes order (int = 4 bytes)
    int matrix_count, matrix_order;
    int signal_reversion = 0;
    if (!fread(&matrix_count, sizeof(int), 1, file))
    {
        printf("Error reading number of matrices!");
        exit(1);
    }
    if (!fread(&matrix_order, sizeof(int), 1, file))
    {
        printf("Error reading matrices order!");
        exit(1);
    }

    SIZE = matrix_order;

    printf("Number of matrices to be read: %d\n", matrix_count);
    printf("Matrices order: %d\n\n", SIZE);

    double matrix[SIZE][SIZE];
    int cont = 1;
    while (matrix_count-- > 0)
    {
        printf("Processing matrix %d\n", cont++);
        for (int i = 0; i < matrix_order; i++)
        {
            if (!fread(&matrix[i], sizeof(double), matrix_order, file))
            {
                printf("Error reading values from matrix!");
                exit(2);
            }
        }

        double det = process_matrix(matrix, &signal_reversion);

        if (signal_reversion)
            det = -det;
        printf("The determinant is %.3e\n", det);
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