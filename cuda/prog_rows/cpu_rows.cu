#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void transformation(double *kj, double ki, double ii, double ij)
{
    *kj = *kj - ((ki / ii) * ij);
}

// Looks for matrix coefficients on the upper triangule with value different from 0
// and transforms the matrix on an upper trigular matrix
double process_matrix(int id, double *matrix, int order)
{
    // Identifier of the matrix
    int matrix_id = id;

    // Identifier of the row to be processed
    int col_id;

    // Start of the matrix
    int mat = matrix_id * order * order;

    // Start of row
    double result = 1.0;

    for (col_id = 0; col_id < order; col_id++)
    {
        int col = mat + col_id;
        for (int i = 0; i < order; i++)
        {
            int current_col = mat + i;

            if (col_id < i)
            {
                continue;
            }

            if (col_id == i)
            {
                result *= matrix[current_col + i * order];
                continue;
            }

            for (int j = i + 1; j < order; j++)
            {
                matrix[col + j * order] -= matrix[current_col + j * order] * matrix[col + i * order] / matrix[current_col + i * order];
            }
        }
    }
    return result;
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
    int count, order;
    if (!fread(&count, sizeof(int), 1, file))
    {
        printf("Error reading number of matrices!");
        exit(1);
    }
    if (!fread(&order, sizeof(int), 1, file))
    {
        printf("Error reading matrices order!");
        exit(1);
    }

    // printf("Number of matrices to be read: %d\n", count);
    // printf("Matrices order: %d\n\n", order);

    double *matrix = (double *)malloc(sizeof(double) * order * order * count);

    if (fread(matrix, sizeof(double), order * order * count, file) != (order * order * count))
    {
        printf("Error reading values from matrix!");
        exit(2);
    }

    clock_t begin = clock();
    for (int i = 0; i < count; i++)
    {
        double det = process_matrix(i, matrix, order);
        // printf("det(Matrix %d) = %.3e\n", i + 1, det);
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed time = %.6f s\n", time_spent);

    free(matrix);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Wrong number of arguments\n, Usage: ./a.out mat1.bin mat2.bin ...\n");
        exit(-1);
    }

    for (int i = 1; i < argc; i++)
    {
        process_file(argv[i]);
    }
    return 0;
}