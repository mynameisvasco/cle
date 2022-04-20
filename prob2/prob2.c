#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fifo.h"

#define WORKERS_N 4

static fifo_t *queue;
static double **results;
static pthread_mutex_t **results_mutex;
static pthread_t workers[WORKERS_N];

void init_shared_memory(int num_files)
{
    queue = malloc(sizeof(fifo_t));
    results = malloc(sizeof(double *) * num_files);
    results_mutex = malloc(sizeof(pthread_mutex_t *) * num_files);

    init_fifo(queue, 256);
}

void init_individual_memory(unsigned int file, unsigned int matrix_count)
{
    results[file] = malloc(sizeof(double) * matrix_count);
    results_mutex[file] = malloc(sizeof(pthread_mutex_t) * matrix_count);
}

void free_memory(unsigned int num_files)
{
    for (int i = 0; i < num_files; i++)
    {
        free(results[i]);
        free(results_mutex[i]);
    }
    free(queue->array);
    free(queue);
    free(results);
    free(results_mutex);
}

// Swaps the values of columns X and Y
void swap_columns(double *matrix, int *x, int *y, int size)
{
    for (int i = 0; i < size; i++)
    {
        double tmp = matrix[size * i + (*x)];
        matrix[size * i + (*x)] = matrix[size * i + (*y)];
        matrix[size * i + (*y)] = tmp;
    }
}

// Applies Gaussian Elimination
void transformation(double *kj, double ki, double ii, double ij)
{
    *kj = *kj - ((ki / ii) * ij);
}

void *process_matrix(void *argp)
{
    while (1)
    {
        matrix_t matrix = retrieve_fifo(queue);

        int width = matrix.order;
        int height = matrix.order;
        int signal_reversion = 0;
        double determinant = 1;
        for (int i = 0; i < height; i++)
        {
            if (matrix.values[width * i + i] == 0)
            {

                for (int j = i + 1; j < width; j++)
                {
                    if (matrix.values[width * i + j] != 0)
                    {
                        swap_columns(matrix.values, &i, &j, width);
                        signal_reversion = !signal_reversion;
                        break;
                    }
                }
            }
            for (int j = width - 1; j > i - 1; j--)
            {
                for (int k = i + 1; k < width; k++)
                {
                    transformation(&matrix.values[width * k + j], matrix.values[width * k + i], matrix.values[width * i + i], matrix.values[width * i + j]);
                }
            }
            if (matrix.values[width * i + i] == 0)
                return 0;
            determinant *= matrix.values[width * i + i];
        }
        pthread_mutex_lock(&(*results_mutex[matrix.file_id]));
        results[matrix.file_id][matrix.matrix_id] = determinant;
        pthread_mutex_unlock(&(*results_mutex[matrix.file_id]));
    }
}

int main(int argc, char **argv)
{
    clock_t begin = clock();
    int num_files = argc - 1;
    int matrices_size[num_files];
    int matrices_order[num_files];
    if (argc < 2)
    {
        printf("Wrong number of arguments\n, Usage: ./a.out mat128_32.bin mat128_64.bin ...\n");
        exit(-1);
    }

    init_shared_memory(num_files);

    for (int worker_index = 0; worker_index < WORKERS_N; worker_index++)
    {
        pthread_create(&workers[worker_index], NULL, process_matrix, NULL);
    }

    for (int file_index = 0; file_index < num_files; file_index++)
    {
        FILE *file = fopen(argv[file_index + 1], "rb");

        if (file == NULL)
        {
            printf("File could not be read");
            exit(1);
        }

        int matrix_count, matrix_order;
        if (fread(&matrix_count, sizeof(int), 1, file) != 1)
        {
            printf("Error reading number of matrices!");
            exit(1);
        }
        if (fread(&matrix_order, sizeof(int), 1, file) != 1)
        {
            printf("Error reading matrices order!");
            exit(1);
        }

        init_individual_memory(file_index, matrix_count);
        matrices_size[file_index] = matrix_count;
        matrices_order[file_index] = matrix_order;

        double *matrix = malloc(sizeof(double) * matrix_order * matrix_order);

        matrix_t data;
        int cont = 1;
        while (matrix_count-- > 0)
        {
            for (int i = 0; i < matrix_order; i++)
            {
                if (fread(matrix, sizeof(double), matrix_order * matrix_order, file) != (matrix_order * matrix_order))
                {
                    printf("Error reading values from matrix!");
                    exit(2);
                }
            }
            data.file_id = file_index;
            data.matrix_id = cont++;
            data.values = matrix;
            data.order = matrix_order;
            insert_fifo(queue, data);
            free(matrix);
        }

        fclose(file);
    }

    for (int worker_index = 0; worker_index < WORKERS_N; worker_index++)
    {
        pthread_join(workers[worker_index], NULL);
    }

    for (int file_index = 0; file_index < num_files; file_index++)
    {
        printf("Number of matrices to be read: %d\n", matrices_size[file_index]);
        printf("Matrices order: %d\n\n", matrices_order[file_index]);
        for (int mat_id = 0; mat_id < matrices_size[file_index]; mat_id++)
        {
            printf("Processing matrix %d\n", mat_id);
            printf("The determinant is %.3e\n", results[file_index][mat_id]);
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed time = %.5f s\n", time_spent);
    free_memory(num_files);
    return 0;
}