#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "fifo.h"
#include "matrix.h"

struct timespec start, finish;

static int files_n = 0;
static int workers_n = 0;
static fifo_t fifo;
static char *files_paths[10];
static double **results;
static pthread_t *workers;

void init_memory()
{
    workers = malloc(sizeof(pthread_t) * workers_n);
    results = malloc(sizeof(double *) * files_n);
    init_fifo(&fifo);
}

void free_memory()
{
    for (int i = 0; i < files_n; i++)
    {
        free(results[i]);
    }

    free(workers);
    free(results);
}

void *worker_lifecycle(void *argp)
{
    while (1)
    {
        matrix_t *matrix = retrieve_fifo(&fifo);

        if (matrix->matrix_id == SIGNAL_TERMINATE)
        {
            free(matrix);
            break;
        }

        int signal_reversion = 0;
        double determinant = 1;

        for (int i = 0; i < matrix->order; i++)
        {
            int index = matrix->order * i + i;

            if (matrix->values[index] == 0)
            {
                for (int j = i + 1; j < matrix->order; j++)
                {
                    if (matrix->values[index] != 0)
                    {
                        swap_columns(matrix->values, &i, &j, matrix->order);
                        signal_reversion = !signal_reversion;
                        break;
                    }
                }
            }

            for (int j = matrix->order - 1; j > i - 1; j--)
            {
                for (int k = i + 1; k < matrix->order; k++)
                {
                    transformation(&matrix->values[matrix->order * k + j], matrix->values[matrix->order * k + i], matrix->values[matrix->order * i + i], matrix->values[matrix->order * i + j]);
                }
            }

            if (matrix->values[matrix->order * i + i] == 0)
                return 0;

            determinant *= matrix->values[matrix->order * i + i];
        }

        results[matrix->file_id][matrix->matrix_id] = determinant;
        free(matrix->values);
        free(matrix);
    }
}

int main(int argc, char **argv)
{

    clock_gettime(CLOCK_MONOTONIC, &start);
    double elapsed;
    int input = 0;

    while (input != -1)
    {
        input = getopt(argc, argv, "t:i:");
        if (input == 't')
            workers_n = atoi(optarg);
        else if (input == 'i')
            files_paths[files_n++] = optarg;
    }

    int matrices_size[files_n];
    int matrices_order[files_n];
    init_memory();

    for (int worker_index = 0; worker_index < workers_n; worker_index++)
    {
        pthread_create(&workers[worker_index], NULL, worker_lifecycle, NULL);
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

        results[file_index] = malloc(sizeof(double) * matrix_count);
        matrices_size[file_index] = matrix_count;
        matrices_order[file_index] = matrix_order;

        for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++)
        {
            int num_read = 0;
            matrix_t *matrix = malloc(sizeof(matrix_t));
            matrix->values = malloc(sizeof(double) * matrix_order * matrix_order);

            if ((num_read = fread(matrix->values, sizeof(double), matrix_order * matrix_order, file)) != (matrix_order * matrix_order))
            {
                perror("Error reading values from matrix!\n");
                exit(2);
            }

            matrix->file_id = file_index;
            matrix->matrix_id = matrix_index;
            matrix->order = matrix_order;
            insert_fifo(&fifo, matrix);
        }

        fclose(file);
    }

    for (int i = 0; i < workers_n; i++)
    {
        matrix_t *data = malloc(sizeof(matrix_t));
        data->matrix_id = SIGNAL_TERMINATE;
        insert_fifo(&fifo, data);
    }

    for (int worker_index = 0; worker_index < workers_n; worker_index++)
    {
        pthread_join(workers[worker_index], NULL);
    }

    for (int file_index = 0; file_index < files_n; file_index++)
    {
        printf("\n\nProcessing file: %s\n", argv[file_index + 1]);
        printf("Number of matrices to be read: %d\n", matrices_size[file_index]);
        printf("Matrices order: %d\n\n", matrices_order[file_index]);

        for (int mat_id = 0; mat_id < matrices_size[file_index]; mat_id++)
        {
            printf("Processing matrix %d\n", mat_id);
            printf("The determinant is %.3e\n", results[file_index][mat_id]);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Elapsed time = %.5f s\n", elapsed);
    free_memory();
    return 0;
}