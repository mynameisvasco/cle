#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include "matrix.h"
#include "fifo.h"

void dispatcher(int argc, char *argv[], int workers_size);

void worker(int worker);

void *populator_lifecycle(void *argp);

static fifo_t fifo;
struct timespec start, finish;

int main(int argc, char *argv[])
{
    int rank, size;
    init_fifo(&fifo);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int workers_size = size - 1;

    if (size < 2)
    {
        printf("This application is meant to be run with 2 or more processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    else if (rank == 0)
    {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        double elapsed;
        pthread_t populator;
        int *_workers_size = malloc(sizeof(int));
        *_workers_size = workers_size;
        pthread_create(&populator, NULL, populator_lifecycle, _workers_size);
        dispatcher(argc, argv, workers_size);
        pthread_join(populator, NULL);
        clock_gettime(CLOCK_MONOTONIC_RAW, &finish);
        elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        printf("Elapsed time = %.5f s\n", elapsed);
    }
    else
    {
        worker(rank);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

void *populator_lifecycle(void *argp)
{
    MPI_Datatype matrix_type = create_matrix_type();
    MPI_Datatype file_results_type = create_result_type();

    int workers_size = *(int *)argp;
    int worker_index = 0;
    int finished_workers = 0;
    matrix_t *matrices[workers_size];
    file_result_t result[workers_size];
    double determinants[4][256];
    int recv_flags[workers_size];
    int send_flags[workers_size];
    int recv_finished[workers_size];
    int send_finished[workers_size];
    MPI_Request send_requests[workers_size];
    MPI_Request recv_requests[workers_size];

    for (int i = 0; i < workers_size; i++)
    {
        recv_flags[i] = 1;
        send_flags[i] = 1;
        recv_finished[i] = 0;
        send_finished[i] = 0;
    }

    while (1)
    {
        if ((send_flags[worker_index]) && !send_finished[worker_index])
        {
            matrices[worker_index] = retrieve_fifo(&fifo);

            if (matrices[worker_index]->file_id == -1)
            {
                send_finished[worker_index] = 1;
            }

            MPI_Isend(matrices[worker_index], 1, matrix_type, worker_index + 1, 0, MPI_COMM_WORLD, &send_requests[worker_index]);
        }

        if ((recv_flags[worker_index]) && !recv_finished[worker_index])
        {
            MPI_Irecv(&result[worker_index], 1, file_results_type, worker_index + 1, 0, MPI_COMM_WORLD, &recv_requests[worker_index]);
        }

        MPI_Test(&send_requests[worker_index], &send_flags[worker_index], MPI_STATUS_IGNORE);
        MPI_Test(&recv_requests[worker_index], &recv_flags[worker_index], MPI_STATUS_IGNORE);

        if ((recv_flags[worker_index]) && !recv_finished[worker_index])
        {
            if (result[worker_index].file_id == -1)
            {
                recv_finished[worker_index] = 1;
                finished_workers++;
            }
            else
            {
                determinants[result[worker_index].file_id][result[worker_index].matrix_id] = result[worker_index].determinant;
            }
        }

        worker_index = (worker_index + 1) % workers_size;
        if (finished_workers == workers_size)
        {
            for (int file_index = 0; file_index < 4; file_index++)
            {
                printf("\nFile: %1d\n\n", file_index);
                for (int matrix_index = 0; matrix_index < 128; matrix_index++)
                {
                    printf("Determinant is %.3e\n", determinants[file_index][matrix_index]);
                }
            }

            break;
        }
    }

    free(argp);
    return NULL;
}

void dispatcher(int argc, char *argv[], int workers_size)
{
    int input = 0;
    int files_n = 0;
    char *files_paths[10];

    while (input != -1)
    {
        input = getopt(argc, argv, "t:i:");
        if (input == 'i')
            files_paths[files_n++] = optarg;
    }

    if (files_n == 0)
    {
        printf("This application is meant to be run with at least 1 file.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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

        for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++)
        {
            int num_read = 0;
            matrix_t *matrix = malloc(sizeof(matrix_t));

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

    for (int i = 0; i < workers_size; i++)
    {
        matrix_t *data = malloc(sizeof(matrix_t));
        data->file_id = -1;
        insert_fifo(&fifo, data);
    }
}

void worker(int rank)
{
    MPI_Datatype matrix_type = create_matrix_type();
    MPI_Datatype file_results_type = create_result_type();

    while (1)
    {
        matrix_t *matrix = malloc(sizeof(matrix_t));
        MPI_Recv(matrix, 1, matrix_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (matrix->file_id == -1)
        {
            matrix->file_id = -1;
            MPI_Send(matrix, 1, file_results_type, 0, 0, MPI_COMM_WORLD);
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
                    if (matrix->values[matrix->order * i + j] != 0)
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
                determinant = 0;

            determinant *= matrix->values[matrix->order * i + i];
        }

        file_result_t *results = malloc(sizeof(file_result_t));
        results->file_id = matrix->file_id;
        results->matrix_id = matrix->matrix_id;
        results->determinant = determinant;
        MPI_Send(results, 1, file_results_type, 0, 0, MPI_COMM_WORLD);
        free(matrix);
    }
}