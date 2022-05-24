#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include "fifo.h"
#include "utf8.h"

void dispatcher(int argc, char *argv[], int workers_size);

void worker(int worker);

void *populator_lifecycle(void *argp);

MPI_Datatype MPI_CREATE_FILE_CHUNK_TYPE();
MPI_Datatype MPI_CREATE_FILE_RESULTS_TYPE();

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

  MPI_Datatype file_chunk_type = MPI_CREATE_FILE_CHUNK_TYPE();
  MPI_Datatype file_results_type = MPI_CREATE_FILE_RESULTS_TYPE();

  int workers_size = *(int *)argp;
  int worker_index = 0;
  int finished_workers = 0;
  file_chunk_t *chunks[workers_size];
  file_result_t results[5];
  file_result_t partial_results[workers_size];
  int recv_flags[workers_size];
  int send_flags[workers_size];
  int recv_finished[workers_size];
  int send_finished[workers_size];
  MPI_Request send_requests[workers_size];
  MPI_Request recv_requests[workers_size];

  for (int i = 0; i < 1; i++)
  {
    results[i].file_id = i;
    results[i].words_number = 0;
    results[i].words_vowel_start_number = 0;
    results[i].words_consonant_ending_number = 0;
  }

  for (int i = 0; i < workers_size; i++)
  {
    partial_results[i].file_id = 0;
    partial_results[i].words_number = 0;
    partial_results[i].words_vowel_start_number = 0;
    recv_flags[i] = 1;
    send_flags[i] = 1;
    recv_finished[i] = 0;
    send_finished[i] = 0;
  }

  while (1)
  {
    if ((send_flags[worker_index]) && !send_finished[worker_index])
    {
      chunks[worker_index] = retrieve_fifo(&fifo);

      if (chunks[worker_index]->file_id == -1)
      {
        send_finished[worker_index] = 1;
      }

      MPI_Isend(chunks[worker_index], 1, file_chunk_type, worker_index + 1, 0, MPI_COMM_WORLD, &send_requests[worker_index]);
    }

    if ((recv_flags[worker_index]) && !recv_finished[worker_index])
    {
      MPI_Irecv(&partial_results[worker_index], 1, file_results_type, worker_index + 1, 0, MPI_COMM_WORLD, &recv_requests[worker_index]);
    }

    MPI_Test(&send_requests[worker_index], &send_flags[worker_index], MPI_STATUS_IGNORE);
    MPI_Test(&recv_requests[worker_index], &recv_flags[worker_index], MPI_STATUS_IGNORE);

    if ((recv_flags[worker_index]) && !recv_finished[worker_index])
    {
      if (partial_results[worker_index].file_id == -1)
      {
        recv_finished[worker_index] = 1;
        finished_workers++;
      }
      else
      {
        int file_id = partial_results[worker_index].file_id;
        results[file_id].words_number += partial_results[worker_index].words_number;
        results[file_id].words_vowel_start_number += partial_results[worker_index].words_vowel_start_number;
        results[file_id].words_consonant_ending_number += partial_results[worker_index].words_consonant_ending_number;
      }
    }

    worker_index = (worker_index + 1) % workers_size;

    if (finished_workers == workers_size)
    {
      for (int file_index = 0; file_index < 5; file_index++)
      {
        printf("File: %1d\n", file_index);
        printf("Number of words: %d\n", results[file_index].words_number);
        printf("Number of words starting with vowels: %d\n", results[file_index].words_vowel_start_number);
        printf("Number of words ending with consonants: %d\n", results[file_index].words_consonant_ending_number);
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
    input = getopt(argc, argv, "i:");

    if (input == 'i')
    {
      files_paths[files_n++] = optarg;
    }
  }

  if (files_n == 0)
  {
    printf("This application is meant to be run with at least 1 file.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  for (int file_index = 0; file_index < files_n; file_index++)
  {
    char *file_path = files_paths[file_index];
    FILE *file = fopen(file_path, "rb");
    unsigned int c;

    do
    {
      file_chunk_t *chunk = malloc(sizeof(file_chunk_t));
      unsigned int last_separator_index = 0;
      long backward_bytes = 0;

      for (unsigned int c_index = 0; c_index < CHUNK_SIZE; c_index++)
      {
        c = read_u8char(file);

        if (is_separator(c))
        {
          last_separator_index = c_index;
        }

        chunk->file_id = file_index;
        chunk->buffer[c_index] = c;
      }
      for (unsigned int c_index = last_separator_index + 1; c_index < CHUNK_SIZE; c_index++)
      {
        if (chunk->buffer[c_index] != 0)
        {
          backward_bytes += get_needed_bytes(chunk->buffer[c_index]);
          chunk->buffer[c_index] = 0;
        }
      }

      fseek(file, -backward_bytes, SEEK_CUR);
      insert_fifo(&fifo, chunk);
    } while (c != 0);

    fclose(file);
  }

  for (int worker_index = 0; worker_index < workers_size; worker_index++)
  {
    file_chunk_t *chunk = malloc(sizeof(file_chunk_t));
    chunk->file_id = -1;
    insert_fifo(&fifo, chunk);
  }
}

void worker(int rank)
{
  MPI_Datatype file_chunk_type = MPI_CREATE_FILE_CHUNK_TYPE();
  MPI_Datatype file_results_type = MPI_CREATE_FILE_RESULTS_TYPE();

  while (1)
  {
    file_chunk_t *chunk = malloc(sizeof(file_chunk_t));
    MPI_Recv(chunk, 1, file_chunk_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (chunk->file_id == -1)
    {
      file_result_t *results = malloc(sizeof(file_result_t));
      results->file_id = -1;
      results->words_consonant_ending_number = 0;
      results->words_number = 0;
      results->words_vowel_start_number = 0;
      MPI_Send(results, 1, file_results_type, 0, 0, MPI_COMM_WORLD);
      break;
    }

    int is_in_word = 0;
    unsigned int last_c = 0;
    unsigned int words_consonant_ending_number = 0;
    unsigned int words_vowel_start_number = 0;
    unsigned int words_number = 0;

    for (int i = 0; i < CHUNK_SIZE; i++)
    {
      unsigned int c = chunk->buffer[i];

      if (c == 0)
        break;

      if (is_in_word && is_separator(c))
      {
        if (is_consonant(last_c))
        {
          words_consonant_ending_number++;
        }

        is_in_word = 0;
      }
      else if (!is_in_word && is_vowel(c))
      {
        is_in_word = 1;
        words_number++;
        words_vowel_start_number++;
      }
      else if (!is_in_word && (is_number(c) || is_consonant(c) || c == '_'))
      {
        is_in_word = 1;
        words_number++;
      }

      if (is_separator(c) || is_vowel(c) || is_number(c) || is_consonant(c) || c == '_' || c == '\'')
      {
        last_c = c;
      }
    }

    file_result_t *results = malloc(sizeof(file_result_t));
    results->file_id = chunk->file_id;
    results->words_consonant_ending_number = words_consonant_ending_number;
    results->words_number = words_number;
    results->words_vowel_start_number = words_vowel_start_number;
    MPI_Send(results, 1, file_results_type, 0, 0, MPI_COMM_WORLD);
    free(results);
    free(chunk);
  }
}

MPI_Datatype MPI_CREATE_FILE_CHUNK_TYPE()
{
  MPI_Datatype file_chunk_type;
  int lengths[2] = {1, CHUNK_SIZE};
  MPI_Aint displacements[2];
  struct file_chunk_t dummy;
  MPI_Aint base_address;
  MPI_Get_address(&dummy, &base_address);
  MPI_Get_address(&dummy.file_id, &displacements[0]);
  MPI_Get_address(&dummy.buffer, &displacements[1]);
  displacements[0] = MPI_Aint_diff(displacements[0], base_address);
  displacements[1] = MPI_Aint_diff(displacements[1], base_address);
  MPI_Datatype types[2] = {MPI_INT, MPI_UNSIGNED};
  MPI_Type_create_struct(2, lengths, displacements, types, &file_chunk_type);
  MPI_Type_commit(&file_chunk_type);
  return file_chunk_type;
}

MPI_Datatype MPI_CREATE_FILE_RESULTS_TYPE()
{
  MPI_Datatype file_results_type;
  int lengths[4] = {1, 1, 1, 1};
  MPI_Aint displacements[4];
  struct file_result_t dummy;
  MPI_Aint base_address;
  MPI_Get_address(&dummy, &base_address);
  MPI_Get_address(&dummy.file_id, &displacements[0]);
  MPI_Get_address(&dummy.words_number, &displacements[1]);
  MPI_Get_address(&dummy.words_vowel_start_number, &displacements[2]);
  MPI_Get_address(&dummy.words_consonant_ending_number, &displacements[3]);
  displacements[0] = MPI_Aint_diff(displacements[0], base_address);
  displacements[1] = MPI_Aint_diff(displacements[1], base_address);
  displacements[2] = MPI_Aint_diff(displacements[2], base_address);
  displacements[3] = MPI_Aint_diff(displacements[3], base_address);
  MPI_Datatype types[4] = {MPI_INT, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED};
  MPI_Type_create_struct(4, lengths, displacements, types, &file_results_type);
  MPI_Type_commit(&file_results_type);
  return file_results_type;
}