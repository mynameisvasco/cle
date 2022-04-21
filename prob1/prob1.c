#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "fifo.h"
#include "utf8.h"

struct timespec start, finish;

typedef struct file_result_t
{
  int file_id;
  unsigned int words_number;
  unsigned int words_vowel_start_number;
  unsigned int words_consonant_ending_number;
} file_result_t;

static int files_n = 0;
static int workers_n = 0;
static file_result_t *results;
static pthread_mutex_t *results_mutex;
static pthread_t *workers;
static fifo_t fifo;
static char *files_paths[10];

void init_memory()
{
  workers = malloc(sizeof(pthread_t) * workers_n);
  results = malloc(sizeof(file_result_t) * files_n);
  results_mutex = malloc(sizeof(pthread_mutex_t) * files_n);

  for (int i = 0; i < files_n; i++)
  {
    results[i].words_number = 0;
    results[i].words_vowel_start_number = 0;
    results[i].words_consonant_ending_number = 0;
    pthread_mutex_init(&results_mutex[i], NULL);
  }

  init_fifo(&fifo);
}

void free_shared_memory()
{
  free(workers);
  free(results);
  free(results_mutex);
}

void *worker_lifecycle(void *argp)
{
  while (1)
  {
    int worker_id = *(int *)argp;
    file_chunk_t *chunk = retrieve_fifo(&fifo);

    if (chunk->file_id == SIGNAL_TERMINATE)
      break;

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

    pthread_mutex_lock(&results_mutex[chunk->file_id]);
    results[chunk->file_id].words_number += words_number;
    results[chunk->file_id].words_consonant_ending_number += words_consonant_ending_number;
    results[chunk->file_id].words_vowel_start_number += words_vowel_start_number;
    pthread_mutex_unlock(&results_mutex[chunk->file_id]);
    free(chunk);
  }

  free(argp);
  return NULL;
}

int main(int argc, char **argv)
{
  srandom(time(NULL) * getpid());
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

  init_memory();

  for (int worker_index = 0; worker_index < workers_n; worker_index++)
  {
    int *worker_id = malloc(sizeof(int));
    *worker_id = worker_index;
    pthread_create(&workers[worker_index], NULL, worker_lifecycle, worker_id);
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
        else if (c == 0)
        {
          break;
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

  for (int worker_index = 0; worker_index < workers_n; worker_index++)
  {
    file_chunk_t *terminate_chunk = malloc(sizeof(file_chunk_t));
    terminate_chunk->file_id = SIGNAL_TERMINATE;
    insert_fifo(&fifo, terminate_chunk);
  }

  for (int worker_index = 0; worker_index < workers_n; worker_index++)
  {
    pthread_join(workers[worker_index], NULL);
  }

  for (int file_index = 0; file_index < files_n; file_index++)
  {
    printf("File: %1d\n", file_index);
    printf("Number of words: %d\n", results[file_index].words_number);
    printf("Number of words starting with vowels: %d\n", results[file_index].words_vowel_start_number);
    printf("Number of words ending with consonants: %d\n", results[file_index].words_consonant_ending_number);
  }

  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
  printf("Elapsed time = %.5f s\n", elapsed);
  free_shared_memory();
  return 0;
}