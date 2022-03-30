#include <time.h>
#include <stdlib.h>
#include "utf8.h"
#include "file.h"
#include <stdio.h>
#include <pthread.h>

#define WORKERS_N 3

typedef struct
{
  FILE *file;
  long start;
  long end;
  int words_number;
  int words_vowel_start_number;
  int words_consonant_ending_number;
} worker_unit_t;

static worker_unit_t **workers_units;
static pthread_t *worker_threads;

void *worker(void *argp)
{
  int worker_index = *(int *)argp;
  int files_n = *(int *)(argp + 4);

  for (int i = worker_index * files_n; i < (worker_index + 1) * files_n; i++)
  {
    worker_unit_t *worker_unit = workers_units[i];
    unsigned char buffer[worker_unit->end - worker_unit->start];
    fseek(worker_unit->file, worker_unit->start, SEEK_SET);
    fread(buffer, worker_unit->end - worker_unit->start, 1, worker_unit->file);
    int is_in_word = 0;
    int last_c = 0;
    int read_bytes = 0;
    int c = read_u8char_buffer(buffer, &read_bytes);

    while ((worker_unit->start + read_bytes) <= worker_unit->end && c != 0)
    {
      if (is_in_word && is_separator(c))
      {
        if (is_consonant(last_c))
        {
          worker_unit->words_consonant_ending_number++;
        }

        is_in_word = 0;
      }
      else if (!is_in_word && is_vowel(c))
      {
        is_in_word = 1;
        worker_unit->words_number++;
        worker_unit->words_vowel_start_number++;
      }
      else if (!is_in_word && (is_number(c) || is_consonant(c) || c == '_'))
      {
        is_in_word = 1;
        worker_unit->words_number++;
      }

      if (is_separator(c) || is_vowel(c) || is_number(c) || is_consonant(c) || c == '_' || c == '\'')
      {
        last_c = c;
      }

      c = read_u8char_buffer(buffer, &read_bytes);
    }

    fclose(worker_unit->file);
  }

  free(argp);
  return NULL;
}

int alloc_shared_variables(int files_n)
{
  workers_units = malloc(files_n * WORKERS_N * sizeof(worker_unit_t *));
  worker_threads = malloc(WORKERS_N * sizeof(pthread_t));
  return 0;
}

int free_shared_variables()
{
  free(worker_threads);
  return 0;
}

int main(int argc, char **argv)
{
  clock_t begin = clock();

  if (argc < 2)
  {
    printf("Wrong number of arguments\n, Usage: ./a.out test1.txt test2.txt ...\n");
    exit(-1);
  }

  int files_n = argc - 1;
  long **partitions = get_files_partitions(WORKERS_N, argv, files_n);
  alloc_shared_variables(files_n);

  for (int file_index = 0; file_index < files_n; file_index++)
  {
    for (long worker_index = 0; worker_index < WORKERS_N; worker_index++)
    {
      worker_unit_t *worker_unit = malloc(sizeof(worker_unit_t));
      worker_unit->file = fopen(argv[file_index + 1], "rb");
      worker_unit->start = partitions[file_index][worker_index];
      worker_unit->end = partitions[file_index][worker_index + 1];
      worker_unit->words_consonant_ending_number = 0;
      worker_unit->words_vowel_start_number = 0;
      worker_unit->words_number = 0;
      workers_units[worker_index * files_n + file_index] = worker_unit;
    }
  }

  for (int thread_index = 0; thread_index < WORKERS_N; thread_index++)
  {
    int *args = malloc(sizeof(int) * 2);
    args[0] = thread_index;
    args[1] = files_n;
    pthread_create(&worker_threads[thread_index], NULL, worker, args);
  }

  for (int thread_index = 0; thread_index < WORKERS_N; thread_index++)
  {
    pthread_join(worker_threads[thread_index], NULL);
  }

  for (int file_index = 0; file_index < files_n; file_index++)
  {
    int words_number = 0;
    int words_vowel_start_number = 0;
    int words_consonant_ending_number = 0;

    for (long partition = 0; partition < WORKERS_N; partition++)
    {
      worker_unit_t *worker_unit = workers_units[partition * files_n + file_index];
      words_number += worker_unit->words_number;
      words_vowel_start_number += worker_unit->words_vowel_start_number;
      words_consonant_ending_number += worker_unit->words_consonant_ending_number;
      free(worker_unit);
    }

    printf("File: %1d\n", file_index);
    printf("Number of words: %d\n", words_number);
    printf("Number of words starting with vowels: %d\n", words_vowel_start_number);
    printf("Number of words ending with consonants: %d\n", words_consonant_ending_number);
  }

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Elapsed time = %.3f s\n", time_spent);
  free_shared_variables();
  free(partitions);
  free(workers_units);
  return 0;
}