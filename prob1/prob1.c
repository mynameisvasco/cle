#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include "utf8.h"

struct timespec start, finish;

typedef struct file_result_t
{
  int file_id;
  unsigned int words_number;
  unsigned int words_vowel_start_number;
  unsigned int words_consonant_ending_number;
} file_result_t;

static FILE *files[10];
static int current_file = 0;
static int workers_n = 4;
static int files_n = 0;
static file_result_t *results;
static pthread_mutex_t *results_mutex;
static pthread_t *workers;

void init_shared_memory(unsigned int files_n)
{
  workers = malloc(sizeof(pthread_t) * workers_n);
  results = malloc(sizeof(file_result_t) * files_n);
  results_mutex = malloc(sizeof(pthread_mutex_t) * files_n);

  for (unsigned int i = 0; i < files_n; i++)
  {
    results[i].words_number = 0;
    results[i].words_vowel_start_number = 0;
    results[i].words_consonant_ending_number = 0;
    pthread_mutex_init(&results_mutex[i], NULL);
  }
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
    file_chunk_t *chunk = get_file_chunk(files, &current_file, files_n);

    if (chunk == NULL)
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
          words_consonant_ending_number++;

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
        last_c = c;
    }

    pthread_mutex_lock(&results_mutex[chunk->file_id]);
    results[chunk->file_id].words_number += words_number;
    results[chunk->file_id].words_consonant_ending_number += words_consonant_ending_number;
    results[chunk->file_id].words_vowel_start_number += words_vowel_start_number;
    pthread_mutex_unlock(&results_mutex[chunk->file_id]);
    free(chunk);
  }

  return NULL;
}

int main(int argc, char **argv)
{
  clock_gettime(CLOCK_MONOTONIC, &start);
  double elapsed;

  if (argc < 2)
  {
    printf("Wrong number of arguments\n, Usage: ./a.out test1.txt test2.txt ...\n");
    exit(-1);
  }

  int c = 0;

  while (c != -1)
  {
    c = getopt(argc, argv, "t:i:");
    if (c == 't')
      workers_n = atoi(optarg);
    else if (c == 'i')
      files[files_n++] = fopen(optarg, "rb");
  }

  init_shared_memory(files_n);

  for (int worker_index = 0; worker_index < workers_n; worker_index++)
  {
    if (pthread_create(&workers[worker_index], NULL, worker_lifecycle, NULL))
    {
      printf("Thread creation error\n");
      exit(1);
    }
  }

  for (int worker_index = 0; worker_index < workers_n; worker_index++)
  {
    if (pthread_join(workers[worker_index], NULL))
    {
      printf("Thread join error\n");
      exit(1);
    }
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