#include <time.h>
#include <stdlib.h>
#include "consts.h"
#include "utf8.h"
#include "file.h"
#include <stdio.h>
#include <pthread.h>

typedef struct
{
  int file_id;
  long start_position;
  long end_position;
  char *buffer;
  int words_number;
} worker_args_t;

static pthread_t *threads;
static worker_args_t *workers_args;

void *worker(void *argp)
{
  worker_args_t *args = (worker_args_t *)argp;

  FILE *partial_file = fmemopen(args->buffer, args->end_position - args->start_position, "rb");
  int last_c = 0;
  int is_in_word = 0;
  int words_number = 0;

  while (1)
  {
    int c = read_u8char(partial_file);

    if (c == 0)
      break;

    if (is_in_word && is_separator(c))
    {
      is_in_word = 0;
    }
    else if (!is_in_word && is_vowel(c))
    {
      is_in_word = 1;
      words_number++;
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

  args->words_number = words_number;
  return NULL;
}

int alloc_shared_variables(int files_n)
{
  threads = malloc(files_n * WORKERS_N * sizeof(pthread_t));
  workers_args = malloc(files_n * WORKERS_N * sizeof(worker_args_t));
  return 0;
}

int free_shared_variables()
{
  free(threads);
  free(workers_args);
  return 0;
}

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    printf("Wrong number of arguments\n, Usage: ./a.out test1.txt test2.txt ...\n");
    exit(-1);
  }

  int files_n = argc - 1;
  long **positions = malloc(files_n * sizeof(long[WORKERS_N + 2]));
  alloc_shared_variables(files_n);
  clock_t begin = clock();

  for (int i = 1; i < argc; i++)
  {
    FILE *file = fopen(argv[i], "rb");

    if (file == NULL)
    {
      printf("File %s not found.\n", argv[i]);
      exit(-1);
    }

    positions[i - 1] = get_file_partitions(WORKERS_N, file);

    for (int j = 0; j < WORKERS_N; j++)
    {
      int index = j * files_n + (i - 1);

      workers_args[index].file_id = i - 1;
      workers_args[index].start_position = positions[i - 1][j];
      workers_args[index].end_position = positions[i - 1][j + 1];
      workers_args[index].words_number = 0;
      long size = workers_args[index].end_position - workers_args[index].start_position;
      workers_args[index].buffer = malloc(sizeof(char) * size);
      fseek(file, workers_args[index].start_position, SEEK_SET);
      fread(workers_args[index].buffer, 1, size, file);
      pthread_create(&threads[index], NULL, worker, &(workers_args[index]));
    }
  }

  for (int i = 1; i < argc; i++)
  {
    for (int j = 0; j < WORKERS_N; j++)
    {
      int index = j * files_n + (i - 1);
      pthread_join(threads[index], NULL);
    }
  }

  int *words_number = calloc(files_n, sizeof(int));

  for (int i = 1; i < argc; i++)
  {
    for (int j = 0; j < WORKERS_N; j++)
    {
      int index = j * files_n + (i - 1);
      words_number[i - 1] += workers_args[index].words_number;
    }

    printf("File: %d\n", i - 1);
    printf("Number of words: %d\n", words_number[i - 1]);
  }

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Elapsed time = %.3f s\n", time_spent);
  free_shared_variables();
  free(words_number);
  return 0;
}