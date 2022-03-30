#include "file.h"
#include "utf8.h"
#include <stdlib.h>

size_t get_file_size(FILE *file)
{
  long cur_postition = ftell(file);
  fseek(file, 0, SEEK_END);
  size_t size = ftell(file);
  fseek(file, cur_postition, SEEK_SET);
  return size;
}

long *get_file_partitions(int n, FILE *file)
{
  long file_size = get_file_size(file);
  long partition_size = file_size / n;
  long *positions = malloc((n + 2) * sizeof(long));
  int current_n = 0;
  positions[current_n] = 0;

  for (int i = 0; i < n; i++)
  {
    current_n++;
    fseek(file, (i + 1) * partition_size, SEEK_SET);
    int c = read_u8char(file, NULL);

    while (!is_separator(c) && c != 0)
    {
      c = read_u8char(file, NULL);
    }

    positions[current_n] = ftell(file);
  }

  current_n++;
  positions[current_n] = file_size;

  fseek(file, 0, SEEK_SET);
  return positions;
}

long **get_files_partitions(int n, char **files_paths, int files_n)
{
  long **positions = malloc(files_n * sizeof(long *));

  for (int i = 0; i < files_n; i++)
  {
    FILE *file = fopen(files_paths[i + 1], "rb");

    if (file == NULL)
    {
      printf("File %s not found.\n", files_paths[i + 1]);
      exit(-1);
    }

    positions[i] = get_file_partitions(n, file);
  }

  return positions;
}