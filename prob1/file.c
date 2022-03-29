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
  size_t file_size = get_file_size(file);
  size_t partition_size = file_size / n;
  long *positions = malloc(sizeof(long) * (n + 2));
  int current_n = 0;
  positions[current_n++] = 0;

  for (int i = 0; i < n; i++)
  {
    fseek(file, (i + 1) * partition_size, SEEK_SET);
    int c = read_u8char(file);

    while (!is_separator(c) && c != 0)
    {
      c = read_u8char(file);
    }

    positions[current_n++] = ftell(file);
  }

  positions[current_n + 1] = file_size;
  fseek(file, 0, SEEK_SET);
  return positions;
}