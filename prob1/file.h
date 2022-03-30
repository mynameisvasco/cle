#include <stdio.h>
#include <stdlib.h>

size_t get_file_size(FILE *file);

long *get_file_partitions(int n, FILE *file);

long **get_files_partitions(int n, char **files_paths, int files_n);