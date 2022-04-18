#include <stdio.h>

#define CHUNK_SIZE 4096

typedef struct file_chunk_t
{
  int file_id;
  unsigned int buffer[CHUNK_SIZE];
} file_chunk_t;

int is_vowel(unsigned int c);

int is_consonant(unsigned int c);

int is_separator(unsigned int c);

int is_number(unsigned int c);

int read_u8char(FILE *file);

long get_needed_bytes(unsigned int n);

file_chunk_t *get_file_chunk(FILE **files, int *current_file, int files_n);