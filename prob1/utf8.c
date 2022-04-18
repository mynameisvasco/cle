#include <stdlib.h>
#include <pthread.h>
#include "utf8.h"

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

int is_vowel(unsigned int c)
{
  if (c == 0x61 ||
      c == 0x65 ||
      c == 0x69 ||
      c == 0x6F ||
      c == 0x75 ||
      c == 0x41 ||
      c == 0x45 ||
      c == 0x49 ||
      c == 0x4F ||
      c == 0x55)
    return 1;

  if (c >> 8 != 0xC3)
    return 0;

  int ac_vowels[30] = {
      0xC3A1,
      0xC3A0,
      0xC3A2,
      0xC3A3,
      0xC3A9,
      0xC3A8,
      0xC3AA,
      0xC3AD,
      0xC3AC,
      0xC3B3,
      0xC3B2,
      0xC3B4,
      0xC3B5,
      0xC3BA,
      0xC3B9,
      0xC381,
      0xC380,
      0xC382,
      0xC383,
      0xC389,
      0xC388,
      0xC38A,
      0xC38D,
      0xC38C,
      0xC393,
      0xC392,
      0xC394,
      0xC395,
      0xC39A,
      0xC399};

  for (int i = 0; i < 30; i++)
    if (c == ac_vowels[i])
      return 1;

  return 0;
}

int is_consonant(unsigned int c)
{
  int lower_cedilha = 0xC3A7;
  int upper_cedilha = 0xC387;
  return !is_vowel(c) && (((c >= 65 && c <= 90) || (c >= 97 && c <= 122)) || c == lower_cedilha || c == upper_cedilha);
}

int is_separator(unsigned int c)
{
  int dash = 0xE28093;
  int ellipsis = 0xE280A6;
  int doubleQuotationMarkLeft = 0xE2809C;
  int doubleQuotationMarkRight = 0xE2809D;
  return c == '[' || c == ']' || c == '(' || c == ')' || c == '-' || c == '"' || c == 0x20 || c == 0x9 || c == 0xA || c == 0xD || c == '.' || c == '?' || c == ';' || c == ':' || c == ',' || c == '!' || c == dash || c == ellipsis || c == doubleQuotationMarkLeft || c == doubleQuotationMarkRight;
}

int is_number(unsigned int c)
{
  return c >= 48 && c <= 57;
}

int read_u8char(FILE *file)
{
  unsigned char buffer[4] = {0, 0, 0, 0};
  unsigned int c = 0;
  fread(buffer, 1, 1, file);
  c = buffer[0];

  if ((buffer[0] >> 5) == 0b110)
  {
    fread(&buffer[1], 1, 1, file);
    c = (c << 8) | (buffer[1] & 0xff);
  }
  else if ((buffer[0] >> 4) == 0b1110)
  {
    fread(&buffer[1], 1, 2, file);
    c = (c << 8) | (buffer[1] & 0xff);
    c = (c << 8) | (buffer[2] & 0xff);
  }
  else if ((buffer[0] >> 3) == 0b11110)
  {
    fread(&buffer[1], 1, 3, file);
    c = (c << 8) | (buffer[1] & 0xff);
    c = (c << 8) | (buffer[2] & 0xff);
    c = (c << 8) | (buffer[2] & 0xff);
  }

  return c;
}

file_chunk_t *get_file_chunk(FILE **files, int *current_file, int files_n)
{
  pthread_mutex_lock(&mutex);

  if (*current_file >= files_n)
  {
    pthread_mutex_unlock(&mutex);
    return (void *)NULL;
  }

  FILE *file = files[*current_file];
  file_chunk_t *chunk = malloc(sizeof(file_chunk_t));
  unsigned int c;
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
      *current_file += 1;
      break;
    }

    chunk->file_id = *current_file;
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
  pthread_mutex_unlock(&mutex);
  return chunk;
}

long get_needed_bytes(unsigned int n)
{
  if (n < 256)
    return 1;
  else if (n < 65536)
    return 2;
  else if (n < 16777216)
    return 3;
  return 4;
}