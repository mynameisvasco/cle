#include <stdint.h>
#include <pthread.h>

#define CHUNK_SIZE 4096
#define FIFO_SIZE 32

typedef struct file_chunk_t
{
  int file_id;
  unsigned int buffer[CHUNK_SIZE];
} file_chunk_t;

typedef struct file_result_t
{
  int file_id;
  unsigned int words_number;
  unsigned int words_vowel_start_number;
  unsigned int words_consonant_ending_number;
} file_result_t;

typedef struct
{
  file_chunk_t *array[FIFO_SIZE];
  unsigned int inp;
  unsigned int out;
  unsigned int cnt;
  pthread_mutex_t mutex;
  pthread_cond_t isNotEmpty;
  pthread_cond_t isNotFull;
} fifo_t;

void init_fifo(fifo_t *fifo);
int empty_fifo(fifo_t *fifo);
int full_fifo(fifo_t *fifo);
void insert_fifo(fifo_t *fifo, file_chunk_t *chunk);
file_chunk_t *retrieve_fifo(fifo_t *fifo);