#include <stdint.h>
#include <pthread.h>

typedef struct matrix_t
{
  int file_id;
  int matrix_id;
  double *values;
  int order;
} matrix_t;

typedef struct fifo_t
{
  matrix_t **array;
  unsigned int inp; // head
  unsigned int out; // tail
  unsigned int cnt; // size
  unsigned int max; // max_size
  pthread_mutex_t mutex;
  pthread_cond_t isNotEmpty;
  pthread_cond_t isNotFull;
} fifo_t;

void init_fifo(fifo_t *fifo, int size);
int empty_fifo(fifo_t *fifo);
int full_fifo(fifo_t *fifo);
void insert_fifo(fifo_t *fifo, matrix_t *mat);
matrix_t *retrieve_fifo(fifo_t *fifo);