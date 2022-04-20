#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include "fifo.h"

void init_fifo(fifo_t *fifo, int size)
{
  fifo->inp = fifo->out = fifo->cnt = 0;
  fifo->array = (matrix_t *)(malloc(sizeof(matrix_t) * size));
  fifo->max = size;
  pthread_mutex_init(&fifo->mutex, NULL);
  pthread_cond_init(&fifo->isNotEmpty, NULL);
  pthread_cond_init(&fifo->isNotFull, NULL);
}

int empty_fifo(fifo_t *fifo)
{
  return fifo->cnt == 0;
}

int full_fifo(fifo_t *fifo)
{
  return fifo->cnt == fifo->max;
}

void insert_fifo(fifo_t *fifo, matrix_t mat)
{
  pthread_mutex_lock(&fifo->mutex);

  while (full_fifo(fifo))
  {
    pthread_cond_wait(&fifo->isNotFull, &fifo->mutex);
  }

  unsigned int idx = fifo->inp;
  unsigned int prev = (idx + fifo->max - 1) % fifo->max;
  fifo->array[idx] = mat;
  fifo->inp = (fifo->inp + 1) % fifo->max;
  fifo->cnt++;
  pthread_cond_broadcast(&fifo->isNotEmpty);
  pthread_mutex_unlock(&fifo->mutex);
}

matrix_t retrieve_fifo(fifo_t *fifo)
{
  pthread_mutex_lock(&fifo->mutex);

  while (empty_fifo(fifo))
  {
    pthread_cond_wait(&fifo->isNotEmpty, &fifo->mutex);
  }

  matrix_t result = fifo->array[fifo->out];
  fifo->array[fifo->out] = *(matrix_t *)NULL;
  fifo->out = (fifo->out + 1) % fifo->max;
  fifo->cnt--;
  pthread_cond_broadcast(&fifo->isNotFull);
  pthread_mutex_unlock(&fifo->mutex);
  return result;
}
