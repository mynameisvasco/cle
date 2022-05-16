#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include "fifo.h"

void init_fifo(fifo_t *fifo)
{
  fifo->inp = fifo->out = fifo->cnt = 0;
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
  return fifo->cnt == FIFO_SIZE;
}

void insert_fifo(fifo_t *fifo, matrix_t *mat)
{
  if (pthread_mutex_lock(&fifo->mutex))
  {
    perror("Failed to enter monitor mode");
    pthread_exit(NULL);
  }

  while (full_fifo(fifo))
  {
    if (pthread_cond_wait(&fifo->isNotFull, &fifo->mutex))
    {
      perror("Failed to wait is not full condition");
    }
  }

  unsigned int idx = fifo->inp;
  fifo->array[idx] = mat;
  fifo->inp = (fifo->inp + 1) % FIFO_SIZE;
  fifo->cnt++;

  if (pthread_cond_broadcast(&fifo->isNotEmpty))
  {
    perror("Failed to broadcast is not empty condition");
    pthread_exit(NULL);
  }

  if (pthread_mutex_unlock(&fifo->mutex))
  {
    perror("Failed to exit monitor mode");
    pthread_exit(NULL);
  }
}

matrix_t *retrieve_fifo(fifo_t *fifo)
{
  if (pthread_mutex_lock(&fifo->mutex))
  {
    perror("Failed to enter monitor mode");
    pthread_exit(NULL);
  }

  while (empty_fifo(fifo))
  {
    if (pthread_cond_wait(&fifo->isNotEmpty, &fifo->mutex))
    {
      perror("Failed to wait is not empty condition");
      pthread_exit(NULL);
    }
  }

  matrix_t *result = fifo->array[fifo->out];
  fifo->array[fifo->out] = NULL;
  fifo->out = (fifo->out + 1) % FIFO_SIZE;
  fifo->cnt--;

  if (pthread_cond_broadcast(&fifo->isNotFull))
  {
    perror("Failed to broadcast is not full condition");
    pthread_exit(NULL);
  }

  if (pthread_mutex_unlock(&fifo->mutex))
  {
    perror("Failed to exit monitor mode");
    pthread_exit(NULL);
  }

  return result;
}
