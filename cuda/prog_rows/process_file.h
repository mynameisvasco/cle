#include <cuda_runtime.h>

// Process all the matrices on a file
__global__ process_file(double *matrix, double *results, int count, int order);