#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ double process_matrix(double* matrix, double* determinant, unsigned int order){
    unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned row = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int index = row * order + column;
}