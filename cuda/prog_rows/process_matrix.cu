#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "matrix.h"

__global__ void process_matrix(double *matrix, double *determinant, unsigned int order)
{
    unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned row = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int index = row * order + column;

    bool signal_reversion = false;
    if (column == row)
    {
        if (matrix[index] == 0)
        {
            for (int j = index + 1; j < row * order; j++)
            {
                if (matrix[order * row + j] != 0)
                {
                    swap_columns(matrix, &index, &j, order);
                    signal_reversion = !signal_reversion;
                    break;
                }
            }
        }
        for (int j = order - 1; j > column - 1; j--)
        {
            for (int k = row + 1; k < order; k++)
            {
                transformation(&matrix[order * k + j], matrix[order * k + i], matrix[order * i + i], matrix[order * i + j]);
            }
        }
        if (matrix[index] == 0)
        {
            *determinant = 0.0;
        }
        *determinant *= matrix[index];
    }
}