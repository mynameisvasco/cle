#include "../common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "matrix.h"

// Process all the matrices on a file
__global__ process_file(double *matrix, double *results, int count, int order)
{
    // Identifier of the matrix
    int matrix_id = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

    // Identifier of the row to be processed
    int row_id = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

    // Start of the matrix
    double *mat = matrix + (matrix_id * order * order);

    // Start of the row
    double *row = mat + (row_id * order);

    double *ii = row + row_id;
    unsigned int signal_reversion = 0;

    if (*ii == 0.0)
    {
        for (int *i = ii + 1; i < ii + order; i++)
        {
            if (*i != 0)
            {
                swap_columns(mat, *ii, *i, order);
                signal_reversion = !signal_reversion;
                break;
            }
        }
    }

    for (int j = order - 1; j > row_id - 1; j--)
    {
        for (int k = row_id + 1; k < order; k++)
        {
            transformation(&mat[order * k + j], mat[order * k + row_id], *ii, mat[order * row_id + j]);
        }
    }

    if (*ii == 0)
    {
        results[matrix_id] = 0;
    }
    else
    {
        results[matrix_id] *= (*ii);
    }
}