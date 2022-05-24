#include <mpi.h>

void swap_columns(double *matrix, int *x, int *y, int size);

void transformation(double *kj, double ki, double ii, double ij);

MPI_Datatype create_matrix_type();

MPI_Datatype create_result_type();
