#include "matrix.h"
#include "fifo.h"

MPI_Datatype create_matrix_type()
{
  MPI_Datatype matrix_type;
  int lengths[4] = {1, 1, 256 * 256, 1};
  MPI_Aint displacements[4];
  struct matrix_t dummy;
  MPI_Aint base_address;
  MPI_Get_address(&dummy, &base_address);
  MPI_Get_address(&dummy.file_id, &displacements[0]);
  MPI_Get_address(&dummy.matrix_id, &displacements[1]);
  MPI_Get_address(&dummy.values, &displacements[2]);
  MPI_Get_address(&dummy.order, &displacements[3]);
  displacements[0] = MPI_Aint_diff(displacements[0], base_address);
  displacements[1] = MPI_Aint_diff(displacements[1], base_address);
  displacements[2] = MPI_Aint_diff(displacements[2], base_address);
  displacements[3] = MPI_Aint_diff(displacements[3], base_address);
  MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_INT};
  MPI_Type_create_struct(4, lengths, displacements, types, &matrix_type);
  MPI_Type_commit(&matrix_type);
  return matrix_type;
}

MPI_Datatype create_result_type()
{
  MPI_Datatype file_result_type;
  int lengths[3] = {1, 1, 1};
  MPI_Aint displacements[3];
  struct file_result_t dummy;
  MPI_Aint base_address;
  MPI_Get_address(&dummy, &base_address);
  MPI_Get_address(&dummy.file_id, &displacements[0]);
  MPI_Get_address(&dummy.matrix_id, &displacements[1]);
  MPI_Get_address(&dummy.determinant, &displacements[2]);
  displacements[0] = MPI_Aint_diff(displacements[0], base_address);
  displacements[1] = MPI_Aint_diff(displacements[1], base_address);
  displacements[2] = MPI_Aint_diff(displacements[2], base_address);

  MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};
  MPI_Type_create_struct(3, lengths, displacements, types, &file_result_type);
  MPI_Type_commit(&file_result_type);
  return file_result_type;
}

// Swaps the values of columns X and Y
void swap_columns(double *matrix, int *x, int *y, int size)
{
  for (int i = 0; i < size; i++)
  {
    double tmp = matrix[size * i + (*x)];
    matrix[size * i + (*x)] = matrix[size * i + (*y)];
    matrix[size * i + (*y)] = tmp;
  }
}

// Applies Gaussian Elimination
void transformation(double *kj, double ki, double ii, double ij)
{
  *kj = *kj - ((ki / ii) * ij);
}