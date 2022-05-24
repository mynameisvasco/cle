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