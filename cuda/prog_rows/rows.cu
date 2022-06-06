#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../common.h"

/*
 * Main method of the program
 *
 * Arguments - Runtime arguments 
 */
int main(int argc, char** argv){
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int files_n;
    char* files_path[10];
    double** results;
    double** matrix;
    int* specs
    int input = 0;

    results = malloc(sizeof(double**));
    matrix = malloc(sizeof(double**));
    specs = malloc(sizeof(int*))
    

    while (input != -1)
    {
        input = getopt(argc, argv, "f:");
        if (input == 'f')
            files_paths[files_n++] = optarg;
    }

    for(int file_index = 0; i < file_index; i++){
        FILE *file = fopen(files_paths[file_index], "rb");

        if (file == NULL)
        {
            perror("File could not be read");
            exit(1);
        }

        int matrix_count, matrix_order;
        if (fread(&matrix_count, sizeof(int), 1, file) != 1)
        {
            perror("Error reading number of matrices!");
            exit(1);
        }
        if (fread(&matrix_order, sizeof(int), 1, file) != 1)
        {
            perror("Error reading matrices order!");
            exit(1);
        }

        results[file_index] = malloc(sizeof(double*) * matrix_count);
        matrix[file_index] = malloc(sizeof(double*) * matrix_count);
        specs[file_index*2] = matrix_count;
        specs[file_index*2 + 1] = matrix_order;

        for(int matrix_index = 0; matrix_index < matrix_count; matrix_index++){
            int num_read = fread(matrix[file_index][matrix_index], sizeof(double), matrix_order * matrix_order, file);
            if (num_read != (matrix_order * matrix_order))
            {
                perror("Error reading values from matrix!\n");
                exit(2);
            }
        }
        

    }


}