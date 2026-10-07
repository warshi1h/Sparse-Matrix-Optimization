#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


typedef struct {
    int rows;
    int cols;

    double *values;
    int *column_index;
    int *row_ptr;

    int nonzeros;
} CSRMatrix;


/* Generate the same sparse matrix as the CPU version */
CSRMatrix generate_sparse_matrix(
    int rows,
    int cols,
    double sparsity)
{
    CSRMatrix matrix;

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nonzeros = 0;

    matrix.values =
        (double *)malloc(rows * cols * sizeof(double));

    matrix.column_index =
        (int *)malloc(rows * cols * sizeof(int));

    matrix.row_ptr =
        (int *)malloc((rows + 1) * sizeof(int));

    if (matrix.values == NULL ||
        matrix.column_index == NULL ||
        matrix.row_ptr == NULL)
    {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    srand(42);

    matrix.row_ptr[0] = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double probability =
                (double)rand() / RAND_MAX;

            if (probability > sparsity)
            {
                matrix.values[matrix.nonzeros] =
                    1.0 +
                    ((double)rand() / RAND_MAX) * 9.0;

                matrix.column_index[matrix.nonzeros] = j;

                matrix.nonzeros++;
            }
        }

        matrix.row_ptr[i + 1] =
            matrix.nonzeros;
    }

    return matrix;
}


/* CUDA SpMV kernel */
__global__ void spmv_kernel(
    int rows,
    const double *values,
    const int *column_index,
    const int *row_ptr,
    const double *x,
    double *y)
{
    int row =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (row < rows)
    {
        double sum = 0.0;

        for (int index = row_ptr[row];
             index < row_ptr[row + 1];
             index++)
        {
            sum +=
                values[index] *
                x[column_index[index]];
        }

        y[row] = sum;
    }
}


void free_matrix(CSRMatrix *matrix)
{
    free(matrix->values);
    free(matrix->column_index);
    free(matrix->row_ptr);
}


int main()
{
    int rows = 4000;
    int cols = 4000;

    double sparsity = 0.90;


    printf("Generating sparse matrix...\n");

    CSRMatrix matrix =
        generate_sparse_matrix(
            rows,
            cols,
            sparsity);


    double *x =
        (double *)malloc(
            cols * sizeof(double));

    double *y =
        (double *)malloc(
            rows * sizeof(double));


    if (x == NULL || y == NULL)
    {
        printf("Memory allocation failed!\n");
        return 1;
    }


    /* Initialize input vector */

    for (int i = 0; i < cols; i++)
    {
        x[i] = 1.0;
    }


    printf("Matrix size: %d x %d\n",
           rows,
           cols);

    printf("Non-zero elements: %d\n",
           matrix.nonzeros);


    /* GPU memory */

    double *d_values;
    int *d_column_index;
    int *d_row_ptr;
    double *d_x;
    double *d_y;


    cudaMalloc(
        &d_values,
        matrix.nonzeros *
        sizeof(double));

    cudaMalloc(
        &d_column_index,
        matrix.nonzeros *
        sizeof(int));

    cudaMalloc(
        &d_row_ptr,
        (rows + 1) *
        sizeof(int));

    cudaMalloc(
        &d_x,
        cols *
        sizeof(double));

    cudaMalloc(
        &d_y,
        rows *
        sizeof(double));


    /* CPU → GPU */

    cudaMemcpy(
        d_values,
        matrix.values,
        matrix.nonzeros *
        sizeof(double),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_column_index,
        matrix.column_index,
        matrix.nonzeros *
        sizeof(int),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_row_ptr,
        matrix.row_ptr,
        (rows + 1) *
        sizeof(int),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_x,
        x,
        cols *
        sizeof(double),
        cudaMemcpyHostToDevice);


    /* Launch CUDA kernel */

    /* CUDA kernel timing */
    
    int threads_per_block = 256;
    
    int blocks =
        (rows + threads_per_block - 1)
        / threads_per_block;
    
    
    /* Warm-up run */
    
    spmv_kernel<<<blocks, threads_per_block>>>(
        rows,
        d_values,
        d_column_index,
        d_row_ptr,
        d_x,
        d_y
    );
    
    cudaDeviceSynchronize();
    
    
    /* CUDA events for timing */
    
    cudaEvent_t start;
    cudaEvent_t stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    
    int iterations = 100;
    
    cudaEventRecord(start);
    
    
    /* Run SpMV 100 times */
    
    for (int i = 0; i < iterations; i++)
    {
        spmv_kernel<<<blocks, threads_per_block>>>(
            rows,
            d_values,
            d_column_index,
            d_row_ptr,
            d_x,
            d_y
        );
    }
    
    
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    
    float total_time_ms;
    
    cudaEventElapsedTime(
        &total_time_ms,
        start,
        stop
    );
    
    
    printf(
        "CUDA kernel time for %d iterations: %.6f ms\n",
        iterations,
        total_time_ms
    );
    
    printf(
        "Average CUDA kernel time: %.6f ms\n",
        total_time_ms / iterations
    );
    
    
    /* Check for kernel errors */
    
    cudaError_t error =
        cudaGetLastError();
    
    if (error != cudaSuccess)
    {
        printf(
            "CUDA kernel error: %s\n",
            cudaGetErrorString(error)
        );
    
        return 1;
    }
    
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    /* GPU → CPU */

    cudaMemcpy(
        y,
        d_y,
        rows *
        sizeof(double),
        cudaMemcpyDeviceToHost);


    printf(
        "Result check: y[0] = %.4f\n",
        y[0]);


    printf(
        "CUDA SpMV completed successfully.\n");


    /* Free GPU memory */

    cudaFree(d_values);
    cudaFree(d_column_index);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);

    free(x);
    free(y);

    free_matrix(&matrix);


    return 0;
}
