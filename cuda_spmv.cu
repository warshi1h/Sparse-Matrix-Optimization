#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include "matrix_generator/matrix.h"


typedef struct
{
    int rows;
    int cols;
    int long_rows;
    int max_row_length;

    double *values;
    int *column_index;
    int *row_ids;
} ELLMatrix;


/* ============================================================
   MATRIX ANALYSIS
   ============================================================ */

double analyze_matrix(
    SpmvDataset *matrix,
    int *min_nnz,
    int *max_nnz,
    double *average_nnz,
    double *stddev)
{
    *min_nnz =
        matrix->row_ptr[1] -
        matrix->row_ptr[0];

    *max_nnz = *min_nnz;

    long long total = 0;

    for (int i = 0; i < matrix->rows; i++)
    {
        int row_nnz =
            matrix->row_ptr[i + 1] -
            matrix->row_ptr[i];

        total += row_nnz;

        if (row_nnz < *min_nnz)
            *min_nnz = row_nnz;

        if (row_nnz > *max_nnz)
            *max_nnz = row_nnz;
    }

    *average_nnz =
        (double)total / matrix->rows;

    double variance = 0.0;

    for (int i = 0; i < matrix->rows; i++)
    {
        double row_nnz =
            matrix->row_ptr[i + 1] -
            matrix->row_ptr[i];

        double difference =
            row_nnz - *average_nnz;

        variance +=
            difference * difference;
    }

    variance /= matrix->rows;

    *stddev = sqrt(variance);

    return (*stddev / *average_nnz) * 100.0;
}


/* ============================================================
   CSR CUDA KERNEL
   ============================================================ */

__global__ void spmv_csr_kernel(
    int rows,
    const double *values,
    const int *column_index,
    const int *row_ptr,
    const double *x,
    double *y,
    int threshold)
{
    int row =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (row < rows)
    {
    	int row_nnz =
    	    row_ptr[row + 1] - row_ptr[row];
    	
    	if (row_nnz > threshold)
    	    return;
    	    
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


/* ============================================================
   ELL CUDA KERNEL
   ============================================================ */

__global__ void spmv_ell_kernel(
    int long_rows,
    int max_row_length,
    const double *values,
    const int *column_index,
    const int *row_ids,
    const double *x,
    double *y)
{
    int index =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (index < long_rows)
    {
        double sum = 0.0;

        for (int k = 0;
             k < max_row_length;
             k++)
        {
            int position =
                k * long_rows + index;

            sum +=
                values[position] *
                x[column_index[position]];
        }

        y[row_ids[index]] = sum;
    }
}


/* ============================================================
   BUILD ELL FROM LONG CSR ROWS
   ============================================================ */

ELLMatrix build_ell(
    SpmvDataset *matrix,
    int threshold)
{
    ELLMatrix ell;

    ell.rows = matrix->rows;
    ell.cols = matrix->cols;
    ell.long_rows = 0;
    ell.max_row_length = 0;

    for (int i = 0; i < matrix->rows; i++)
    {
        int row_nnz =
            matrix->row_ptr[i + 1] -
            matrix->row_ptr[i];

        if (row_nnz > threshold)
        {
            ell.long_rows++;

            if (row_nnz > ell.max_row_length)
                ell.max_row_length = row_nnz;
        }
    }

    ell.values =
        (double *)calloc(
            ell.long_rows *
            ell.max_row_length,
            sizeof(double));

    ell.column_index =
        (int *)calloc(
            ell.long_rows *
            ell.max_row_length,
            sizeof(int));

    ell.row_ids =
        (int *)malloc(
            ell.long_rows *
            sizeof(int));

    if (ell.values == NULL ||
        ell.column_index == NULL ||
        ell.row_ids == NULL)
    {
        printf("ELL allocation failed!\n");
        exit(1);
    }

    int long_row_index = 0;

    for (int row = 0; row < matrix->rows; row++)
    {
        int row_nnz =
            matrix->row_ptr[row + 1] -
            matrix->row_ptr[row];

        if (row_nnz > threshold)
        {
            ell.row_ids[long_row_index] = row;

            int start =
                matrix->row_ptr[row];

            for (int k = 0; k < row_nnz; k++)
            {
                ell.values[
                    k * ell.long_rows +
                    long_row_index] =
                    matrix->values[start + k];

                ell.column_index[
                    k * ell.long_rows +
                    long_row_index] =
                    matrix->col_idx[start + k];
            }

            long_row_index++;
        }
    }

    return ell;
}


/* ============================================================
   CUDA ERROR CHECK
   ============================================================ */

void check_cuda(
    cudaError_t error,
    const char *message)
{
    if (error != cudaSuccess)
    {
        printf(
            "CUDA error at %s: %s\n",
            message,
            cudaGetErrorString(error));

        exit(1);
    }
}


/* ============================================================
   FREE MATRICES
   ============================================================ */

void free_ell(ELLMatrix *ell)
{
    free(ell->values);
    free(ell->column_index);
    free(ell->row_ids);
}


/* ============================================================
   MAIN
   ============================================================ */

int main(int argc, char **argv)
{


    int irregular = 0;

    if (argc > 1 &&
        strcmp(argv[1], "--irregular") == 0)
    {
        irregular = 1;
    }


    printf("========================================\n");
    printf("Runtime Adaptive CUDA SpMV\n");
    printf("========================================\n");

    const char *filename = irregular
        ? "matrix_generator/data/irregular_matrix.bin"
        : "matrix_generator/data/uniform_matrix.bin";

    printf("Loading shared dataset: %s\\n", filename);

    SpmvDataset matrix = {0};

    if (!load_matrix(filename, &matrix))
    {
        printf("Failed to load matrix dataset!\\n");
        return 1;
    }

    int rows = matrix.rows;
    int cols = matrix.cols;

    int min_nnz;
    int max_nnz;

    double average_nnz;
    double stddev;

    double cv =
        analyze_matrix(
            &matrix,
            &min_nnz,
            &max_nnz,
            &average_nnz,
            &stddev);


    printf("Matrix size: %d x %d\n",
           rows,
           cols);

    printf("Non-zero elements: %d\n",
           matrix.nnz);

    printf("Minimum NNZ/row: %d\n",
           min_nnz);

    printf("Maximum NNZ/row: %d\n",
           max_nnz);

    printf("Average NNZ/row: %.2f\n",
           average_nnz);

    printf("Row NNZ standard deviation: %.2f\n",
           stddev);

    printf("Coefficient of variation: %.2f%%\n",
           cv);


    /*
     * Runtime layout selection.
     */

    int use_hybrid =
        (cv >= 10.0);


    printf("\nRuntime layout decision:\n");

    if (use_hybrid)
    {
        printf("Selected layout: HYBRID\n");
        printf("Reason: high row-length variation.\n");
    }
    else
    {
        printf("Selected layout: CSR\n");
        printf("Reason: row lengths are relatively uniform.\n");
    }


    /*
     * Build HYBRID representation if required.
     *
     * Rows above the average are stored in ELL.
     * Shorter rows remain in CSR.
     */

    int threshold =
        (int)ceil(average_nnz);

    if (!use_hybrid)
        threshold = rows;

    ELLMatrix ell;

    ell.values = NULL;
    ell.column_index = NULL;
    ell.row_ids = NULL;
    ell.long_rows = 0;
    ell.max_row_length = 0;


    if (use_hybrid)
    {
        ell =
            build_ell(
                &matrix,
                threshold);

        printf(
            "HYBRID threshold: %d NNZ/row\n",
            threshold);

        printf(
            "Rows using ELL: %d\n",
            ell.long_rows);

        printf(
            "Maximum ELL row length: %d\n",
            ell.max_row_length);
    }


    /*
     * Shared input vector and output vector.
     */

    double *x = matrix.x;

    double *y =
        (double *)malloc(
            rows * sizeof(double));

    if (y == NULL)
    {
        printf("Output vector allocation failed!\\n");
        free_dataset(&matrix);
        return 1;
    }


    /*
     * GPU CSR memory.
     */

    double *d_values;
    int *d_column_index;
    int *d_row_ptr;
    double *d_x;
    double *d_y;


    check_cuda(
        cudaMalloc(
            &d_values,
            matrix.nnz *
            sizeof(double)),
        "cudaMalloc d_values");

    check_cuda(
        cudaMalloc(
            &d_column_index,
            matrix.nnz *
            sizeof(int)),
        "cudaMalloc d_column_index");

    check_cuda(
        cudaMalloc(
            &d_row_ptr,
            (rows + 1) *
            sizeof(int)),
        "cudaMalloc d_row_ptr");

    check_cuda(
        cudaMalloc(
            &d_x,
            cols *
            sizeof(double)),
        "cudaMalloc d_x");

    check_cuda(
        cudaMalloc(
            &d_y,
            rows *
            sizeof(double)),
        "cudaMalloc d_y");


    /*
     * Copy CSR data to GPU.
     */

    check_cuda(
        cudaMemcpy(
            d_values,
            matrix.values,
            matrix.nnz *
            sizeof(double),
            cudaMemcpyHostToDevice),
        "copy values");

    check_cuda(
        cudaMemcpy(
            d_column_index,
            matrix.col_idx,
            matrix.nnz *
            sizeof(int),
            cudaMemcpyHostToDevice),
        "copy column indices");

    check_cuda(
        cudaMemcpy(
            d_row_ptr,
            matrix.row_ptr,
            (rows + 1) *
            sizeof(int),
            cudaMemcpyHostToDevice),
        "copy row pointers");

    check_cuda(
        cudaMemcpy(
            d_x,
            x,
            cols *
            sizeof(double),
            cudaMemcpyHostToDevice),
        "copy x");


    /*
     * GPU ELL memory.
     */

    double *d_ell_values = NULL;
    int *d_ell_column_index = NULL;
    int *d_ell_row_ids = NULL;


    if (use_hybrid)
    {
        check_cuda(
            cudaMalloc(
                &d_ell_values,
                ell.long_rows *
                ell.max_row_length *
                sizeof(double)),
            "cudaMalloc ELL values");

        check_cuda(
            cudaMalloc(
                &d_ell_column_index,
                ell.long_rows *
                ell.max_row_length *
                sizeof(int)),
            "cudaMalloc ELL columns");

        check_cuda(
            cudaMalloc(
                &d_ell_row_ids,
                ell.long_rows *
                sizeof(int)),
            "cudaMalloc ELL row IDs");


        check_cuda(
            cudaMemcpy(
                d_ell_values,
                ell.values,
                ell.long_rows *
                ell.max_row_length *
                sizeof(double),
                cudaMemcpyHostToDevice),
            "copy ELL values");

        check_cuda(
            cudaMemcpy(
                d_ell_column_index,
                ell.column_index,
                ell.long_rows *
                ell.max_row_length *
                sizeof(int),
                cudaMemcpyHostToDevice),
            "copy ELL columns");

        check_cuda(
            cudaMemcpy(
                d_ell_row_ids,
                ell.row_ids,
                ell.long_rows *
                sizeof(int),
                cudaMemcpyHostToDevice),
            "copy ELL row IDs");
    }


    int threads_per_block = 256;

    int blocks =
        (rows +
         threads_per_block - 1)
        / threads_per_block;


    int ell_blocks =
        (ell.long_rows +
         threads_per_block - 1)
        / threads_per_block;


    /*
     * Warm-up.
     */

    if (!use_hybrid)
    {
        spmv_csr_kernel<<<
            blocks,
            threads_per_block>>>(
                rows,
                d_values,
                d_column_index,
                d_row_ptr,
                d_x,
                d_y,
                threshold);
    }
    else
    {
        /*
         * HYBRID execution:
         *
         * For this prototype the selected ELL
         * rows are computed with ELL.
         *
         * Short rows remain in CSR and are
         * computed with the CSR kernel.
         */

        spmv_csr_kernel<<<
            blocks,
            threads_per_block>>>(
                rows,
                d_values,
                d_column_index,
                d_row_ptr,
                d_x,
                d_y,
                threshold);

        spmv_ell_kernel<<<
            ell_blocks,
            threads_per_block>>>(
                ell.long_rows,
                ell.max_row_length,
                d_ell_values,
                d_ell_column_index,
                d_ell_row_ids,
                d_x,
                d_y);
    }


    check_cuda(
        cudaDeviceSynchronize(),
        "warm-up");


    /*
     * Timing.
     */

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    int iterations = 100;

    cudaEventRecord(start);


    for (int i = 0;
         i < iterations;
         i++)
    {
        if (!use_hybrid)
        {
            spmv_csr_kernel<<<
                blocks,
                threads_per_block>>>(
                    rows,
                    d_values,
                    d_column_index,
                    d_row_ptr,
                    d_x,
                    d_y,
                    threshold);
        }
        else
        {
            spmv_csr_kernel<<<
                blocks,
                threads_per_block>>>(
                    rows,
                    d_values,
                    d_column_index,
                    d_row_ptr,
                    d_x,
                    d_y,
                    threshold);

            spmv_ell_kernel<<<
                ell_blocks,
                threads_per_block>>>(
                    ell.long_rows,
                    ell.max_row_length,
                    d_ell_values,
                    d_ell_column_index,
                    d_ell_row_ids,
                    d_x,
                    d_y);
        }
    }


    cudaEventRecord(stop);

    cudaEventSynchronize(stop);


    float total_time_ms;

    cudaEventElapsedTime(
        &total_time_ms,
        start,
        stop);


    printf(
        "\nCUDA selected-layout time for %d iterations: %.6f ms\n",
        iterations,
        total_time_ms);

    printf(
        "Average selected-layout time: %.6f ms\n",
        total_time_ms / iterations);


    check_cuda(
        cudaGetLastError(),
        "kernel execution");


    /*
     * Copy result back.
     */

    check_cuda(
        cudaMemcpy(
            y,
            d_y,
            rows * sizeof(double),
            cudaMemcpyDeviceToHost),
        "copy result");


    printf(
        "Result check: y[0] = %.4f\n",
        y[0]);

    FILE *output = fopen("cuda_output.txt", "w");
    
    if (output == NULL)
    {
        printf("Failed to create CUDA output file!\n");
        return 1;
    }
    
    for (int i = 0; i < rows; i++)
    {
        fprintf(output, "%.10f\n", y[i]);
    }
    
    fclose(output);
    
    printf("Full CUDA output saved to cuda_output.txt\n");

    printf(
        "Runtime adaptive CUDA SpMV completed successfully.\n");


    /*
     * Cleanup.
     */

    cudaFree(d_values);
    cudaFree(d_column_index);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);

    cudaFree(d_ell_values);
    cudaFree(d_ell_column_index);
    cudaFree(d_ell_row_ids);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(y);

    free_dataset(&matrix);

    if (use_hybrid)
        free_ell(&ell);


    return 0;
}
