#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "matrix_generator/matrix.h"


/*
 * Sequential Sparse Matrix-Vector Multiplication
 */

void spmv(
    const SpmvDataset *matrix,
    const double *x,
    double *y)
{
    for (int row = 0; row < matrix->rows; row++)
    {
        y[row] = 0.0;

        for (int index = matrix->row_ptr[row];
             index < matrix->row_ptr[row + 1];
             index++)
        {
            y[row] +=
                matrix->values[index] *
                x[matrix->col_idx[index]];
        }
    }
}


int main()
{
    SpmvDataset matrix = {0};

    /*
     * Load the shared matrix dataset
     */

    printf("Loading shared matrix dataset...\n");

    if (!load_matrix(
            "matrix_generator/data/uniform_matrix.bin",
            &matrix))
    {
        printf("Failed to load matrix dataset!\n");
        return 1;
    }

    int rows = matrix.rows;
    int cols = matrix.cols;

    double *x = matrix.x;

    double *y = malloc((size_t)rows * sizeof(double));

    if (y == NULL)
    {
        printf("Memory allocation failed!\n");
        free_dataset(&matrix);
        return 1;
    }


    printf("Matrix size: %d x %d\n", rows, cols);

    printf("Non-zero elements: %d\n", matrix.nnz);


    /*
     * Analyze row structure
     */

    int min_nnz =
        matrix.row_ptr[1] - matrix.row_ptr[0];

    int max_nnz = min_nnz;

    long long total_nnz = 0;

    for (int i = 0; i < rows; i++)
    {
        int row_nnz =
            matrix.row_ptr[i + 1] -
            matrix.row_ptr[i];

        if (row_nnz < min_nnz)
            min_nnz = row_nnz;

        if (row_nnz > max_nnz)
            max_nnz = row_nnz;

        total_nnz += row_nnz;
    }


    double average_nnz =
        (double)total_nnz / rows;


    /*
     * Calculate standard deviation
     */

    double variance = 0.0;

    for (int i = 0; i < rows; i++)
    {
        int row_nnz =
            matrix.row_ptr[i + 1] -
            matrix.row_ptr[i];

        double difference =
            row_nnz - average_nnz;

        variance += difference * difference;
    }

    variance /= rows;

    double stddev = sqrt(variance);


    printf("Minimum nonzeros per row: %d\n", min_nnz);

    printf("Maximum nonzeros per row: %d\n", max_nnz);

    printf("Average nonzeros per row: %.2f\n", average_nnz);

    printf("Row NNZ standard deviation: %.2f\n", stddev);


    /*
     * Sequential SpMV
     */

    printf("Performing sequential SpMV...\n");

    clock_t start = clock();

    for (int i = 0; i < 100; i++)
    {
        spmv(&matrix, x, y);
    }

    clock_t end = clock();

    double execution_time =
        ((double)(end - start)) / CLOCKS_PER_SEC;


    printf("Execution time: %.6f seconds\n", execution_time);

    printf("Result check: y[0] = %.4f\n", y[0]);


    /*
     * Free allocated memory
     */

    free(y);
    free_dataset(&matrix);

    return 0;
}
