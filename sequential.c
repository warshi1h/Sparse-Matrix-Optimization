#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int rows;
    int cols;

    double *values;
    int *column_index;
    int *row_ptr;

    int nonzeros;
} CSRMatrix;


/* Generate a random sparse matrix in CSR format */
CSRMatrix generate_sparse_matrix(int rows, int cols, double sparsity)
{
    CSRMatrix matrix;

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nonzeros = 0;

    /* Allocate maximum possible space */
    matrix.values = malloc(rows * cols * sizeof(double));
    matrix.column_index = malloc(rows * cols * sizeof(int));
    matrix.row_ptr = malloc((rows + 1) * sizeof(int));

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
                    1.0 + ((double)rand() / RAND_MAX) * 9.0;

                matrix.column_index[matrix.nonzeros] = j;

                matrix.nonzeros++;
            }
        }

        matrix.row_ptr[i + 1] = matrix.nonzeros;
    }

    return matrix;
}


/* Sequential Sparse Matrix-Vector Multiplication */
void spmv(const CSRMatrix *matrix,
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
                x[matrix->column_index[index]];
        }
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

    /* 90% of the matrix elements are zeros */
    double sparsity = 0.90;

    printf("Generating sparse matrix...\n");

    CSRMatrix matrix =
        generate_sparse_matrix(rows, cols, sparsity);

    double *x = malloc(cols * sizeof(double));
    double *y = malloc(rows * sizeof(double));

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
           rows, cols);

    printf("Non-zero elements: %d\n",
           matrix.nonzeros);

    printf("Performing sequential SpMV...\n");

    clock_t start = clock();

    for (int i = 0; i < 100; i++)
    {
        spmv(&matrix, x, y);
    }

    clock_t end = clock();

    double execution_time =
        ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Execution time: %.6f seconds\n",
           execution_time);

    /* Result check */
    printf("Result check: y[0] = %.4f\n",
           y[0]);

    free(x);
    free(y);
    free_matrix(&matrix);

    return 0;
}
