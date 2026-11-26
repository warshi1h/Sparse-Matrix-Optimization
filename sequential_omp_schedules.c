#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include "matrix_generator/matrix.h"

/* Sequential reference: visit rows one at a time. */
static void spmv_sequential(const SpmvDataset *matrix,
                            const double *x,
                            double *y)
{
    for (int row = 0; row < matrix->rows; row++) {
        double sum = 0.0;
        for (int i = matrix->row_ptr[row]; i < matrix->row_ptr[row + 1]; i++)
            sum += matrix->values[i] * x[matrix->col_idx[i]];
        y[row] = sum;
    }
}

/* OpenMP with static scheduling: each thread gets a fixed row range. */
static void spmv_static(const SpmvDataset *matrix, const double *x, double *y)
{
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < matrix->rows; row++) {

        double sum = 0.0;

        for (int i = matrix->row_ptr[row]; i < matrix->row_ptr[row + 1]; i++)
            sum += matrix->values[i] * x[matrix->col_idx[i]];
        y[row] = sum;
    }
}

/* Dynamic scheduling gives threads another small group of rows when they
 * finish their current group. Chunk size 8 reduces scheduling overhead.
 */

static void spmv_dynamic(const SpmvDataset *matrix, const double *x, double *y)
{
    #pragma omp parallel for schedule(dynamic, 8)
    for (int row = 0; row < matrix->rows; row++) {

        double sum = 0.0;

        for (int i = matrix->row_ptr[row]; i < matrix->row_ptr[row + 1]; i++)
            sum += matrix->values[i] * x[matrix->col_idx[i]];
        y[row] = sum;
    }
}

/* Guided scheduling starts with larger row groups and makes groups smaller
 * as the loop nears completion. Minimum chunk size is 8 rows.
 */
static void spmv_guided(const SpmvDataset *matrix, const double *x, double *y)
{
    #pragma omp parallel for schedule(guided, 8)
    for (int row = 0; row < matrix->rows; row++) {

        double sum = 0.0;

        for (int i = matrix->row_ptr[row]; i < matrix->row_ptr[row + 1]; i++)
            sum += matrix->values[i] * x[matrix->col_idx[i]];
        y[row] = sum;
    }
}

/* Check all rows against the sequential output. */
static int check_results(const double *reference, const double *result, int rows, const char *name)
{
    const double tolerance = 1e-9;
    double max_error = 0.0;
    int mismatches = 0;

    for (int row = 0; row < rows; row++) {

        double error = fabs(reference[row] - result[row]);

        if (error > max_error) max_error = error;
        if (error > tolerance) mismatches++;
    }

    printf("  %s correctness: %s (%d rows above tolerance)\n", name, mismatches == 0 ? "PASS" : "FAIL", mismatches);
    printf("  Maximum absolute error: %.12e\n", max_error);
    return mismatches == 0;
}

/* Time 100 calls of one implementation after a warm-up call. */

static double time_method(
    void (*method)(const SpmvDataset *, const double *, double *),
    const SpmvDataset *matrix,
    const double *x,
    double *y,
    int iterations)
{
    method(matrix, x, y); /* warm-up */

    double start = omp_get_wtime();
    for (int i = 0; i < iterations; i++)
        method(matrix, x, y);
    return omp_get_wtime() - start;
}

int main(int argc, char **argv)
{
    /* Choose the shared uniform or irregular dataset. */
    const char *filename =
        (argc > 1 && strcmp(argv[1], "--irregular") == 0) ? "matrix_generator/data/irregular_matrix.bin" : "matrix_generator/data/uniform_matrix.bin";

    const int iterations = 100;
    SpmvDataset matrix = {0};

    /* Load the matrix and input vector. */
    printf("Loading shared dataset: %s\n", filename);

    if (!load_matrix(filename, &matrix)) {

        fprintf(stderr, "Could not load dataset. Run from the project root.\n");
        return EXIT_FAILURE;
    }

    /* Allocate separate outputs for reference and OpenMP results. */
    int rows = matrix.rows;
    double *y_sequential = (double *)malloc((size_t)rows * sizeof(double));
    double *y_parallel = (double *)malloc((size_t)rows * sizeof(double));
    
    if (!y_sequential || !y_parallel) {
    
        fprintf(stderr, "Output allocation failed.\n");
        free(y_sequential);
        free(y_parallel);
        free_dataset(&matrix);
        return EXIT_FAILURE;
    }

    /* Print benchmark settings. */
    printf("Matrix: %d x %d, NNZ: %d\n", matrix.rows, matrix.cols, matrix.nnz);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    printf("Each timing is one batch of %d SpMV calls.\n", iterations);

    /* Time the sequential reference. */
    double sequential_time = time_method( spmv_sequential, &matrix, matrix.x, y_sequential, iterations);
    printf("\nSequential reference: %.6f seconds\n", sequential_time);

    /* Compare each OpenMP scheduling strategy with the reference. */
    int all_correct = 1;
    const char *names[] = {"static", "dynamic, chunk 8", "guided, chunk 8"};
    void (*methods[])(const SpmvDataset *, const double *, double *) = {
        spmv_static, spmv_dynamic, spmv_guided
    };

    for (int i = 0; i < 3; i++) {
        double parallel_time = time_method(
            methods[i], &matrix, matrix.x, y_parallel, iterations);

        printf("\nSchedule: %s\n", names[i]);
        printf("  Time: %.6f seconds\n", parallel_time);
        printf("  Speedup over sequential: %.2fx\n", sequential_time / parallel_time);

        if (!check_results(y_sequential, y_parallel, rows, names[i]))
            all_correct = 0;
    }

    /* Report final correctness and release memory. */
    printf("Overall correctness: %s\n", all_correct ? "PASS" : "FAIL");
    free(y_sequential);
    free(y_parallel);
    free_dataset(&matrix);
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
