#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#include "matrix_generator/matrix.h"

/* Keep all CUDA calls checked so an error is reported where it happens. */
static void check_cuda(cudaError_t error, const char *where)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where,
                cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

/* Reference CUDA CSR kernel: one thread computes one complete row.
 * This is the same basic strategy as the original CSR kernel.
 */
__global__ void spmv_csr_one_thread_per_row(
    int rows,
    const double *values,
    const int *col_idx,
    const int *row_ptr,
    const double *x,
    double *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        double sum = 0.0;

        for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++)
            sum += values[i] * x[col_idx[i]];

        y[row] = sum;
    }
}

/* Optimized CSR kernel: one warp (32 threads) works on one row.
 * Each lane sums some of that row's nonzeros, then the warp combines
 * the 32 partial sums with shuffle instructions (no shared-memory array).
 */
__global__ void spmv_csr_one_warp_per_row(
    int rows,
    const double *values,
    const int *col_idx,
    const int *row_ptr,
    const double *x,
    double *y)
{
    int lane = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int row = blockIdx.x * warps_per_block + warp_in_block;

    if (row < rows) {
        double sum = 0.0;

        /* Lane 0 handles entries 0, 32, 64, ...;
         * lane 1 handles entries 1, 33, 65, and so on.
         */
        for (int i = row_ptr[row] + lane; i < row_ptr[row + 1]; i += 32)
            sum += values[i] * x[col_idx[i]];

        /* Reduce the per-lane partial sums inside the warp. */
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        /* Only one lane writes the final answer for this row. */
        if (lane == 0)
            y[row] = sum;
    }
}

/* CPU SpMV is used only as a correctness reference, not for GPU timing. */
static void spmv_cpu_reference(
    const SpmvDataset *matrix,
    double *y)
{
    for (int row = 0; row < matrix->rows; row++) {
        double sum = 0.0;
        for (int i = matrix->row_ptr[row]; i < matrix->row_ptr[row + 1]; i++)
            sum += matrix->values[i] * matrix->x[matrix->col_idx[i]];
        y[row] = sum;
    }
}

/* Run one warm-up launch, then time a batch of kernel launches with CUDA events.
 * Matrix and vector transfers are outside this kernel-only timing.
 */
static float time_one_thread_kernel(
    const SpmvDataset *matrix,
    const double *d_values,
    const int *d_col_idx,
    const int *d_row_ptr,
    const double *d_x,
    double *d_y,
    int block_size,
    int iterations)
{
    int blocks = (matrix->rows + block_size - 1) / block_size;

    spmv_csr_one_thread_per_row<<<blocks, block_size>>>(
        matrix->rows, d_values, d_col_idx, d_row_ptr, d_x, d_y);
    check_cuda(cudaGetLastError(), "one-thread warm-up launch");
    check_cuda(cudaDeviceSynchronize(), "one-thread warm-up sync");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    check_cuda(cudaEventRecord(start), "record start event");
    for (int i = 0; i < iterations; i++) {
        spmv_csr_one_thread_per_row<<<blocks, block_size>>>(
            matrix->rows, d_values, d_col_idx, d_row_ptr, d_x, d_y);
    }
    check_cuda(cudaGetLastError(), "one-thread timed launches");
    check_cuda(cudaEventRecord(stop), "record stop event");
    check_cuda(cudaEventSynchronize(stop), "wait for one-thread timing");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, stop),
               "read one-thread timing");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_ms / iterations;
}

/* Same timing procedure for the warp-per-row kernel. */
static float time_warp_kernel(
    const SpmvDataset *matrix,
    const double *d_values,
    const int *d_col_idx,
    const int *d_row_ptr,
    const double *d_x,
    double *d_y,
    int block_size,
    int iterations)
{
    int warps_per_block = block_size / 32;
    int blocks = (matrix->rows + warps_per_block - 1) / warps_per_block;

    spmv_csr_one_warp_per_row<<<blocks, block_size>>>(
        matrix->rows, d_values, d_col_idx, d_row_ptr, d_x, d_y);
    check_cuda(cudaGetLastError(), "warp warm-up launch");
    check_cuda(cudaDeviceSynchronize(), "warp warm-up sync");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start event");
    check_cuda(cudaEventCreate(&stop), "create stop event");

    check_cuda(cudaEventRecord(start), "record start event");
    for (int i = 0; i < iterations; i++) {
        spmv_csr_one_warp_per_row<<<blocks, block_size>>>(
            matrix->rows, d_values, d_col_idx, d_row_ptr, d_x, d_y);
    }
    check_cuda(cudaGetLastError(), "warp timed launches");
    check_cuda(cudaEventRecord(stop), "record stop event");
    check_cuda(cudaEventSynchronize(stop), "wait for warp timing");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, stop),
               "read warp timing");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_ms / iterations;
}

/* Compare every GPU output row with the CPU reference. */
static int check_result(
    const double *cpu_y,
    const double *gpu_y,
    int rows,
    double tolerance)
{
    int mismatches = 0;
    double max_error = 0.0;

    for (int row = 0; row < rows; row++) {
        double error = fabs(cpu_y[row] - gpu_y[row]);
        if (error > max_error)
            max_error = error;
        if (error > tolerance)
            mismatches++;
    }

    printf("  Correctness: %s (%d rows exceed %.1e)\n",
           mismatches == 0 ? "PASS" : "FAIL", mismatches, tolerance);
    printf("  Maximum absolute error: %.12e\n", max_error);
    return mismatches == 0;
}

/* Save the most recent warp-kernel result for comparison with the team's
 * cuda_output.txt or cpu_output.txt files for this same dataset.
 */
static int save_output(const char *filename, const double *y, int rows)
{
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Unable to create output file");
        return 0;
    }

    for (int row = 0; row < rows; row++)
        fprintf(file, "%.10f\n", y[row]);

    fclose(file);
    return 1;
}

int main(int argc, char **argv)
{
    /* Use the exact shared dataset loaded by the updated CPU/CUDA programs. */
    int irregular = (argc > 1 && strcmp(argv[1], "--irregular") == 0);
    const char *filename = irregular
        ? "matrix_generator/data/irregular_matrix.bin"
        : "matrix_generator/data/uniform_matrix.bin";

    SpmvDataset matrix = {0};
    printf("Loading shared dataset: %s\n", filename);
    if (!load_matrix(filename, &matrix)) {
        fprintf(stderr, "Could not load the dataset. Run from the project root and check the file path.\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp device;
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "get current device");
    check_cuda(cudaGetDeviceProperties(&device, device_id), "get device properties");

    printf("GPU: %s\n", device.name);
    printf("Matrix: %d x %d, NNZ: %d\n",
           matrix.rows, matrix.cols, matrix.nnz);

    /* Allocate device arrays sized to the actual CSR data. */
    double *d_values = NULL;
    int *d_col_idx = NULL;
    int *d_row_ptr = NULL;
    double *d_x = NULL;
    double *d_y = NULL;

    check_cuda(cudaMalloc((void **)&d_values,
                          (size_t)matrix.nnz * sizeof(double)), "allocate values");
    check_cuda(cudaMalloc((void **)&d_col_idx,
                          (size_t)matrix.nnz * sizeof(int)), "allocate column indices");
    check_cuda(cudaMalloc((void **)&d_row_ptr,
                          ((size_t)matrix.rows + 1) * sizeof(int)), "allocate row pointers");
    check_cuda(cudaMalloc((void **)&d_x,
                          (size_t)matrix.cols * sizeof(double)), "allocate x");
    check_cuda(cudaMalloc((void **)&d_y,
                          (size_t)matrix.rows * sizeof(double)), "allocate y");

    /* Copy the shared inputs to the GPU once; these copies are not timed. */
    check_cuda(cudaMemcpy(d_values, matrix.values,
                          (size_t)matrix.nnz * sizeof(double),
                          cudaMemcpyHostToDevice), "copy values to GPU");
    check_cuda(cudaMemcpy(d_col_idx, matrix.col_idx,
                          (size_t)matrix.nnz * sizeof(int),
                          cudaMemcpyHostToDevice), "copy columns to GPU");
    check_cuda(cudaMemcpy(d_row_ptr, matrix.row_ptr,
                          ((size_t)matrix.rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice), "copy row pointers to GPU");
    check_cuda(cudaMemcpy(d_x, matrix.x,
                          (size_t)matrix.cols * sizeof(double),
                          cudaMemcpyHostToDevice), "copy x to GPU");

    double *cpu_y = (double *)malloc((size_t)matrix.rows * sizeof(double));
    double *gpu_y = (double *)malloc((size_t)matrix.rows * sizeof(double));
    if (!cpu_y || !gpu_y) {
        fprintf(stderr, "Host output allocation failed.\n");
        free(cpu_y);
        free(gpu_y);
        cudaFree(d_values);
        cudaFree(d_col_idx);
        cudaFree(d_row_ptr);
        cudaFree(d_x);
        cudaFree(d_y);
        free_dataset(&matrix);
        return EXIT_FAILURE;
    }

    /* Compute the CPU reference once for correctness checking. */
    spmv_cpu_reference(&matrix, cpu_y);

    const int iterations = 100;
    const double tolerance = 1e-8;
    const int baseline_block_size = 256;
    const int optimized_block_sizes[] = {128, 256, 512};
    const int number_of_block_sizes =
        (int)(sizeof(optimized_block_sizes) / sizeof(optimized_block_sizes[0]));

    printf("\nOriginal CSR kernel (one thread per row), block size %d:\n",
           baseline_block_size);
    float baseline_ms = time_one_thread_kernel(
        &matrix, d_values, d_col_idx, d_row_ptr, d_x, d_y,
        baseline_block_size, iterations);
    check_cuda(cudaMemcpy(gpu_y, d_y,
                          (size_t)matrix.rows * sizeof(double),
                          cudaMemcpyDeviceToHost), "copy baseline result to CPU");
    printf("  Average kernel time: %.6f ms per SpMV (%d iterations)\n",
           baseline_ms, iterations);
    int all_correct = check_result(cpu_y, gpu_y, matrix.rows, tolerance);

    printf("\nOptimized CSR kernel (one warp per row):\n");
    float fastest_ms = 0.0f;
    int fastest_block_size = 0;

    for (int i = 0; i < number_of_block_sizes; i++) {
        int block_size = optimized_block_sizes[i];
        if (block_size > device.maxThreadsPerBlock)
            continue;

        float optimized_ms = time_warp_kernel(
            &matrix, d_values, d_col_idx, d_row_ptr, d_x, d_y,
            block_size, iterations);

        check_cuda(cudaMemcpy(gpu_y, d_y,
                              (size_t)matrix.rows * sizeof(double),
                              cudaMemcpyDeviceToHost), "copy optimized result to CPU");

        int correct = check_result(cpu_y, gpu_y, matrix.rows, tolerance);
        all_correct = all_correct && correct;

        printf("  Block size %d (%d warps/block): %.6f ms per SpMV, speedup %.2fx\n",
               block_size, block_size / 32, optimized_ms,
               baseline_ms / optimized_ms);

        if (fastest_block_size == 0 || optimized_ms < fastest_ms) {
            fastest_ms = optimized_ms;
            fastest_block_size = block_size;
        }
    }

    if (fastest_block_size != 0) {
        printf("\nFastest tested warp-kernel block size: %d threads\n",
               fastest_block_size);
        printf("Best measured speedup over original CSR kernel: %.2fx\n",
               baseline_ms / fastest_ms);

        /* Run the fastest configuration once more so the saved file contains
         * the output from that configuration.
         */
        int warps_per_block = fastest_block_size / 32;
        int blocks = (matrix.rows + warps_per_block - 1) / warps_per_block;
        spmv_csr_one_warp_per_row<<<blocks, fastest_block_size>>>(
            matrix.rows, d_values, d_col_idx, d_row_ptr, d_x, d_y);
        check_cuda(cudaGetLastError(), "final optimized launch");
        check_cuda(cudaDeviceSynchronize(), "final optimized sync");
        check_cuda(cudaMemcpy(gpu_y, d_y,
                              (size_t)matrix.rows * sizeof(double),
                              cudaMemcpyDeviceToHost), "copy final result to CPU");
        all_correct = check_result(cpu_y, gpu_y, matrix.rows, tolerance) && all_correct;
        if (!save_output("cuda_optimized_output.txt", gpu_y, matrix.rows))
            all_correct = 0;
        else
            printf("Optimized output saved to cuda_optimized_output.txt\n");
    }

    printf("\nOverall correctness: %s\n", all_correct ? "PASS" : "FAIL");

    free(cpu_y);
    free(gpu_y);
    cudaFree(d_values);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);
    free_dataset(&matrix);

    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
