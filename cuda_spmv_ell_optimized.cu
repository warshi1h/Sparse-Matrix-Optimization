// Standalone ELL companion to cuda_spmv_optimized.cu.
// Build in the project root on a Colab T4:
//   gcc -O3 -c matrix_generator/matrix_io.c matrix_generator/matrix_generator.c
//   nvcc -O3 -std=c++17 -arch=sm_75 cuda_spmv_ell_optimized.cu matrix_io.o matrix_generator.o -o ell_spmv
// Run ./ell_spmv for the uniform matrix or ./ell_spmv --irregular.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "matrix_generator/matrix.h"

constexpr int tile_rows = 32;
constexpr int max_workers = 16;  // 512-thread block / 32 rows

__global__ void csr_to_ell_kernel(int count, int width,
                                  const int *row_ptr, const double *values, const int *columns,
                                  double *ell_values, int *ell_columns)
{
    int erow = blockIdx.x * blockDim.x + threadIdx.x;
    if (erow >= count) return;
    int begin = row_ptr[erow];
    int length = row_ptr[erow + 1] - begin;
    for (long long k = threadIdx.y; k < width; k += blockDim.y) {
        size_t pos = (size_t)k * count + erow;
        ell_values[pos] = k < length ? values[begin + k] : 0.0;
        ell_columns[pos] = k < length ? columns[begin + k] : 0;
    }
}

// A block covers 32 rows. Its 4, 8, or 16 warps split the slots of each row.
// Across a warp, neighboring rows read neighboring values at the same slot.
__global__ void spmv_ell_kernel(int count, const int *row_ptr,
                                const double *values, const int *columns,
                                const double *x, double *y)
{
    __shared__ double partial[max_workers][tile_rows];
    int erow = blockIdx.x * tile_rows + threadIdx.x;
    double sum = 0.0;
    if (erow < count) {
        int length = row_ptr[erow + 1] - row_ptr[erow];
        // Stop at the original row length to avoid reading padded entries.
        for (long long k = threadIdx.y; k < length; k += blockDim.y) {
            size_t pos = (size_t)k * count + erow;
            sum += values[pos] * x[columns[pos]];
        }
    }
    partial[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();  // Lanes past the last row also reach this barrier.
    if (erow < count && threadIdx.y == 0) {
        double total = 0.0;
        for (int worker = 0; worker < (int)blockDim.y; ++worker)
            total += partial[worker][threadIdx.x];
        y[erow] = total;
    }
}

static void check_cuda(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

#define CUDA(call) check_cuda((call), #call)

template<typename T>
static T *device_array(size_t count, const T *source = nullptr)
{
    if (!count) return nullptr;
    if (count > SIZE_MAX / sizeof(T)) {
        std::fprintf(stderr, "Requested GPU array exceeds addressable size.\n");
        std::exit(1);
    }
    T *data = nullptr;
    CUDA(cudaMalloc(&data, count * sizeof(T)));
    if (source) CUDA(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice));
    return data;
}

static int blocks_for(int count, int threads)
{
    return count / threads + (count % threads != 0);
}

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


// Each version runs once before its timed batch, using the same device buffers.
template<typename Launch>
static float time_kernel(Launch launch, int iterations)
{
    launch();
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));
    CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    CUDA(cudaGetLastError());
    CUDA(cudaEventRecord(stop));
    CUDA(cudaEventSynchronize(stop));
    float elapsed = 0;
    CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA(cudaEventDestroy(stop));
    CUDA(cudaEventDestroy(start));
    return elapsed / iterations;
}

static bool check_result(const std::vector<double>& reference, const std::vector<double>& actual)
{
    int mismatches = 0;
    double max_error = 0;
    for (size_t row = 0; row < reference.size(); ++row) {
        double error = std::fabs(actual[row] - reference[row]);
        if (!std::isfinite(error) || error > 1e-8) ++mismatches;
        max_error = std::isfinite(error) ? std::max(max_error, error) : INFINITY;
    }
    std::printf("  Correctness: %s (%d rows exceed 1e-8), max absolute error %.12e\n",
                mismatches ? "FAIL" : "PASS", mismatches, max_error);
    return mismatches == 0;
}

int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--irregular") != 0)) {
        std::fprintf(stderr, "Usage: %s [--irregular]\n", argv[0]);
        return 2;
    }
    const char *file = argc == 2 ? "matrix_generator/data/irregular_matrix.bin"
                                 : "matrix_generator/data/uniform_matrix.bin";
    SpmvDataset matrix = {};
    if (!load_matrix(file, &matrix)) {
        std::fprintf(stderr, "Could not load %s. Run from the project root.\n", file);
        return 1;
    }
    int device_id;
    cudaDeviceProp device;
    CUDA(cudaGetDevice(&device_id));
    CUDA(cudaGetDeviceProperties(&device, device_id));
    const int rows = matrix.rows;
    int width = 0;
    for (int row = 0; row < rows; ++row)
        width = std::max(width, matrix.row_ptr[row + 1] - matrix.row_ptr[row]);
    size_t slots = (size_t)rows * width;
    std::printf("GPU: %s\nDataset: %s\nMatrix: %d x %d, NNZ: %d\n",
                device.name, file, rows, matrix.cols, matrix.nnz);
    std::printf("ELL width: %d, slots: %zu, padding: %zu\n", width, slots, slots - matrix.nnz);

    int *row_ptr = device_array((size_t)rows + 1, matrix.row_ptr);
    double *values = device_array(matrix.nnz, matrix.values);
    int *columns = device_array(matrix.nnz, matrix.col_idx);
    double *ell_values = device_array<double>(slots);
    int *ell_columns = device_array<int>(slots);
    double *x = device_array(matrix.cols, matrix.x);
    double *y = device_array<double>(rows);

    int tiles = blocks_for(rows, tile_rows);
    if (slots) {
        dim3 tile(tile_rows, 8);
        csr_to_ell_kernel<<<tiles, tile>>>(rows, width, row_ptr, values, columns, ell_values, ell_columns);
        CUDA(cudaGetLastError());
    }
    CUDA(cudaFree(values));
    CUDA(cudaFree(columns));

    constexpr int iterations = 100;
    std::vector<double> reference(rows), actual(rows), fastest_output;
    spmv_cpu_reference(&matrix, reference.data());
    size_t bytes = (size_t)rows * sizeof(double);
    bool passed = true;
    int fastest_block = 0;
    float fastest_ms = 0;
    for (int block_size : {128, 256, 512}) {
        if (block_size > device.maxThreadsPerBlock) continue;
        dim3 tile(tile_rows, block_size / tile_rows);
        CUDA(cudaMemset(y, 0xff, bytes));  // Unwritten rows appear as NaN.
        float ms = time_kernel([&]() {
            spmv_ell_kernel<<<tiles, tile>>>(rows, row_ptr, ell_values, ell_columns, x, y);
        }, iterations);
        CUDA(cudaMemcpy(actual.data(), y, bytes, cudaMemcpyDeviceToHost));
        std::printf("Optimized ELL, %d threads (%u per row): %.6f ms per SpMV\n",
                    block_size, tile.y, ms);
        bool correct = check_result(reference, actual);
        passed = passed && correct;
        if (correct && (!fastest_block || ms < fastest_ms)) {
            fastest_block = block_size;
            fastest_ms = ms;
            fastest_output = actual;
        }
    }

    if (passed && fastest_block) {
        std::printf("Fastest tested block: %d threads, %.6f ms per SpMV\n",
                    fastest_block, fastest_ms);
        FILE *out = std::fopen("cuda_ell_optimized_output.txt", "w");
        if (out) {
            for (double value : fastest_output) std::fprintf(out, "%.10f\n", value);
            bool written = !std::ferror(out);
            passed = std::fclose(out) == 0 && written;
        } else {
            passed = false;
        }
        if (passed) std::puts("Saved cuda_ell_optimized_output.txt");
        else std::perror("cuda_ell_optimized_output.txt");
    } else {
        passed = false;
    }
    CUDA(cudaFree(y));
    CUDA(cudaFree(x));
    CUDA(cudaFree(ell_columns));
    CUDA(cudaFree(ell_values));
    CUDA(cudaFree(row_ptr));
    free_dataset(&matrix);
    std::printf("Overall correctness: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
