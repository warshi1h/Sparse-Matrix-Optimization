// Standalone row-partitioned CSR + ELL CUDA benchmark.
// Build in the project root on a Colab T4:
//   gcc -O3 -c matrix_generator/matrix_io.c matrix_generator/matrix_generator.c
//   nvcc -O3 -std=c++17 -arch=sm_75 cuda_spmv_hybrid_optimized.cu matrix_io.o matrix_generator.o -o hybrid_spmv
// Run ./hybrid_spmv [--irregular] [--threshold N] [--concurrent] [--force-hybrid].
// By default, CV >= 10% enables HYBRID. N sets the maximum CSR row length within HYBRID.
#include <cuda_runtime.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "matrix_generator/matrix.h"

constexpr int tile_rows = 32;
constexpr int max_workers = 16;  // 512-thread block / 32 rows

// Long rows are packed into slot-major ELL on the device.
__global__ void csr_to_ell_kernel(int count, int width, const int *row_ids,
                                  const int *row_ptr, const double *values, const int *columns,
                                  double *ell_values, int *ell_columns)
{
    int erow = blockIdx.x * blockDim.x + threadIdx.x;
    if (erow >= count) return;
    int row = row_ids[erow];
    int begin = row_ptr[row];
    int length = row_ptr[row + 1] - begin;
    for (long long k = threadIdx.y; k < width; k += blockDim.y) {
        size_t pos = (size_t)k * count + erow;
        ell_values[pos] = k < length ? values[begin + k] : 0.0;
        ell_columns[pos] = k < length ? columns[begin + k] : 0;
    }
}

__global__ void spmv_ell_kernel(int count, const int *row_ids, const int *row_ptr,
                                const double *values, const int *columns,
                                const double *x, double *y)
{
    __shared__ double partial[max_workers][tile_rows];
    int erow = blockIdx.x * tile_rows + threadIdx.x;
    int row = 0;
    double sum = 0.0;
    if (erow < count) {
        row = row_ids[erow];
        int length = row_ptr[row + 1] - row_ptr[row];
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
        y[row] = total;
    }
}

/* Optimized CSR kernel: one warp (32 threads) works on one row.
 * Each lane sums some of that row's nonzeros, then the warp combines
 * the 32 partial sums with shuffle instructions (no shared-memory array).
 */
__global__ void spmv_csr_one_warp_per_row(
    int count, const int *row_ids,
    const double *values,
    const int *col_idx,
    const int *row_ptr,
    const double *x,
    double *y)
{
    int lane = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int index = blockIdx.x * warps_per_block + warp_in_block;

    if (index < count) {
        int row = row_ids ? row_ids[index] : index;
        double sum = 0.0;

        /* Lane 0 handles entries 0, 32, 64, ...;
         * lane 1 handles entries 1, 33, 65, and so on.
         */
        for (long long i = (long long)row_ptr[row] + lane; i < row_ptr[row + 1]; i += 32)
            sum += values[i] * x[col_idx[i]];

        /* Reduce the per-lane partial sums inside the warp. */
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        /* Only one lane writes the final answer for this row. */
        if (lane == 0)
            y[row] = sum;
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
static float time_kernel(Launch launch, int iterations, cudaStream_t csr_stream = nullptr,
                         cudaStream_t ell_stream = nullptr)
{
    CUDA(cudaDeviceSynchronize());  // Finish setup, including initialization of y.
    launch();
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));
    cudaEvent_t ell_done = nullptr;
    if (ell_stream != csr_stream)
        CUDA(cudaEventCreateWithFlags(&ell_done, cudaEventDisableTiming));
    CUDA(cudaEventRecord(start, csr_stream));
    if (ell_done) CUDA(cudaStreamWaitEvent(ell_stream, start, 0));
    for (int i = 0; i < iterations; ++i) launch();
    CUDA(cudaGetLastError());
    // Each stream processes a fixed, disjoint subset of rows throughout the batch.
    // Join once at the end so the timer includes the slower stream as well.
    if (ell_done) {
        CUDA(cudaEventRecord(ell_done, ell_stream));
        CUDA(cudaStreamWaitEvent(csr_stream, ell_done, 0));
    }
    CUDA(cudaEventRecord(stop, csr_stream));
    CUDA(cudaEventSynchronize(stop));
    float elapsed = 0;
    CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    if (ell_done) CUDA(cudaEventDestroy(ell_done));
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
    bool irregular = false;
    bool concurrent = false;
    bool force_hybrid = false;
    int threshold = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--irregular") == 0) {
            irregular = true;
        } else if (std::strcmp(argv[i], "--concurrent") == 0) {
            concurrent = true;
        } else if (std::strcmp(argv[i], "--force-hybrid") == 0) {
            force_hybrid = true;
        } else if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            const char *arg = argv[++i];
            const char *end = arg + std::strlen(arg);
            auto result = std::from_chars(arg, end, threshold);
            if (result.ec != std::errc{} || result.ptr != end || threshold < 0) {
                std::fprintf(stderr, "--threshold expects a nonnegative integer.\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "Usage: %s [--irregular] [--threshold N] [--concurrent] [--force-hybrid]\n", argv[0]);
            return 2;
        }
    }
    const char *file = irregular ? "matrix_generator/data/irregular_matrix.bin"
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

    // Match the original selector: CV uses the population deviation of row lengths.
    double average_nnz = (double)matrix.nnz / rows;
    double squared_differences = 0.0;
    for (int row = 0; row < rows; ++row) {
        int length = matrix.row_ptr[row + 1] - matrix.row_ptr[row];
        double difference = length - average_nnz;
        squared_differences += difference * difference;
    }
    double cv = average_nnz > 0.0 ? 100.0 * std::sqrt(squared_differences / rows) / average_nnz : 0.0;
    bool split_rows = force_hybrid || cv >= 10.0;

    // Match the project rule: strictly longer than the rounded-up mean uses ELL.
    if (threshold < 0)
        threshold = matrix.nnz / rows + (matrix.nnz % rows != 0);
    std::vector<int> csr_ids, ell_ids;
    int width = 0;
    size_t ell_nnz = 0;
    if (split_rows) {
        for (int row = 0; row < rows; ++row) {
            int length = matrix.row_ptr[row + 1] - matrix.row_ptr[row];
            if (length > threshold) {
                ell_ids.push_back(row);
                width = std::max(width, length);
                ell_nnz += length;
            } else {
                csr_ids.push_back(row);
            }
        }
    }
    int ell_count = (int)ell_ids.size();
    int csr_count = rows - ell_count;
    const char *layout = ell_count ? (csr_count ? "HYBRID" : "ELL") : "CSR";
    size_t slots = (size_t)ell_count * width;
    std::printf("GPU: %s\nDataset: %s\nMatrix: %d x %d, NNZ: %d\n",
                device.name, file, rows, matrix.cols, matrix.nnz);
    std::printf("CV: %.2f%% | Selection: %s | Layout: %s\n",
                cv, force_hybrid ? "forced row split" : "CV >= 10% enables HYBRID", layout);
    std::printf("Threshold: %d | CSR rows: %d | ELL rows: %d | ELL padding slots: %zu\n",
                threshold, csr_count, ell_count, slots - ell_nnz);

    int *csr_map = ell_count ? device_array(csr_ids.size(), csr_ids.data()) : nullptr;
    int *ell_map = device_array(ell_ids.size(), ell_ids.data());
    int *row_ptr = device_array((size_t)rows + 1, matrix.row_ptr);
    double *values = device_array(matrix.nnz, matrix.values);
    int *columns = device_array(matrix.nnz, matrix.col_idx);
    double *ell_values = device_array<double>(slots);
    int *ell_columns = device_array<int>(slots);
    double *x = device_array(matrix.cols, matrix.x);
    double *y = device_array<double>(rows);

    int tiles = blocks_for(ell_count, tile_rows);
    if (slots) {
        dim3 tile(tile_rows, 8);
        csr_to_ell_kernel<<<tiles, tile>>>(ell_count, width, ell_map, row_ptr, values, columns, ell_values, ell_columns);
        CUDA(cudaGetLastError());
    }

    cudaStream_t csr_stream = nullptr, ell_stream = nullptr;
    if (concurrent && csr_count && ell_count) {
        CUDA(cudaStreamCreateWithFlags(&csr_stream, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&ell_stream, cudaStreamNonBlocking));
    }
    std::printf("%s execution: %s | Device concurrent kernel support: %s\n", layout,
                ell_stream ? "separate CSR and ELL streams" : "single stream",
                device.concurrentKernels ? "yes" : "no");

    // Both kernels write to the same y using disjoint original row IDs.
    auto optimized = [&](int block_size) {
        int warps = block_size / 32;
        if (csr_count)
            spmv_csr_one_warp_per_row<<<blocks_for(csr_count, warps), block_size, 0, csr_stream>>>(csr_count, csr_map, values, columns, row_ptr, x, y);
        if (ell_count) {
            dim3 tile(tile_rows, warps);
            spmv_ell_kernel<<<tiles, tile, 0, ell_stream>>>(ell_count, ell_map, row_ptr, ell_values, ell_columns, x, y);
        }
    };

    constexpr int iterations = 100;
    std::vector<double> reference(rows), actual(rows), fastest_output;
    spmv_cpu_reference(&matrix, reference.data());
    size_t bytes = (size_t)rows * sizeof(double);
    bool passed = true;
    int fastest_block = 0;
    float fastest_ms = 0;
    for (int block_size : {128, 256, 512}) {
        if (block_size > device.maxThreadsPerBlock) continue;
        CUDA(cudaMemset(y, 0xff, bytes));  // Unwritten rows appear as NaN.
        float ms = time_kernel([&]() {
            optimized(block_size);
        }, iterations, csr_stream, ell_stream);
        CUDA(cudaMemcpy(actual.data(), y, bytes, cudaMemcpyDeviceToHost));
        std::printf("Optimized %s, %d threads: %.6f ms per SpMV\n",
                    layout, block_size, ms);
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
        FILE *out = std::fopen("cuda_hybrid_optimized_output.txt", "w");
        if (out) {
            for (double value : fastest_output) std::fprintf(out, "%.10f\n", value);
            bool written = !std::ferror(out);
            passed = std::fclose(out) == 0 && written;
        } else {
            passed = false;
        }
        if (passed) std::puts("Saved cuda_hybrid_optimized_output.txt");
        else std::perror("cuda_hybrid_optimized_output.txt");
    } else {
        passed = false;
    }
    if (ell_stream) CUDA(cudaStreamDestroy(ell_stream));
    if (csr_stream) CUDA(cudaStreamDestroy(csr_stream));
    CUDA(cudaFree(y));
    CUDA(cudaFree(x));
    CUDA(cudaFree(ell_columns));
    CUDA(cudaFree(ell_values));
    CUDA(cudaFree(row_ptr));
    CUDA(cudaFree(columns));
    CUDA(cudaFree(values));
    CUDA(cudaFree(csr_map));
    CUDA(cudaFree(ell_map));
    free_dataset(&matrix);
    std::printf("Overall correctness: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
