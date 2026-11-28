# Sparse Matrix-Vector Multiplication (SpMV) Optimization

A High Performance Computing project exploring CPU and GPU approaches for optimizing Sparse Matrix-Vector Multiplication (SpMV) using different sparse matrix storage formats.

## 1. Project Overview

Sparse Matrix-Vector Multiplication (SpMV) is a fundamental operation in scientific computing, numerical simulations, and large-scale data processing.

It involves multiplying a sparse matrix \(A\) with an input vector \(x\) to produce an output vector \(y\):

\[
y = Ax
\]

A sparse matrix contains mostly zero-valued elements. Storing and processing these zeros wastes memory and computational resources. This project explores efficient ways to represent sparse matrices and accelerate SpMV using CPU parallelism and GPU computing.

## 2. Project Objectives

- Implement a sequential CPU baseline for SpMV.
- Explore parallel execution using OpenMP and CUDA.
- Compare sparse matrix storage formats: CSR, ELL and Hybrid CSR + ELL.
- Evaluate performance using uniform and irregular sparse matrices.
- Measure execution time and investigate workload distribution.
- Verify computational correctness by comparing results against the sequential implementation.

## 3. Sparse Matrix Storage Formats

### Compressed Sparse Row (CSR)

CSR stores a sparse matrix using three arrays:

- **Values:** Stores all non-zero matrix elements.
- **Column Indices:** Stores the column position of each non-zero element.
- **Row Pointers:** Stores the starting and ending positions of each row in the values and column-index arrays.

CSR avoids storing zeros and provides compact storage for sparse matrices.

### ELLPACK (ELL)

ELL stores matrix rows using a fixed number of slots determined by the maximum row length.

Shorter rows are padded with zeros to maintain a regular storage structure.

This layout can provide predictable memory access patterns, but may introduce additional storage and computation due to padding.

### Hybrid CSR + ELL

The Hybrid approach combines CSR and ELL:

- Shorter rows are stored and processed using CSR.
- Selected longer rows are extracted and stored using ELL.

The implementation uses the coefficient of variation (CV) of row non-zero counts to decide whether to use CSR or Hybrid.

\[
CV = \frac{\text{Standard Deviation}}{\text{Average NNZ per Row}}\times100
\]

The runtime selection rule is:

| Condition | Selected Layout |
|---|---|
| CV < 10% | CSR |
| CV ≥ 10% | Hybrid CSR + ELL |

For Hybrid execution, the default row-selection threshold is the ceiling of the average number of non-zero elements per row.

Rows exceeding this threshold are assigned to ELL, while the remaining rows stay in CSR.

## 4. Matrix Generation

The project uses two types of sparse matrix datasets.

### Uniform Matrix

- Generated using a configurable density.
- Non-zero elements are distributed probabilistically across matrix positions.
- Uses two passes to count non-zero elements and allocate exact CSR storage.

### Irregular Matrix

- Generates rows with different numbers of non-zero elements.
- Uses four row-length ranges:
  - 20–100
  - 150–300
  - 400–600
  - 800–1200
- Uses partial Fisher-Yates shuffling to select unique column indices within each row.

Both datasets are stored in CSR format and include a common input vector.

## 5. Implementation

### Sequential CPU Implementation

File: `sequential.c`

The sequential implementation serves as the baseline.

It processes matrix rows one at a time. For each row, it traverses its non-zero elements, multiplies them by the corresponding input-vector elements, and accumulates the result.

The implementation executes SpMV repeatedly and measures CPU execution time using `clock()`.

### OpenMP Implementation

File: `sequential_omp_schedules.c`

The OpenMP implementation explores CPU parallelism and different scheduling approaches for distributing row-processing work across CPU threads.

### CUDA Implementation

The project includes CUDA implementations for GPU-based SpMV.

#### Runtime-Adaptive CUDA Implementation

File: `cuda_spmv.cu`

This implementation analyzes the matrix at runtime and selects CSR or Hybrid execution based on the coefficient of variation.

**CSR CUDA Kernel**
- One CUDA thread processes one original matrix row.
- Uses CSR row pointers to locate non-zero elements.
- Performs multiplication and accumulation for the assigned row.
- Skips rows assigned to ELL during Hybrid execution.

**ELL CUDA Kernel**
- One CUDA thread processes one selected ELL row.
- Traverses the ELL slots and accumulates the result.
- Uses stored original row IDs to write results into the correct output positions.

**Kernel Configuration**
- Uses 256 threads per block.
- Calculates the number of blocks based on the number of rows being processed.
- Uses separate block calculations for CSR and ELL execution.

**Memory Management**
- Allocates GPU memory using `cudaMalloc()`.
- Transfers input data using `cudaMemcpyHostToDevice`.
- Copies the output vector back using `cudaMemcpyDeviceToHost`.

**Timing**
- Uses CUDA events to measure repeated kernel execution.
- Performs a warm-up execution before timing.
- Executes the selected kernel configuration for 100 iterations.

The runtime-adaptive CUDA timing excludes the initial host-to-device transfers and final device-to-host output transfer.

#### Additional Optimized CUDA Implementations

The repository also includes optimized CSR, ELL and Hybrid CUDA implementations developed as part of the team project:

- `cuda_spmv_optimized.cu`
- `cuda_spmv_ell_optimized.cu`
- `cuda_spmv_hybrid_optimized.cu`

These implementations explore additional GPU optimization techniques and benchmarking configurations.

## 6. Project Structure

```text
Sparse-Matrix-Optimization/
│
├── matrix_generator/
│   ├── data/
│   │   ├── uniform_matrix.bin
│   │   └── irregular_matrix.bin
│   │
│   ├── matrix.h
│   ├── matrix_generator.c
│   ├── matrix_io.c
│   └── test_generator.c
│
├── sequential.c
├── sequential_omp_schedules.c
│
├── cuda_spmv.cu
├── cuda_spmv_optimized.cu
├── cuda_spmv_ell_optimized.cu
├── cuda_spmv_hybrid_optimized.cu
│
├── cuda_test.cu
├── matrix_analysis.c
│
├── gprof_report.txt
├── RUN_COMMANDS.txt
└── Run_ELL_and_HYBRID.txt
```

## 7. Requirements

- Linux or Ubuntu (WSL2 is also supported).
- GCC compiler.
- NVIDIA CUDA Toolkit and `nvcc` for CUDA implementations.
- NVIDIA GPU compatible with the installed CUDA Toolkit.
- OpenMP support for the OpenMP implementation.

Check the available tools:

```bash
gcc --version
nvcc --version
nvidia-smi
```

## 8. Building and Running

Run commands from the repository root directory because dataset paths are relative.

### Generate or Regenerate Datasets

The repository includes pre-generated datasets.

To regenerate them:

```bash
gcc -O2 matrix_generator/test_generator.c \
    matrix_generator/matrix_generator.c \
    matrix_generator/matrix_io.c \
    -lm -o matrix_generator/test_generator

cd matrix_generator
./test_generator
cd ..
```

### Sequential CPU Implementation

Build:

```bash
gcc -O2 sequential.c \
    matrix_generator/matrix_io.c \
    -lm -o sequential
```

Run with the uniform matrix:

```bash
./sequential
```

Run with the irregular matrix:

```bash
./sequential --irregular
```

The output is saved to `cpu_output.txt`.

### Runtime-Adaptive CUDA Implementation

Build:

```bash
gcc -O2 -c matrix_generator/matrix_io.c -o matrix_io.o

nvcc -O3 cuda_spmv.cu matrix_io.o \
    -lm -o cuda_spmv
```

Run with the uniform matrix:

```bash
./cuda_spmv
```

Run with the irregular matrix:

```bash
./cuda_spmv --irregular
```

The output is saved to `cuda_output.txt`.

### OpenMP Implementation

Build:

```bash
gcc -O2 -fopenmp sequential_omp_schedules.c \
    matrix_generator/matrix_io.c \
    matrix_generator/matrix_generator.c \
    -lm -o sequential_omp_schedules
```

Run with four CPU threads:

```bash
OMP_NUM_THREADS=4 ./sequential_omp_schedules
```

For the irregular matrix:

```bash
OMP_NUM_THREADS=4 ./sequential_omp_schedules --irregular
```

### Additional Optimized CUDA Implementations

Refer to `RUN_COMMANDS.txt` and `Run_ELL_and_HYBRID.txt` for the corresponding build and execution commands.

## 9. Performance Evaluation

The project evaluates implementations using uniform and irregular sparse matrix inputs.

The main evaluation criteria include:

- Execution time.
- Speedup relative to the sequential baseline.
- Effect of row-length variation.
- Impact of different sparse storage formats.
- Correctness of computed output vectors.

Speedup is calculated as:

\[
Speedup = \frac{\text{Sequential Execution Time}}{\text{Parallel Execution Time}}
\]

For meaningful comparisons, measurements should use the same input dataset and clearly distinguish kernel execution time from total application time, including memory transfers and format conversion where applicable.

The sequential and CUDA implementations use different timing methods and report their respective measurement intervals.

## 10. Correctness Verification

The CPU and GPU implementations save their output vectors to separate files:

- `cpu_output.txt`
- `cuda_output.txt`

The output vectors can be compared element by element.

Because floating-point calculations may produce small rounding differences, correctness should be checked using an appropriate numerical tolerance rather than exact equality.

## 11. Profiling

The sequential implementation was profiled using GPROF.

The profiling results identified the SpMV computation as the dominant measured CPU hotspot, motivating the exploration of parallel execution approaches.

## 12. Conclusion

This project explores the performance of Sparse Matrix-Vector Multiplication across sequential CPU, OpenMP and CUDA implementations.

By comparing CSR, ELL and Hybrid storage layouts on uniform and irregular sparse matrices, the project investigates how sparse matrix structure affects workload distribution, memory usage and execution performance.

The overall objective is to understand and improve the efficiency of SpMV through appropriate data representation and parallel computation.

---

**Project:** High Performance Computing – Sparse Matrix Optimization  
**Repository:** [warshi1h/Sparse-Matrix-Optimization](https://github.com/warshi1h/Sparse-Matrix-Optimization)
