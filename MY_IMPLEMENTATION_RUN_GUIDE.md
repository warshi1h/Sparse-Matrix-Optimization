HPC Project – My Implementation Run Guide

Project: Sparse Matrix Optimization
Scope: Sequential CPU, Basic CUDA CSR, Optimized CUDA CSR
Environment: Ubuntu 22.04 / WSL2

1. Compile the Matrix Generator Files

Run these commands from the project root:

mkdir -p build

gcc -O2 -c matrix_generator/matrix_io.c -o build/matrix_io.o

gcc -O2 -c matrix_generator/matrix_generator.c -o build/matrix_generator.o

2. Sequential CPU Implementation

Compile:

gcc -O2 sequential.c build/matrix_io.o build/matrix_generator.o -lm -o sequential

Run:

./sequential

This is the sequential CPU baseline used for performance comparison.

3. Basic CUDA CSR Implementation

Compile:

nvcc -O3 cuda_spmv.cu build/matrix_io.o build/matrix_generator.o -lm -o cuda_spmv

Run:

./cuda_spmv

This implementation performs Sparse Matrix-Vector Multiplication using CUDA with CSR storage.

4. Optimized CUDA CSR Implementation

Compile:

nvcc -O3 cuda_spmv_optimized.cu build/matrix_io.o build/matrix_generator.o -lm -o cuda_spmv_optimized

Run:

./cuda_spmv_optimized

This implementation uses an optimized CUDA CSR kernel to improve GPU execution performance.

5. CPU Profiling Using GPROF

Compile:

gcc -O2 -pg sequential.c matrix_generator/matrix_io.c matrix_generator/matrix_generator.c -lm -o sequential_gprof

Run:

./sequential_gprof

Generate the profiling report:

gprof ./sequential_gprof gmon.out > gprof_report.txt

View the report:

less gprof_report.txt

GPROF identifies the CPU functions that consume the most execution time.

6. Verify and Compare Results

Check the generated output files:

ls -lh cpu_output.txt cuda_output.txt

Compare CPU and CUDA results:

diff -u cpu_output.txt cuda_output.txt
