
#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/*
 * Common CSR matrix and input vector.
 */

typedef struct
{
    int rows;
    int cols;
    int nnz;

    double *values;
    int *col_idx;
    int *row_ptr;

    double *x;

} SpmvDataset;


/*
 * Matrix generation configuration.
 */

typedef struct
{
    int rows;
    int cols;

    double density;

    unsigned int seed;

} MatrixConfig;


/*
 * Generate a uniform sparse matrix.
 */

int generate_uniform_matrix(
    SpmvDataset *dataset,
    const MatrixConfig *config
);


/*
 * Generate an irregular sparse matrix.
 */

int generate_irregular_matrix(
    SpmvDataset *dataset,
    int rows,
    int cols,
    unsigned int seed
);


/*
 * Save and load a complete dataset.
 */

int save_matrix(
    const char *filename,
    const SpmvDataset *dataset
);

int load_matrix(
    const char *filename,
    SpmvDataset *dataset
);


/*
 * Release allocated memory.
 */

void free_dataset(SpmvDataset *dataset);


/*
 * Display matrix characteristics.
 */

void print_matrix_info(
    const SpmvDataset *dataset
);

#endif
