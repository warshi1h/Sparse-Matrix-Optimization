
#include "matrix.h"
#include <string.h>
#include <limits.h>

/*
 * Binary file format:
 *
 * Magic number
 * Version
 * Rows
 * Columns
 * NNZ
 * CSR values
 * CSR column indices
 * CSR row pointers
 * Input vector X
 *
 * Integer arrays are stored as int32_t.
 */

#define MATRIX_MAGIC "SPMVDAT1"
#define MATRIX_VERSION 1

/*
 * Validate dataset before saving.
 */

static int validate_dataset(const SpmvDataset *d)
{
    if (!d || d->rows <= 0 || d->cols <= 0 || d->nnz < 0)
        return 0;

    if (!d->row_ptr || !d->x)
        return 0;

    if (d->nnz > 0 && (!d->values || !d->col_idx))
        return 0;

    if (d->row_ptr[0] != 0 ||
        d->row_ptr[d->rows] != d->nnz)
        return 0;

    for (int row = 0; row < d->rows; row++)
    {
        if (d->row_ptr[row] < 0 ||
            d->row_ptr[row] > d->row_ptr[row + 1] ||
            d->row_ptr[row + 1] > d->nnz)
            return 0;
    }

    for (int i = 0; i < d->nnz; i++)
    {
        if (d->col_idx[i] < 0 || d->col_idx[i] >= d->cols)
            return 0;
    }

    return 1;
}

/*
 * Save complete CSR dataset to a binary file.
 */

int save_matrix(
    const char *filename,
    const SpmvDataset *dataset)
{
    if (!filename || !validate_dataset(dataset))
        return 0;

    if (sizeof(int) != sizeof(int32_t))
        return 0;

    FILE *file = fopen(filename, "wb");

    if (!file)
    {
        perror("Unable to open matrix file");
        return 0;
    }

    const char magic[8] = MATRIX_MAGIC;

    uint32_t version = MATRIX_VERSION;
    uint32_t rows = (uint32_t)dataset->rows;
    uint32_t cols = (uint32_t)dataset->cols;
    uint64_t nnz = (uint64_t)dataset->nnz;

    int success = 1;

    /*
     * Write metadata.
     */

    success &= fwrite(magic, 1, 8, file) == 8;
    success &= fwrite(&version, sizeof(version), 1, file) == 1;
    success &= fwrite(&rows, sizeof(rows), 1, file) == 1;
    success &= fwrite(&cols, sizeof(cols), 1, file) == 1;
    success &= fwrite(&nnz, sizeof(nnz), 1, file) == 1;

    /*
     * Write CSR arrays.
     */

    if (success)
        success &= fwrite(
            dataset->values,
            sizeof(double),
            (size_t)dataset->nnz,
            file) == (size_t)dataset->nnz;

    if (success)
        success &= fwrite(
            dataset->col_idx,
            sizeof(int32_t),
            (size_t)dataset->nnz,
            file) == (size_t)dataset->nnz;

    if (success)
        success &= fwrite(
            dataset->row_ptr,
            sizeof(int32_t),
            (size_t)dataset->rows + 1,
            file) == (size_t)dataset->rows + 1;

    if (success)
        success &= fwrite(
            dataset->x,
            sizeof(double),
            (size_t)dataset->cols,
            file) == (size_t)dataset->cols;

    if (fclose(file) != 0)
        success = 0;

    if (!success)
        remove(filename);

    return success;
}

/*
 * Load complete CSR dataset from a binary file.
 */

int load_matrix(
    const char *filename,
    SpmvDataset *dataset)
{
    if (!filename || !dataset)
        return 0;

    memset(dataset, 0, sizeof(*dataset));

    if (sizeof(int) != sizeof(int32_t))
        return 0;

    FILE *file = fopen(filename, "rb");

    if (!file)
    {
        perror("Unable to open matrix file");
        return 0;
    }

    char magic[8];

    uint32_t version;
    uint32_t rows;
    uint32_t cols;
    uint64_t nnz;

    int success = 1;

    /*
     * Read metadata.
     */

    success &= fread(magic, 1, 8, file) == 8;
    success &= fread(&version, sizeof(version), 1, file) == 1;
    success &= fread(&rows, sizeof(rows), 1, file) == 1;
    success &= fread(&cols, sizeof(cols), 1, file) == 1;
    success &= fread(&nnz, sizeof(nnz), 1, file) == 1;

    if (!success ||
        memcmp(magic, MATRIX_MAGIC, 8) != 0 ||
        version != MATRIX_VERSION ||
        rows == 0 || cols == 0 ||
        rows > INT_MAX ||
        cols > INT_MAX ||
        nnz > INT_MAX)
    {
        fclose(file);
        return 0;
    }

    /*
     * Allocate dataset arrays.
     */

    dataset->rows = (int)rows;
    dataset->cols = (int)cols;
    dataset->nnz = (int)nnz;

    dataset->values =
        nnz > 0 ? malloc((size_t)nnz * sizeof(double)) : NULL;

    dataset->col_idx =
        nnz > 0 ? malloc((size_t)nnz * sizeof(int)) : NULL;

    dataset->row_ptr =
        malloc(((size_t)rows + 1) * sizeof(int));

    dataset->x =
        malloc((size_t)cols * sizeof(double));

    if ((nnz > 0 && (!dataset->values || !dataset->col_idx)) ||
        !dataset->row_ptr ||
        !dataset->x)
    {
        success = 0;
    }

    /*
     * Read CSR arrays and vector.
     */

    if (success && nnz > 0)
        success &= fread(
            dataset->values,
            sizeof(double),
            (size_t)nnz,
            file) == (size_t)nnz;

    if (success && nnz > 0)
        success &= fread(
            dataset->col_idx,
            sizeof(int32_t),
            (size_t)nnz,
            file) == (size_t)nnz;

    if (success)
        success &= fread(
            dataset->row_ptr,
            sizeof(int32_t),
            (size_t)rows + 1,
            file) == (size_t)rows + 1;

    if (success)
        success &= fread(
            dataset->x,
            sizeof(double),
            (size_t)cols,
            file) == (size_t)cols;

    /*
     * Reject truncated files and unexpected extra data.
     */

    if (success)
        success = fgetc(file) == EOF && !ferror(file);

    fclose(file);

    /*
     * Validate loaded dataset.
     */

    if (!success || !validate_dataset(dataset))
    {
        free_dataset(dataset);
        return 0;
    }

    return 1;
}
