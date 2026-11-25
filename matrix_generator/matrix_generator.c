
#include "matrix.h"
#include <math.h>
#include <string.h>

/*
 * Simple reproducible random number generator.
 * Does not depend on global rand() state.
 */

static uint32_t rng_state;

static void seed_rng(unsigned int seed)
{
    rng_state = seed ? seed : 1;
}

static uint32_t next_random(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;

    return rng_state;
}

static double random_double(void)
{
    return (double)next_random() / 4294967296.0;
}

/*
 * Generate the common input vector.
 */

static void generate_vector(SpmvDataset *dataset)
{
    for (int i = 0; i < dataset->cols; i++)
    {
        dataset->x[i] = random_double();
    }
}

/*
 * Allocate CSR arrays after determining NNZ.
 */

static int allocate_dataset(
    SpmvDataset *dataset,
    int rows,
    int cols,
    int nnz)
{
    dataset->rows = rows;
    dataset->cols = cols;
    dataset->nnz = nnz;

    dataset->values = NULL;
    dataset->col_idx = NULL;
    dataset->row_ptr = NULL;
    dataset->x = NULL;

    dataset->row_ptr =
        malloc(((size_t)rows + 1) * sizeof(int));

    dataset->values =
        nnz > 0 ? malloc((size_t)nnz * sizeof(double)) : NULL;

    dataset->col_idx =
        nnz > 0 ? malloc((size_t)nnz * sizeof(int)) : NULL;

    dataset->x =
        malloc((size_t)cols * sizeof(double));

    if (!dataset->row_ptr ||
        (nnz > 0 && (!dataset->values || !dataset->col_idx)) ||
        !dataset->x)
    {
        free(dataset->row_ptr);
        free(dataset->values);
        free(dataset->col_idx);
        free(dataset->x);

        memset(dataset, 0, sizeof(*dataset));

        return 0;
    }

    return 1;
}

/*
 * Generate a uniform sparse matrix.
 *
 * density = probability that an element is nonzero.
 *
 * Two passes ensure that we allocate exactly enough
 * CSR storage instead of allocating a dense matrix.
 */

int generate_uniform_matrix(
    SpmvDataset *dataset,
    const MatrixConfig *config)
{
    if (!dataset || !config ||
        config->rows <= 0 ||
        config->cols <= 0 ||
        config->density < 0.0 ||
        config->density > 1.0)
    {
        return 0;
    }

    memset(dataset, 0, sizeof(*dataset));

    seed_rng(config->seed);

    long long total = 0;

    /*
     * Pass 1: Count nonzero elements.
     */

    for (int row = 0; row < config->rows; row++)
    {
        for (int col = 0; col < config->cols; col++)
        {
            if (random_double() < config->density)
            {
                if (++total > INT32_MAX)
                    return 0;

                random_double(); // Consume value RNG draw.
            }
        }
    }

    if (!allocate_dataset(
            dataset,
            config->rows,
            config->cols,
            (int)total))
    {
        return 0;
    }

    /*
     * Pass 2: Generate actual CSR matrix.
     */

    seed_rng(config->seed);

    int index = 0;

    dataset->row_ptr[0] = 0;

    for (int row = 0; row < config->rows; row++)
    {
        for (int col = 0; col < config->cols; col++)
        {
            if (random_double() < config->density)
            {
                dataset->values[index] =
                    1.0 + random_double() * 9.0;

                dataset->col_idx[index] = col;

                index++;
            }
        }

        dataset->row_ptr[row + 1] = index;
    }

    generate_vector(dataset);

    return 1;
}

/*
 * Generate an irregular sparse matrix.
 *
 * Each row receives a target NNZ count from
 * one of four distributions.
 *
 * Column indices are unique within each row.
 */

int generate_irregular_matrix(
    SpmvDataset *dataset,
    int rows,
    int cols,
    unsigned int seed)
{
    if (!dataset || rows <= 0 || cols < 1200)
        return 0;

    memset(dataset, 0, sizeof(*dataset));

    int *row_counts = malloc((size_t)rows * sizeof(int));

    int *columns = malloc((size_t)cols * sizeof(int));

    if (!row_counts || !columns)
    {
        free(row_counts);
        free(columns);
        return 0;
    }

    seed_rng(seed);

    long long total = 0;

    /*
     * Determine row lengths.
     */

    for (int row = 0; row < rows; row++)
    {
        double probability = random_double();

        int count;

        if (probability < 0.25)
            count = 20 + next_random() % 81;

        else if (probability < 0.50)
            count = 150 + next_random() % 151;

        else if (probability < 0.75)
            count = 400 + next_random() % 201;

        else
            count = 800 + next_random() % 401;

        if (count > cols)
        {
            free(row_counts);
            free(columns);
            return 0;
        }

        row_counts[row] = count;
        total += count;
    }

    if (total > INT32_MAX ||
        !allocate_dataset(dataset, rows, cols, (int)total))
    {
        free(row_counts);
        free(columns);
        return 0;
    }

    /*
     * Generate unique random columns for each row.
     */

    seed_rng(seed);

    dataset->row_ptr[0] = 0;

    int index = 0;

    for (int row = 0; row < rows; row++)
    {
        /*
         * Initialize column positions.
         */

        for (int col = 0; col < cols; col++)
            columns[col] = col;

        /*
         * Partial Fisher-Yates shuffle.
         * Select exactly row_counts[row] unique columns.
         */

        int count = row_counts[row];

        for (int k = 0; k < count; k++)
        {
            int selected =
                k + next_random() % (cols - k);

            int temp = columns[k];
            columns[k] = columns[selected];
            columns[selected] = temp;

            dataset->col_idx[index] = columns[k];

            dataset->values[index] =
                1.0 + random_double() * 9.0;

            index++;
        }

        dataset->row_ptr[row + 1] = index;
    }

    generate_vector(dataset);

    free(row_counts);
    free(columns);

    return 1;
}

/*
 * Free all allocated dataset memory.
 */

void free_dataset(SpmvDataset *dataset)
{
    if (!dataset)
        return;

    free(dataset->values);
    free(dataset->col_idx);
    free(dataset->row_ptr);
    free(dataset->x);

    memset(dataset, 0, sizeof(*dataset));
}

/*
 * Display matrix characteristics.
 */

void print_matrix_info(const SpmvDataset *dataset)
{
    if (!dataset)
        return;

    printf("\nMatrix Information\n");
    printf("------------------\n");
    printf("Rows       : %d\n", dataset->rows);
    printf("Columns    : %d\n", dataset->cols);
    printf("NNZ        : %d\n", dataset->nnz);

    double density =
        (double)dataset->nnz /
        ((double)dataset->rows * dataset->cols);

    printf("Density    : %.4f%%\n", density * 100.0);

    int min_nnz = dataset->row_ptr[1] - dataset->row_ptr[0];
    int max_nnz = min_nnz;
    long long total_nnz = 0;

    for (int i = 0; i < dataset->rows; i++)
    {
        int row_nnz =
            dataset->row_ptr[i + 1] - dataset->row_ptr[i];

        if (row_nnz < min_nnz)
            min_nnz = row_nnz;

        if (row_nnz > max_nnz)
            max_nnz = row_nnz;

        total_nnz += row_nnz;
    }

    printf("Min NNZ/row: %d\n", min_nnz);
    printf("Max NNZ/row: %d\n", max_nnz);
    printf("Avg NNZ/row: %.2f\n",
           (double)total_nnz / dataset->rows);
}
