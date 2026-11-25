
#include "matrix.h"
#include <string.h>

/*
 * Compare two complete datasets.
 */

int compare_datasets(
    const SpmvDataset *a,
    const SpmvDataset *b)
{
    if (a->rows != b->rows ||
        a->cols != b->cols ||
        a->nnz != b->nnz)
        return 0;

    if (memcmp(a->values, b->values,
               (size_t)a->nnz * sizeof(double)) != 0)
        return 0;

    if (memcmp(a->col_idx, b->col_idx,
               (size_t)a->nnz * sizeof(int)) != 0)
        return 0;

    if (memcmp(a->row_ptr, b->row_ptr,
               ((size_t)a->rows + 1) * sizeof(int)) != 0)
        return 0;

    if (memcmp(a->x, b->x,
               (size_t)a->cols * sizeof(double)) != 0)
        return 0;

    return 1;
}

/*
 * Test one matrix generation and save/load cycle.
 */

void test_matrix(
    const char *name,
    SpmvDataset *original,
    const char *filename)
{
    SpmvDataset loaded = {0};

    printf("\n================================\n");
    printf("Testing: %s\n", name);
    printf("================================\n");

    print_matrix_info(original);

    printf("\nSaving matrix...\n");

    if (!save_matrix(filename, original))
    {
        printf("ERROR: Save failed!\n");
        return;
    }

    printf("Save successful!\n");

    printf("\nLoading matrix...\n");

    if (!load_matrix(filename, &loaded))
    {
        printf("ERROR: Load failed!\n");
        return;
    }

    printf("Load successful!\n");

    printf("\nComparing original and loaded datasets...\n");

    if (compare_datasets(original, &loaded))
        printf("SUCCESS: All arrays are identical!\n");
    else
        printf("FAILURE: Dataset mismatch!\n");

    free_dataset(&loaded);
}

int main(void)
{
    SpmvDataset uniform = {0};
    SpmvDataset irregular = {0};

    /*
     * Uniform matrix configuration.
     */

    MatrixConfig config = {
        .rows = 4000,
        .cols = 4000,
        .density = 0.10,
        .seed = 42
    };

    printf("Generating uniform matrix...\n");

    if (!generate_uniform_matrix(&uniform, &config))
    {
        printf("Uniform generation failed!\n");
        return 1;
    }

    test_matrix(
        "UNIFORM MATRIX",
        &uniform,
        "data/uniform_matrix.bin");

    /*
     * Irregular matrix generation.
     */

    printf("\nGenerating irregular matrix...\n");

    if (!generate_irregular_matrix(
            &irregular,
            4000,
            4000,
            42))
    {
        printf("Irregular generation failed!\n");
        free_dataset(&uniform);
        return 1;
    }

    test_matrix(
        "IRREGULAR MATRIX",
        &irregular,
        "data/irregular_matrix.bin");

    /*
     * Cleanup.
     */

    free_dataset(&uniform);
    free_dataset(&irregular);

    printf("\n================================\n");
    printf("Testing completed!\n");
    printf("================================\n");

    return 0;
}
