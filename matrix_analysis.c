#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main()
{
    int rows = 4000;
    int cols = 4000;

    /*
     * We deliberately create an irregular sparse matrix.
     *
     * Some rows will contain very few nonzeros,
     * while other rows will contain many nonzeros.
     */

    int *row_nnz =
        malloc(rows * sizeof(int));

    if (row_nnz == NULL)
    {
        printf("Memory allocation failed!\n");
        return 1;
    }


    srand(123);


    /*
     * Generate different row densities.
     */

    for (int i = 0; i < rows; i++)
    {
        double probability =
            (double)rand() / RAND_MAX;

        if (probability < 0.25)
        {
            /*
             * Very sparse rows
             */
            row_nnz[i] =
                20 + rand() % 81;
        }
        else if (probability < 0.50)
        {
            /*
             * Medium-sparse rows
             */
            row_nnz[i] =
                150 + rand() % 151;
        }
        else if (probability < 0.75)
        {
            /*
             * Medium-dense rows
             */
            row_nnz[i] =
                400 + rand() % 201;
        }
        else
        {
            /*
             * Dense rows
             */
            row_nnz[i] =
                800 + rand() % 401;
        }
    }


    /*
     * Calculate total nonzeros.
     */

    long long total_nnz = 0;

    int min_nnz = row_nnz[0];
    int max_nnz = row_nnz[0];


    for (int i = 0; i < rows; i++)
    {
        total_nnz += row_nnz[i];

        if (row_nnz[i] < min_nnz)
            min_nnz = row_nnz[i];

        if (row_nnz[i] > max_nnz)
            max_nnz = row_nnz[i];
    }


    /*
     * Calculate average row length.
     */

    double average_nnz =
        (double)total_nnz / rows;


    /*
     * Calculate standard deviation.
     */

    double variance = 0.0;


    for (int i = 0; i < rows; i++)
    {
        double difference =
            row_nnz[i] - average_nnz;

        variance +=
            difference * difference;
    }


    variance /= rows;


    double stddev =
        sqrt(variance);


    /*
     * Coefficient of variation.
     */

    double coefficient_of_variation =
        (stddev / average_nnz) * 100.0;


    /*
     * Print matrix characteristics.
     */

    printf("========================================\n");
    printf("Irregular Sparse Matrix Analysis\n");
    printf("========================================\n");

    printf("Matrix size: %d x %d\n",
           rows,
           cols);

    printf("Total nonzeros: %lld\n",
           total_nnz);

    printf("Minimum nonzeros per row: %d\n",
           min_nnz);

    printf("Maximum nonzeros per row: %d\n",
           max_nnz);

    printf("Average nonzeros per row: %.2f\n",
           average_nnz);

    printf("Row NNZ standard deviation: %.2f\n",
           stddev);

    printf("Coefficient of variation: %.2f%%\n",
           coefficient_of_variation);


    /*
     * Runtime layout decision.
     */

    printf("\nRuntime layout decision:\n");

    if (coefficient_of_variation < 10.0)
    {
        printf("Selected layout: CSR\n");
        printf("Reason: Row lengths are relatively uniform.\n");
    }
    else
    {
        printf("Selected layout: HYBRID\n");
        printf("Reason: Row lengths have high variation.\n");
    }


    free(row_nnz);

    return 0;
}
