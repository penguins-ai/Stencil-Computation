// Build: gcc -O2 -o compare_bins compare_bins.c -lm
// Usage: ./compare_bins <file_a.bin> <file_b.bin> <N>
//
// Reads two raw double arrays (as produced by HEAT_DUMP=... from
// heat2d_serial / heat2d_omp) and reports how different they actually are:
// max absolute difference, max relative (ULP-scale) difference, and how
// many of the N*N entries differ at all. This distinguishes "tiny
// floating-point rounding noise from different compiler codegen" (expected,
// harmless) from "a real logic bug" (large, structured differences).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <file_a> <file_b> <N>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[3]);
    size_t count = (size_t)N * N;

    FILE* fa = fopen(argv[1], "rb");
    FILE* fb = fopen(argv[2], "rb");
    if (!fa || !fb) {
        fprintf(stderr, "Could not open input files\n");
        return 1;
    }

    double* a = malloc(count * sizeof(double));
    double* b = malloc(count * sizeof(double));

    size_t ra = fread(a, sizeof(double), count, fa);
    size_t rb = fread(b, sizeof(double), count, fb);

    if (ra != count || rb != count) {
        fprintf(stderr, "File size mismatch: expected %zu doubles, got %zu and %zu\n",
                count, ra, rb);
        return 1;
    }

    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    size_t n_differ = 0;
    size_t first_diff_idx = (size_t)-1;

    for (size_t k = 0; k < count; k++) {
        double diff = fabs(a[k] - b[k]);
        if (diff != 0.0) {
            n_differ++;
            if (first_diff_idx == (size_t)-1) first_diff_idx = k;
        }
        if (diff > max_abs_diff) max_abs_diff = diff;

        double denom = fmax(fabs(a[k]), 1e-300);
        double rel = diff / denom;
        if (rel > max_rel_diff) max_rel_diff = rel;
    }

    printf("Total points: %zu\n", count);
    printf("Points that differ at all: %zu (%.4f%%)\n",
           n_differ, 100.0 * n_differ / count);
    printf("Max absolute difference: %.6e\n", max_abs_diff);
    printf("Max relative difference: %.6e\n", max_rel_diff);
    if (first_diff_idx != (size_t)-1) {
        int i = first_diff_idx / N, j = first_diff_idx % N;
        printf("First differing point: (i=%d, j=%d) a=%.17g b=%.17g\n",
               i, j, a[first_diff_idx], b[first_diff_idx]);
    }

    if (max_abs_diff == 0.0) {
        printf("\n=> Files are numerically identical.\n");
    } else if (max_abs_diff < 1e-10) {
        printf("\n=> Difference is at floating-point rounding noise level "
               "(likely compiler codegen difference, not a logic bug).\n");
    } else {
        printf("\n=> Difference is LARGE -- likely an actual bug, not rounding noise.\n");
    }

    return 0;
}
