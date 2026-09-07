#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* ---------- Problem / discretization parameters ---------- */
#define ALPHA 0.01       /* thermal diffusivity */
#define DOMAIN_LEN 1.0   /* physical size of domain along one axis */

/* Index helper: row-major flattening of a (N x N) grid, matches the
 * paper's choice of storing 3D data as a flat 1D array for cache-
 * friendly access (Section IV-A, "gmem" description). */
static inline long idx(int i, int j, int N) {
    return (long)i * N + j;
}

/* Analytical solution for 2D heat diffusion of an initial Gaussian
 * "point-ish" heat spot of amplitude A0 and initial variance sigma0^2,
 * on an unbounded domain:
 *
 * u(x,y,t) = A0 * sigma0^2 / (sigma0^2 + 2*alpha*t)
 *          * exp(-((x-xc)^2 + (y-yc)^2)
 *                / (2*(sigma0^2 + 2*alpha*t)))
 *
 * Valid as an approximation near the center of the grid, away from the
 * boundaries, since the true problem is bounded (Dirichlet u=0 edges)
 * while this formula assumes an infinite domain.
 */
static double analytical(double x, double y, double xc, double yc,
                         double t, double alpha, double A0,
                         double sigma0) {
    double denom = sigma0 * sigma0 + 2.0 * alpha * t;
    double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);

    return A0 * (sigma0 * sigma0 / denom)
           * exp(-r2 / (2.0 * denom));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <timesteps>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);      /* grid is N x N interior+boundary points */
    int nsteps = atoi(argv[2]);

    double h = DOMAIN_LEN / (N - 1);  /* grid spacing */
    double dt_stable_limit = 0.25 * h * h / ALPHA;
    double dt = 0.9 * dt_stable_limit; /* stay safely under CFL-like limit */
    double r = ALPHA * dt / (h * h);

    printf("Grid: %dx%d | timesteps: %d | h=%.6f | dt=%.6e | "
           "r=%.4f (stable if <=0.25)\n",
           N, N, nsteps, h, dt, r);

    size_t bytes = (size_t)N * N * sizeof(double);
    double* u_old = (double*)malloc(bytes);
    double* u_new = (double*)malloc(bytes);

    if (!u_old || !u_new) {
        fprintf(stderr, "Allocation failed for N=%d\n", N);
        return 1;
    }

    /* ---------- Initial condition: Gaussian heat spot at center ---------- */
    double xc = DOMAIN_LEN / 2.0, yc = DOMAIN_LEN / 2.0;
    double A0 = 100.0;
    double sigma0 = 5.0 * h; /* a few grid cells wide */

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = i * h, y = j * h;
            u_old[idx(i, j, N)] =
                analytical(x, y, xc, yc, 0.0, ALPHA, A0, sigma0);
        }
    }

    memcpy(u_new, u_old, bytes);

    /* ---------- Dirichlet boundary condition: u = 0 on all edges ---------- */
    for (int i = 0; i < N; i++) {
        u_old[idx(i, 0, N)] = 0.0;
        u_old[idx(i, N - 1, N)] = 0.0;
        u_new[idx(i, 0, N)] = 0.0;
        u_new[idx(i, N - 1, N)] = 0.0;
    }

    for (int j = 0; j < N; j++) {
        u_old[idx(0, j, N)] = 0.0;
        u_old[idx(N - 1, j, N)] = 0.0;
        u_new[idx(0, j, N)] = 0.0;
        u_new[idx(N - 1, j, N)] = 0.0;
    }

    /* ---------- Time-stepping loop (the actual stencil kernel) ---------- */
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int step = 0; step < nsteps; step++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                long c  = idx(i, j, N);
                long up = idx(i - 1, j, N);
                long dn = idx(i + 1, j, N);
                long lf = idx(i, j - 1, N);
                long rt = idx(i, j + 1, N);

                u_new[c] = u_old[c]
                           + r * (u_old[up] + u_old[dn]
                                + u_old[lf] + u_old[rt]
                                - 4.0 * u_old[c]);
            }
        }

        /* swap pointers instead of copying the whole grid */
        double* tmp = u_old;
        u_old = u_new;
        u_new = tmp;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    double elapsed =
        (t_end.tv_sec - t_start.tv_sec)
        + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;

    /* ---------- Metrics (paper-style: Table II / Fig. 10 conventions) ---------- */
    long long interior_pts = (long long)(N - 2) * (N - 2);
    long long total_pt_updates =
        interior_pts * (long long)nsteps;

    /* FLOPs per point update:
     * 4 adds (sum neighbors) + 1 mul (by 4.0)
     * + 1 sub + 1 mul (by r) + 1 add = 8 FLOPs.
     * Adjust if you count differently, but state your convention
     * in the report.
     */
    const int flops_per_point = 8;

    double total_flops =
        (double)total_pt_updates * flops_per_point;
    double gflops_per_sec =
        total_flops / elapsed / 1e9;
    double grid_per_sec =
        (double)total_pt_updates / elapsed;

    /* Naive arithmetic intensity:
     * 5 reads + 1 write per point update,
     * 8 bytes each (double), no reuse credited
     * (worst-case / gmem-style access, matching the paper's
     * "gmem" baseline before any tiling).
     */
    double bytes_per_point = 6.0 * sizeof(double);
    double arith_intensity =
        flops_per_point / bytes_per_point;

    printf("\n--- Performance ---\n");
    printf("Elapsed time: %.4f s\n", elapsed);
    printf("Throughput: %.4e grid-points/s\n", grid_per_sec);
    printf("Compute rate: %.4f GFLOP/s\n", gflops_per_sec);
    printf("Arithmetic intensity: %.4f FLOPs/byte "
           "(naive, untiled)\n",
           arith_intensity);

    /* ---------- Validation against analytical solution ---------- */
    double t_final = nsteps * dt;
    double max_abs_err = 0.0, l2_err = 0.0;

    for (int i = N / 4; i < 3 * N / 4; i++) {
        /* stay away from boundaries */
        for (int j = N / 4; j < 3 * N / 4; j++) {
            double x = i * h, y = j * h;

            double exact =
                analytical(x, y, xc, yc, t_final,
                           ALPHA, A0, sigma0);

            double err =
                fabs(u_old[idx(i, j, N)] - exact);

            if (err > max_abs_err)
                max_abs_err = err;

            l2_err += err * err;
        }
    }

    l2_err =
        sqrt(l2_err /
             ((3 * N / 4 - N / 4)
              * (3 * N / 4 - N / 4)));

    printf("\n--- Validation (interior region only) ---\n");
    printf("Simulated time: %.6f\n", t_final);
    printf("Max abs error: %.6e\n", max_abs_err);
    printf("L2 error: %.6e\n", l2_err);

    free(u_old);
    free(u_new);

    return 0;
}