// Build: gcc -O2 -fopenmp -o heat2d_omp heat2d_omp.c -lm
// Run:   ./heat2d_omp <N> <timesteps> [num_threads]
//        (if num_threads omitted, uses OMP_NUM_THREADS or system default)
//
// This is the OpenMP-parallel counterpart of heat2d_serial.c (Milestone 1).
// Algorithm, initial condition, boundary conditions, metrics, and validation
// are IDENTICAL to the serial version -- only the spatial stencil loop is
// parallelized. This is intentional: it keeps the correctness comparison
// and the metric definitions apples-to-apples with the serial baseline.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>

/* ---------- Problem / discretization parameters ---------- */
#define ALPHA 0.01       /* thermal diffusivity */
#define DOMAIN_LEN 1.0   /* physical size of domain along one axis */

/* Index helper: row-major flattening of a (N x N) grid. */
static inline long idx(int i, int j, int N) {
    return (long)i * N + j;
}

/* Analytical solution for 2D heat diffusion of an initial Gaussian
 * heat spot on an unbounded domain (see heat2d_serial.c for derivation). */
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
        fprintf(stderr, "Usage: %s <N> <timesteps> [num_threads]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);      /* grid is N x N interior+boundary points */
    int nsteps = atoi(argv[2]);

    if (argc >= 4) {
        int nthreads = atoi(argv[3]);
        omp_set_num_threads(nthreads);
    }

    /* Default scheduling policy: dynamic,1. Empirically (see M2 report),
     * on heterogeneous P-core/E-core CPUs, static's equal-sized chunks
     * assume every thread is equally fast -- false here -- and the
     * per-timestep barrier then drags every thread down to the slowest
     * one, every iteration. dynamic,1 self-balances: fast threads simply
     * pull more chunks from the shared queue. This is only a DEFAULT --
     * setting OMP_SCHEDULE in the environment still overrides it, which
     * is how the static/dynamic/guided comparison in the M2 report was
     * produced. */
    if (!getenv("OMP_SCHEDULE")) {
        omp_set_schedule(omp_sched_dynamic, 1);
    }

    double h = DOMAIN_LEN / (N - 1);  /* grid spacing */
    double dt_stable_limit = 0.25 * h * h / ALPHA;
    double dt = 0.9 * dt_stable_limit; /* stay safely under CFL-like limit */
    double r = ALPHA * dt / (h * h);

    /* Report the thread count actually in effect (useful for logging sweeps) */
    int reported_threads;
    #pragma omp parallel
    {
        #pragma omp single
        reported_threads = omp_get_num_threads();
    }

    omp_sched_t sched_kind;
    int sched_chunk;
    omp_get_schedule(&sched_kind, &sched_chunk);
    /* GCC/libgomp sets a "monotonic" modifier bit (0x80000000) on top of
     * the base kind when the policy comes from OMP_SCHEDULE, so mask it
     * off before comparing -- otherwise env-var-set policies misreport
     * as "auto" here (cosmetic only; the actual scheduling is unaffected). */
    unsigned sched_kind_base = ((unsigned)sched_kind) & 0xF;
    const char* sched_name =
        sched_kind_base == omp_sched_static  ? "static"  :
        sched_kind_base == omp_sched_dynamic ? "dynamic" :
        sched_kind_base == omp_sched_guided  ? "guided"  : "auto";

    printf("Grid: %dx%d | timesteps: %d | threads: %d | schedule: %s,%d | "
           "h=%.6f | dt=%.6e | r=%.4f (stable if <=0.25)\n",
           N, N, nsteps, reported_threads, sched_name, sched_chunk, h, dt, r);

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

    /* ---------- Time-stepping loop (the actual stencil kernel) ----------
     *
     * Design notes (see M2 discussion):
     *  - Single `parallel` region wraps the ENTIRE timestep loop, so the
     *    thread team is forked once, not once per timestep. Fork/join
     *    overhead would otherwise dominate for small N / many steps.
     *  - `#pragma omp for schedule(runtime)` on the outer spatial (i) loop.
     *    Policy is chosen at runtime via the OMP_SCHEDULE env var (e.g.
     *    OMP_SCHEDULE=static or OMP_SCHEDULE=dynamic,4), not hardcoded,
     *    so the same binary can A/B test scheduling policies without a
     *    rebuild. On a HOMOGENEOUS set of cores, static is optimal (equal
     *    chunks, minimal overhead, cache-friendly contiguous rows). On
     *    this machine's HETEROGENEOUS P-core/E-core topology, static's
     *    equal-sized chunks assume every thread is equally fast, which is
     *    false here -- the implicit barrier below then forces fast
     *    threads to sit idle waiting on slow E-core threads, EVERY
     *    timestep. dynamic (or guided) lets fast threads opportunistically
     *    grab more chunks, self-balancing without needing explicit thread
     *    pinning. Default if OMP_SCHEDULE is unset is implementation
     *    defined (GCC: static).
     *  - `#pragma omp single` performs the pointer swap once (not once
     *    per thread). Its implicit barrier ensures the swap is visible
     *    to all threads before the next step reads u_old.
     *  - u_old is read-only and u_new is write-only within a step, and
     *    each thread writes only to its own disjoint set of cells, so
     *    there are no races and no reduction is needed. The result is
     *    bit-identical to the serial version regardless of thread count.
     */
    double t_start, t_end;
    t_start = omp_get_wtime();

    #pragma omp parallel
    {
        for (int step = 0; step < nsteps; step++) {
            #pragma omp for schedule(runtime)
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
            /* implicit barrier at end of `omp for` above */

            #pragma omp single
            {
                double* tmp = u_old;
                u_old = u_new;
                u_new = tmp;
            }
            /* implicit barrier at end of `omp single` above */
        }
    }

    t_end = omp_get_wtime();
    double elapsed = t_end - t_start;

    /* ---------- Metrics (identical convention to serial baseline) ---------- */
    long long interior_pts = (long long)(N - 2) * (N - 2);
    long long total_pt_updates =
        interior_pts * (long long)nsteps;

    const int flops_per_point = 8;

    double total_flops =
        (double)total_pt_updates * flops_per_point;
    double gflops_per_sec =
        total_flops / elapsed / 1e9;
    double grid_per_sec =
        (double)total_pt_updates / elapsed;

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

    /* ---------- Optional: dump final grid for bit-exact diff against serial ----------
     * Enable by setting env var HEAT_DUMP=<path>. Lets you verify the parallel
     * and serial versions produce IDENTICAL output arrays, not just similar
     * error metrics -- a stronger correctness claim than error-tolerance alone. */
    const char* dump_path = getenv("HEAT_DUMP");
    if (dump_path) {
        FILE* f = fopen(dump_path, "wb");
        if (f) {
            fwrite(u_old, sizeof(double), (size_t)N * N, f);
            fclose(f);
        }
    }

    free(u_old);
    free(u_new);

    return 0;
}