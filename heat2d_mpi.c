// Build: mpicc -O2 -ffp-contract=off -o heat2d_mpi heat2d_mpi.c -lm
// Run:   mpirun -np <P> ./heat2d_mpi <N> <timesteps>
//
// MPI (distributed-memory) counterpart of heat2d_serial.c / heat2d_omp.c.
// Same algorithm, initial condition, boundary conditions, metrics and
// validation. Only the parallelization model differs:
//
//  - 1D row-block decomposition: the N x N grid is split into P
//    contiguous bands of rows (sizes differ by at most 1 row).
//  - Each rank stores its rows plus 2 GHOST rows (above and below).
//  - Every timestep: exchange boundary rows with the up/down neighbours
//    (MPI_Sendrecv), then update the owned rows, then swap pointers.
//  - Global boundary rows/cols stay 0 (Dirichlet) and are never updated.
//
// The per-point arithmetic is written in exactly the same order as the
// serial code, so the result should be bit-identical to it for any P.
//
// Optional environment variables (all "off" unless set to something != 0):
//   HEAT_DUMP=<path>   gather final grid on rank 0 and write it to <path>
//   HEAT_SPLIT=1       put an MPI_Barrier before each halo exchange and time
//                      it separately, so "waiting for slow neighbours" is
//                      separated from "actually moving the halo rows".
//                      (Adds a barrier per step, so total time is slightly
//                      perturbed - use it for diagnosis, not for headline
//                      timings.)
//   HEAT_SHOW_CPU=1    print which logical CPU each rank is running on
//                      (snapshot at the end of the run) to verify pinning.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sched.h>
#include <mpi.h>

#define ALPHA 0.01
#define DOMAIN_LEN 1.0

static double analytical(double x, double y, double xc, double yc,
                         double t, double alpha, double A0,
                         double sigma0)
{
    double denom = sigma0 * sigma0 + 2.0 * alpha * t;
    double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
    return A0 * (sigma0 * sigma0 / denom) * exp(-r2 / (2.0 * denom));
}

static int env_on(const char *name)
{
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 3)
    {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np P %s <N> <timesteps>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int N = atoi(argv[1]);
    int nsteps = atoi(argv[2]);
    if (N < 3 * P)
    {
        if (rank == 0)
            fprintf(stderr, "N too small for %d ranks\n", P);
        MPI_Finalize();
        return 1;
    }

    /* Rank 0 reads the switches and tells everyone (so all ranks agree). */
    int flags[2] = {0, 0};
    if (rank == 0)
    {
        flags[0] = env_on("HEAT_SPLIT");
        flags[1] = env_on("HEAT_SHOW_CPU");
    }
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    int split = flags[0], show_cpu = flags[1];

    double h = DOMAIN_LEN / (N - 1);
    double dt = 0.9 * (0.25 * h * h / ALPHA);
    double r = ALPHA * dt / (h * h);

    /* ---------- Domain decomposition (rows) ---------- */
    int base = N / P, rem = N % P;
    int local_n = base + (rank < rem ? 1 : 0);           /* rows owned      */
    int start = rank * base + (rank < rem ? rank : rem); /* first global row */
    int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down = (rank < P - 1) ? rank + 1 : MPI_PROC_NULL;

    /* local row l (1..local_n) <-> global row start + l - 1
     * local row 0 and local_n+1 are ghost rows */
    size_t bytes = (size_t)(local_n + 2) * N * sizeof(double);
    double *u_old = (double *)calloc((size_t)(local_n + 2) * N, sizeof(double));
    double *u_new = (double *)calloc((size_t)(local_n + 2) * N, sizeof(double));
    if (!u_old || !u_new)
    {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
        printf("Grid: %dx%d | timesteps: %d | ranks: %d | h=%.6f | dt=%.6e | "
               "r=%.4f (stable if <=0.25)\n",
               N, N, nsteps, P, h, dt, r);

    /* ---------- Initial condition (local rows only) ---------- */
    double xc = DOMAIN_LEN / 2.0, yc = DOMAIN_LEN / 2.0;
    double A0 = 100.0, sigma0 = 5.0 * h;

    for (int l = 1; l <= local_n; l++)
    {
        int g = start + l - 1;
        for (int j = 0; j < N; j++)
        {
            u_old[(size_t)l * N + j] =
                analytical(g * h, j * h, xc, yc, 0.0, ALPHA, A0, sigma0);
        }
    }
    memcpy(u_new, u_old, bytes);

    /* ---------- Dirichlet boundaries ---------- */
    for (int l = 1; l <= local_n; l++)
    {
        u_old[(size_t)l * N + 0] = u_old[(size_t)l * N + N - 1] = 0.0;
        u_new[(size_t)l * N + 0] = u_new[(size_t)l * N + N - 1] = 0.0;
    }
    if (start == 0) /* global row 0 */
        for (int j = 0; j < N; j++)
            u_old[N + j] = u_new[N + j] = 0.0;
    if (start + local_n == N) /* global row N-1 */
        for (int j = 0; j < N; j++)
            u_old[(size_t)local_n * N + j] = u_new[(size_t)local_n * N + j] = 0.0;

    /* range of local rows that are true interior points */
    int l_lo = (start == 0) ? 2 : 1;
    int l_hi = (start + local_n == N) ? local_n - 1 : local_n;

    /* ---------- Time stepping ---------- */
    double comm_time = 0.0; /* time inside the two MPI_Sendrecv calls       */
    double wait_time = 0.0; /* time in the extra barrier (HEAT_SPLIT only)  */
    double comp_time = 0.0; /* time in the stencil loop + pointer swap      */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (int step = 0; step < nsteps; step++)
    {
        if (split)
        {
            double w0 = MPI_Wtime();
            MPI_Barrier(MPI_COMM_WORLD); /* everyone arrives together */
            wait_time += MPI_Wtime() - w0;
        }
        double c0 = MPI_Wtime();
        /* send my first owned row up, receive bottom ghost from below */
        MPI_Sendrecv(&u_old[(size_t)1 * N], N, MPI_DOUBLE, up, 0,
                     &u_old[(size_t)(local_n + 1) * N], N, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* send my last owned row down, receive top ghost from above */
        MPI_Sendrecv(&u_old[(size_t)local_n * N], N, MPI_DOUBLE, down, 1,
                     &u_old[0], N, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double c1 = MPI_Wtime();
        comm_time += c1 - c0;

        for (int l = l_lo; l <= l_hi; l++)
        {
            for (int j = 1; j < N - 1; j++)
            {
                size_t c = (size_t)l * N + j;
                size_t upi = c - N, dni = c + N, lf = c - 1, rt = c + 1;
                u_new[c] = u_old[c] + r * (u_old[upi] + u_old[dni] + u_old[lf] + u_old[rt] - 4.0 * u_old[c]);
            }
        }
        double *tmp = u_old;
        u_old = u_new;
        u_new = tmp;
        comp_time += MPI_Wtime() - c1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_local = MPI_Wtime() - t_start;
    double elapsed, comm_max, wait_max, comp_max, comp_min;
    MPI_Reduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wait_time, &wait_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &comp_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    /* ---------- Metrics (same convention as serial) ---------- */
    if (rank == 0)
    {
        long long interior_pts = (long long)(N - 2) * (N - 2);
        long long total_pt_updates = interior_pts * (long long)nsteps;
        const int flops_per_point = 8; /* kept at 8 to match the serial baseline */
        double gflops = (double)total_pt_updates * flops_per_point / elapsed / 1e9;
        double grid_per_sec = (double)total_pt_updates / elapsed;
        double ai = flops_per_point / (6.0 * sizeof(double));
        double imbalance = comp_max > 0 ? 100.0 * (comp_max - comp_min) / comp_max : 0.0;
        double halo_pct = 100.0 * comm_max / elapsed;

        printf("\n--- Performance ---\n");
        printf("Elapsed time: %.4f s\n", elapsed);
        printf("Halo-exchange time (max over ranks): %.4f s (%.1f%%)\n",
               comm_max, halo_pct);
        if (split)
            printf("Barrier wait before halo (max over ranks): %.4f s\n", wait_max);
        printf("Rank compute time (min / max over ranks): %.4f / %.4f s (imbalance %.1f%%)\n",
               comp_min, comp_max, imbalance);
        printf("Throughput: %.4e grid-points/s\n", grid_per_sec);
        printf("Compute rate: %.4f GFLOP/s\n", gflops);
        printf("Arithmetic intensity: %.4f FLOPs/byte (naive, untiled)\n", ai);

        /* one machine-readable line for the benchmark scripts:
         * RESULT,elapsed,halo_s,halo_pct,throughput,gflops,comp_min,comp_max,wait */
        printf("RESULT,%.6f,%.6f,%.2f,%.6e,%.4f,%.6f,%.6f,%.6f\n",
               elapsed, comm_max, halo_pct, grid_per_sec, gflops,
               comp_min, comp_max, wait_max);
    }

    /* ---------- Optional: which CPU is each rank on? ---------- */
    if (show_cpu)
    {
        int cpu = sched_getcpu();
        int *cpus = (rank == 0) ? (int *)malloc(P * sizeof(int)) : NULL;
        MPI_Gather(&cpu, 1, MPI_INT, cpus, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            printf("CPU map (rank:cpu):");
            for (int p = 0; p < P; p++)
                printf(" %d:%d", p, cpus[p]);
            printf("\n");
            free(cpus);
        }
    }

    /* ---------- Validation (interior region, global indices) ---------- */
    double t_final = nsteps * dt;
    double max_err = 0.0, sq = 0.0;
    for (int g = N / 4; g < 3 * N / 4; g++)
    {
        if (g < start || g >= start + local_n)
            continue;
        int l = g - start + 1;
        for (int j = N / 4; j < 3 * N / 4; j++)
        {
            double exact = analytical(g * h, j * h, xc, yc, t_final,
                                      ALPHA, A0, sigma0);
            double err = fabs(u_old[(size_t)l * N + j] - exact);
            if (err > max_err)
                max_err = err;
            sq += err * err;
        }
    }
    double g_max, g_sq;
    MPI_Reduce(&max_err, &g_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sq, &g_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        int w = 3 * N / 4 - N / 4;
        printf("\n--- Validation (interior region only) ---\n");
        printf("Simulated time: %.6f\n", t_final);
        printf("Max abs error: %.6e\n", g_max);
        printf("L2 error: %.6e\n", sqrt(g_sq / ((double)w * w)));
    }

    /* ---------- Optional dump for bit-exact diff ---------- */
    int do_dump = 0;
    const char *dump_path = NULL;
    if (rank == 0)
    {
        dump_path = getenv("HEAT_DUMP");
        do_dump = (dump_path != NULL);
    }
    MPI_Bcast(&do_dump, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (do_dump)
    {
        double *full = NULL;
        int *counts = NULL, *displs = NULL;
        if (rank == 0)
        {
            full = (double *)malloc((size_t)N * N * sizeof(double));
            counts = (int *)malloc(P * sizeof(int));
            displs = (int *)malloc(P * sizeof(int));
            for (int p = 0; p < P; p++)
            {
                int ln = base + (p < rem ? 1 : 0);
                int st = p * base + (p < rem ? p : rem);
                counts[p] = ln * N;
                displs[p] = st * N;
            }
        }
        MPI_Gatherv(&u_old[N], local_n * N, MPI_DOUBLE,
                    full, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            FILE *f = fopen(dump_path, "wb");
            if (f)
            {
                fwrite(full, sizeof(double), (size_t)N * N, f);
                fclose(f);
            }
            free(full);
            free(counts);
            free(displs);
        }
    }

    free(u_old);
    free(u_new);
    MPI_Finalize();
    return 0;
}
