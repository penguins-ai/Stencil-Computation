/*
 * heat2d_cuda_simple.cu
 * Direct CUDA port of the serial 2D heat diffusion code:
 * one GPU thread per grid point, 5-point stencil, pointer swap each step.
 *
 * BUILD:
 *   nvcc -O3 -arch=sm_75 -o heat2d_cuda_simple heat2d_cuda_simple.cu
 *   (sm_75 = T4 / GTX 16xx / RTX 20xx, sm_86 = RTX 30xx, sm_89 = RTX 40xx;
 *    on recent CUDA you can use -arch=native)
 *
 * RUN:
 *   ./heat2d_cuda_simple <N> <timesteps>
 *   e.g.  ./heat2d_cuda_simple 4096 1000
 *
 * Google Colab (T4 GPU runtime):
 *   !nvcc -O3 -arch=sm_75 -o heat2d_cuda_simple heat2d_cuda_simple.cu
 *   !./heat2d_cuda_simple 4096 1000
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define ALPHA 0.01
#define DOMAIN_LEN 1.0

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), \
            __FILE__, __LINE__); exit(1); } } while (0)

__host__ __device__ static inline long idx(int i, int j, int N) {
    return (long)i * N + j;
}

static double analytical(double x, double y, double xc, double yc,
                         double t, double alpha, double A0, double sigma0) {
    double denom = sigma0 * sigma0 + 2.0 * alpha * t;
    double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
    return A0 * (sigma0 * sigma0 / denom) * exp(-r2 / (2.0 * denom));
}

/* One thread updates one interior point. j maps to threadIdx.x so that
 * neighboring threads touch neighboring memory (coalesced access). */
__global__ void stencil_kernel(const double *u_old, double *u_new,
                               int N, double r) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= N - 1 || j < 1 || j >= N - 1) return;

    long c  = idx(i, j, N);
    long up = idx(i - 1, j, N);
    long dn = idx(i + 1, j, N);
    long lf = idx(i, j - 1, N);
    long rt = idx(i, j + 1, N);

    u_new[c] = u_old[c]
               + r * (u_old[up] + u_old[dn] + u_old[lf] + u_old[rt]
                      - 4.0 * u_old[c]);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <timesteps>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int nsteps = atoi(argv[2]);

    double h = DOMAIN_LEN / (N - 1);
    double dt_stable_limit = 0.25 * h * h / ALPHA;
    double dt = 0.9 * dt_stable_limit;
    double r = ALPHA * dt / (h * h);

    printf("Grid: %dx%d | timesteps: %d | h=%.6f | dt=%.6e | "
           "r=%.4f (stable if <=0.25)\n", N, N, nsteps, h, dt, r);

    size_t bytes = (size_t)N * N * sizeof(double);
    double *h_u = (double *)malloc(bytes);
    if (!h_u) { fprintf(stderr, "Allocation failed for N=%d\n", N); return 1; }

    /* Initial condition on the host */
    double xc = DOMAIN_LEN / 2.0, yc = DOMAIN_LEN / 2.0;
    double A0 = 100.0;
    double sigma0 = 5.0 * h;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            h_u[idx(i, j, N)] =
                analytical(i * h, j * h, xc, yc, 0.0, ALPHA, A0, sigma0);

    /* Dirichlet boundary: u = 0 on all edges */
    for (int i = 0; i < N; i++) {
        h_u[idx(i, 0, N)] = 0.0;
        h_u[idx(i, N - 1, N)] = 0.0;
    }
    for (int j = 0; j < N; j++) {
        h_u[idx(0, j, N)] = 0.0;
        h_u[idx(N - 1, j, N)] = 0.0;
    }

    /* Device buffers; both start identical so boundaries stay zero */
    double *d_old, *d_new;
    CUDA_CHECK(cudaMalloc(&d_old, bytes));
    CUDA_CHECK(cudaMalloc(&d_new, bytes));
    CUDA_CHECK(cudaMemcpy(d_old, h_u, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new, h_u, bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 8);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    /* ---------- Time-stepping loop ---------- */
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int step = 0; step < nsteps; step++) {
        stencil_kernel<<<grid, block>>>(d_old, d_new, N, r);
        double *tmp = d_old; d_old = d_new; d_new = tmp;  /* swap pointers */
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
    double elapsed = ms * 1e-3;

    /* Copy final result back (d_old holds the latest after the swap) */
    CUDA_CHECK(cudaMemcpy(h_u, d_old, bytes, cudaMemcpyDeviceToHost));

    /* ---------- Metrics ---------- */
    long long interior_pts = (long long)(N - 2) * (N - 2);
    long long total_pt_updates = interior_pts * (long long)nsteps;

    const int flops_per_point = 8;
    double total_flops = (double)total_pt_updates * flops_per_point;
    double gflops_per_sec = total_flops / elapsed / 1e9;
    double grid_per_sec = (double)total_pt_updates / elapsed;
    double bytes_per_point = 6.0 * sizeof(double);
    double arith_intensity = flops_per_point / bytes_per_point;

    printf("\n--- Performance ---\n");
    printf("Elapsed time: %.4f s\n", elapsed);
    printf("Throughput: %.4e grid-points/s\n", grid_per_sec);
    printf("Compute rate: %.4f GFLOP/s\n", gflops_per_sec);
    printf("Arithmetic intensity: %.4f FLOPs/byte (naive, untiled)\n",
           arith_intensity);

    /* ---------- Validation against analytical solution ---------- */
    double t_final = nsteps * dt;
    double max_abs_err = 0.0, l2_err = 0.0;

    for (int i = N / 4; i < 3 * N / 4; i++) {
        for (int j = N / 4; j < 3 * N / 4; j++) {
            double exact = analytical(i * h, j * h, xc, yc, t_final,
                                      ALPHA, A0, sigma0);
            double err = fabs(h_u[idx(i, j, N)] - exact);
            if (err > max_abs_err) max_abs_err = err;
            l2_err += err * err;
        }
    }
    l2_err = sqrt(l2_err / ((3 * N / 4 - N / 4) * (3 * N / 4 - N / 4)));

    printf("\n--- Validation (interior region only) ---\n");
    printf("Simulated time: %.6f\n", t_final);
    printf("Max abs error: %.6e\n", max_abs_err);
    printf("L2 error: %.6e\n", l2_err);

    cudaFree(d_old);
    cudaFree(d_new);
    free(h_u);
    return 0;
}
