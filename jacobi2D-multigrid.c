/* Multigrid for solving -u''=f for x in (0,1)*(0,1)
 * Usage: ./jacobi2D-multigrid < Nfine > < iter > [s-steps]
 * NFINE: number of intervals on finest level, must be power of 2
 * ITER: max number of V-cycle iterations
 * S-STEPS: number of Jacobi smoothing steps; optional
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "util.h"

#define id(i,j,Nloc) (i)*(Nloc+1)+j

/* compuate norm of residual */
double compute_norm(double *u, int N)
{
    int i, j;
    double norm = 0.0;
    for (i = 0; i <= N; i++)
        for (j = 0; j <= N; j++)
            norm += u[id(i,j,N)] * u[id(i,j,N)];
    return sqrt(norm);
}

/* set vector to zero */
void set_zero(double *u, int N)
{
    int i ,j;
    for (i = 0; i <= N; i++)
        for (j = 0; j <= N; j++)
            u[id(i,j,N)] = 0.0;
}

/* debug function */
void output_to_screen(double *u, int N)
{
    int i, j;
    for (i = 0; i <= N; i++) {
        for (j = 0; j <= N; j++)
            printf("%f\t", u[id(i,j,N)]);
        printf("\n");
    }
    printf("\n");
}

/* coarsen uf from dimension (N+1)*(N+1) to dimension (N/2+1)*(N/2+1)
 assuming N = 2^l
 */
void coarsen(double *uf, double *uc, int N)
{
    int ic, jc;
    for (ic = 1; ic < N/2; ic++)
        for (jc = 1; jc < N/2; jc++)
            uc[id(ic,jc,N/2)] = 0.25 * uf[id(2*ic,2*jc,N)] + 0.125 * (uf[id(2*ic-1,2*jc,N)] + uf[id(2*ic+1,2*jc,N)] + uf[id(2*ic,2*jc-1,N)] + uf[id(2*ic,2*jc+1,N)]) + 0.0625 * (uf[id(2*ic-1,2*jc-1,N)] + uf[id(2*ic-1,2*jc+1,N)] + uf[id(2*ic+1,2*jc-1,N)] + uf[id(2*ic+1,2*jc+1,N)]);
}

/* refine u from dimension (N+1)*(N+1) to dimension (2*N+1)*(2*N+1)
 assuming N = 2^l, and add to existing uf
 */
void refine_and_add(double *u, double *uf, int N)
{
    int i, j;
    uf[id(1,1,2*N)] += 0.25 * (u[id(0,0,N)] + u[id(0,1,N)] + u[id(1,0,N)] + u[id(1,1,N)]);
    for (j = 1; j < N; j++) {
        uf[id(1,2*j,2*N)] += 0.5 * (u[id(0,j,N)] + u[id(1,j,N)]);
        uf[id(1,2*j+1,2*N)] += 0.25 * (u[id(0,j,N)] + u[id(0,j+1,N)] + u[id(1,j,N)] + u[id(1,j+1,N)]);
    }
    for (i = 1; i < N; i++) {
        uf[id(2*i,1,2*N)] += 0.5 * (u[id(i,0,N)] + u[id(i,1,N)]);
        uf[id(2*i+1,1,2*N)] += 0.25 * (u[id(i,0,N)] + u[id(i,1,N)] + u[id(i+1,0,N)] + u[id(i+1,1,N)]);
        for (j = 1; j < N; j++) {
            uf[id(2*i,2*j,2*N)] += u[id(i,j,N)];
            uf[id(2*i,2*j+1,2*N)] += 0.5 * (u[id(i,j,N)] + u[id(i,j+1,N)]);
            uf[id(2*i+1,2*j,2*N)] += 0.5 * (u[id(i,j,N)] + u[id(i+1,j,N)]);
            uf[id(2*i+1,2*j+1,2*N)] += 0.25 * (u[id(i,j,N)] + u[id(i,j+1,N)] + u[id(i+1,j,N)] + u[id(i+1,j+1,N)]);
        }
    }
}

/* compute residual vector */
void compute_residual(double *u, double *rhs, double *res, int N, double h)
{
    int i, j;
    for (i = 1; i < N; i++)
        for (j = 1; j < N; j++)
            res[id(i,j,N)] = (4 * u[id(i,j,N)] - u[id(i-1,j,N)] - u[id(i,j-1,N)] - u[id(i+1,j,N)] - u[id(i,j+1,N)]) / (h * h) - rhs[id(i,j,N)];
}

/* compute residual and coarsen */
void compute_and_coarsen_residual(double *u, double *rhs, double *resc,
                                  int N, double h)
{
    double * resf = calloc((N+1)*(N+1), sizeof(double));
    compute_residual(u, rhs, resf, N, h);
    coarsen(resf, resc, N);
    free(resf);
}

/* perform Jacobi iterations on u */
void jacobi(double *u, double *rhs, int N, double h, int ssteps)
{
    int i, j, k;
    /* Jacobi damping parameter -- plays an important role in MG */
    double omega = 2./3;
    double * unew = calloc((N+1)*(N+1), sizeof(double));
    for (k = 0; k < ssteps; k++) {
        for (i = 1; i < N; i++)
            for (j = 1; j < N; j++)
                unew[id(i,j,N)] = u[id(i,j,N)] + omega * 0.25 * (h * h * rhs[id(i,j,N)] + u[id(i-1,j,N)] + u[id(i,j-1,N)] + u[id(i+1,j,N)] + u[id(i,j+1,N)] - 4 * u[id(i,j,N)]);
        memcpy(u, unew, (N+1)*(N+1)*sizeof(double));
    }
    free(unew);
}


int main(int argc, char * argv[])
{
    int i, j, Nfine, l, iter, max_iters, levels, ssteps;
    double res_norm, res0_norm, tol = 1e-6;
    
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: ./multigrid_1d Nfine max_iters [s-steps]\n");
        fprintf(stderr, "Nfine: # of intervals, must be power of two number\n");
        fprintf(stderr, "s-steps: # jacobi smoothing steps (optional, default is 3)\n");
        abort();
    }
    sscanf(argv[1], "%d", &Nfine);
    sscanf(argv[2], "%d", &max_iters);
    ssteps = 3 * pow(5, floor(log2(Nfine))-2);
    if (argc > 3)
        sscanf(argv[3], "%d", &ssteps);
    
    /* compute number of multigrid levels */
    levels = floor(log2(Nfine));
    printf("Multigrid Solve using V-cycles for -u'' = f on (0,1)*(0,1)\n");
    printf("Number of intervals = %d, max_iters = %d, s-steps = %d\n", Nfine, max_iters, ssteps);
    printf("Number of MG levels: %d \n", levels);
    
    /* allocation of vectors, including boundary ghost points */
    double *u[levels], *rhs[levels];
    /* N, h on each level */
    int *N = calloc(levels, sizeof(int));
    double *h = calloc(levels, sizeof(double));
    double *res = calloc((Nfine+1)*(Nfine+1), sizeof(double));
    for (l = 0; l < levels; l++) {
        N[l] = Nfine / (int) pow(2,l);
        h[l] = 1./ N[l];
        printf("MG level %2d, N = %8d\n", l, N[l]);
        u[l] = (double *)calloc((N[l]+1)*(N[l]+1), sizeof(double));
        rhs[l] = (double *)calloc((N[l]+1)*(N[l]+1), sizeof(double));
    }
    /* rhs on finest mesh */
    for (i = 0; i <= N[0]; i++)
        for (j = 0; j <= N[0]; j++)
            rhs[0][id(i,j,N[0])] = 1.0;
    
    /* compute initial residual norm */
    compute_residual(u[0], rhs[0], res, N[0], h[0]);
    res_norm = res0_norm = compute_norm(res, N[0]);
    printf("Initial Residual: %f\n", res0_norm);
    
    /* timing */
    timestamp_type time1, time2;
    get_timestamp(&time1);
    
    for (iter = 0; iter < max_iters && res_norm/res0_norm > tol; iter++) {
        
        /* V-cycle: Coarsening */
        for (l = 0; l < levels-1; l++) {
            /* pre-smoothing and coarsen */
            jacobi(u[l], rhs[l], N[l], h[l], ssteps);
            compute_and_coarsen_residual(u[l], rhs[l], rhs[l+1], N[l], h[l]);
            /* initialize correction for solution with zero */
            set_zero(u[l+1], N[l+1]);
        }
        /* V-cycle: Solve on coarsest grid using many smoothing steps */
        jacobi(u[levels-1], rhs[levels-1], N[levels-1], h[levels-1], 1000);
        
        /* V-cycle: Refine and correct */
        for (l = levels-1; l > 0; l--) {
            /* refine and add to u */
            refine_and_add(u[l], u[l-1], N[l]);
            /* post-smoothing steps */
            jacobi(u[l-1], rhs[l-1], N[l-1], h[l-1], ssteps);
        }
        
        /* compute residual for each iteration */
        compute_residual(u[0], rhs[0], res, N[0], h[0]);
        res_norm = compute_norm(res, N[0]);
        printf("[Iter %d] Residual norm: %2.8f\n", iter, res_norm);
    }
    
    /* timing */
    get_timestamp(&time2);
    double elapsed = timestamp_diff_in_seconds(time1, time2);
    printf("Time elapsed is %f seconds.\n", elapsed);
    
    /* clean up */
    free(h);
    free(N);
    free(res);
    for (l = levels-1; l >= 0; l--) {
        free(u[l]);
        free(rhs[l]);
    }
    
    return 0;
}
