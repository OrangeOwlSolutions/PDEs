/* 1D FDTD acoustic wave simulation */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "InputOutput.h"

#include "Utilities.cuh"
#include "Matlab_like.cuh"
#include "TimingCPU.h"
#include "TimingGPU.cuh"

#define BLOCKSIZE	512
//#define DEBUG

#define pi 3.141592653589793115997963468544185161590576171875

/***********************************/
/* HOST-SIZE FIELD UPDATE FUNCTION */
/***********************************/
void updateHost(const double * __restrict__ h_uold, const double * __restrict__ h_u, double * __restrict__ h_unew, const double c, const int N) {

	for (int nx = 1; nx < N - 1; nx++) {
		double u1 = 2. * h_u[nx] - h_uold[nx];
		double u2 = h_u[nx - 1] - 2. * h_u[nx] + h_u[nx + 1];
		h_unew[nx] = u1 + c * c * u2;
	}

}

/********************************************************/
/* DEVICE-SIZE FIELD UPDATE FUNCTION - NO SHARED MEMORY */
/********************************************************/
__global__ void updateDevice_v0(const double * __restrict__ d_uold, const double * __restrict__ d_u, double * __restrict__ d_unew, const double c,
	                            const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if ((tid >= N - 1) || (tid == 0)) return;
	
	const double u1 = 2. * d_u[tid] - d_uold[tid];
	const double u2 = d_u[tid - 1] - 2. * d_u[tid] + d_u[tid + 1];
	d_unew[tid] = u1 + c * c * u2;
}

/**************************************************************/
/* DEVICE-SIZE MAGNETIC FIELD UPDATE FUNCTION - SHARED MEMORY */
/**************************************************************/
__global__ void updateDevice_v1(const double * __restrict__ d_uold, const double * __restrict__ d_u, double * __restrict__ d_unew, const double c,
	const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N) return;

	__shared__ double d_u_temp[BLOCKSIZE + 2], d_uold_temp[BLOCKSIZE + 2];

	d_u_temp[threadIdx.x + 1] = d_u[tid];
	d_uold_temp[threadIdx.x + 1] = d_uold[tid];

	if ((threadIdx.x == 0) && ((tid >= 1))) {
		d_u_temp[0] = d_u[tid - 1];
		d_uold_temp[0] = d_uold[tid - 1];
	}

	if ((threadIdx.x == 0) && ((tid + BLOCKSIZE) < N)) {
		d_u_temp[BLOCKSIZE + 1] = d_u[tid + BLOCKSIZE];
		d_uold_temp[BLOCKSIZE + 1] = d_uold[tid + BLOCKSIZE];
	}

	__syncthreads();

	if ((tid < N - 1) && (tid > 0)) {
		const double u1 = 2. * d_u_temp[threadIdx.x + 1] - d_uold_temp[threadIdx.x + 1];
		const double u2 = d_u_temp[threadIdx.x] - 2. * d_u_temp[threadIdx.x + 1] + d_u_temp[threadIdx.x + 2];
		d_unew[tid] = u1 + c * c * u2;
	}
}

/********/
/* MAIN */
/********/
int main() {

	const int		N		= 100000;																	// --- Number of mesh points
	const double	L		= 2.5;																	// --- Length of the string
	
		  double	*h_x	= h_linspace(0., L, N);													// --- Mesh points
	const double	dx		= h_x[1] - h_x[0];                                                      // --- Mesh step
	const double	v		= 5.;                                                                   // --- Wave speed
	const double	dt		= 0.25 * dx / v;                                                        // --- Time - Step matching the Courant - Friedrichs - Lewy condition
	const int		T		= floor((3 * L / v) / dt);                                              // --- Total number of time steps
		
	double *h_uold	= (double *)calloc(N, sizeof(double));											// --- u(x, t)  - on the host - previous step
	double *h_u		= (double *)calloc(N, sizeof(double));											// --- u(x, t)  - on the host - current step
	double *h_unew	= (double *)calloc(N, sizeof(double));											// --- u(x, t)  - on the host - next step

	double *d_uold;	gpuErrchk(cudaMalloc((void**)&d_uold,	N * sizeof(double)));					// --- u(x, t)  - on the device - previous step
	double *d_u;	gpuErrchk(cudaMalloc((void**)&d_u,		N * sizeof(double)));					// --- u(x, t)  - on the device - current step
	double *d_unew;	gpuErrchk(cudaMalloc((void**)&d_unew,	N * sizeof(double)));					// --- u(x, t)  - on the device - next step
	gpuErrchk(cudaMemset(d_uold, 0, N * sizeof(double)));
	gpuErrchk(cudaMemset(d_u,	 0, N * sizeof(double)));
	gpuErrchk(cudaMemset(d_unew, 0, N * sizeof(double)));

	// --- Initial conditions
	double	kx		= 2.;                                                                           // --- Spatial frequency of source
	int		nmax	= floor(1. / (2. * kx * dx));
	for (int k = 0; k < nmax; k++) {
		h_uold[k]	= sin(2. * pi * kx * h_x[k]);
		h_u   [k]	= sin(2. * pi * kx * h_x[k]);
	}

	// --- Transfering the initial condition from host to device
	gpuErrchk(cudaMemcpy(d_uold,	h_uold,		nmax * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_u,		h_u,		nmax * sizeof(double), cudaMemcpyHostToDevice));

	/*********************/
	/* ITERATIONS - HOST */
	/*********************/
	double c = v * (dt / dx);																		// --- Courant - Friedrichs - Lewy number
	TimingCPU timerCPU;
	timerCPU.StartCounter();
	for (int nt = 2; nt < T; nt++) {
		updateHost(h_uold, h_u, h_unew, c, N);
		h_uold	= h_u;
		h_u		= h_unew;
		h_unew	= h_uold;
	}
	printf("Timing CPU = %f ms\n", timerCPU.GetCounter());

	/***********************/
	/* ITERATIONS - DEVICE */
	/***********************/
	TimingGPU timerGPU;
	timerGPU.StartCounter();
	for (int nt = 2; nt < T; nt++) {
		//updateDevice_v0 << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_uold, d_u, d_unew, c, N);
		updateDevice_v1 << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_uold, d_u, d_unew, c, N);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
		d_uold	= d_u;
		d_u		= d_unew;
		d_unew	= d_uold;
	}
	printf("Timing GPU = %f ms\n", timerGPU.GetCounter());

	/**********************/
	/* SAVING THE RESULTS */
	/**********************/
	//saveCPUrealtxt(h_u, "D:\\FDTD_1D_Acoustics\\FDTD1D_hostResult.txt", N);

	//double *h_uDevice = (double *)malloc(N * sizeof(double));
	//gpuErrchk(cudaMemcpy(h_uDevice, d_u, N * sizeof(double), cudaMemcpyDeviceToHost));
	//saveCPUrealtxt(h_uDevice, "D:\\FDTD_1D_Acoustics\\FDTD1D_deviceResult.txt", N);

	return 0;
}
