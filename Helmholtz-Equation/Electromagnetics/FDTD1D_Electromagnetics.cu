/* 1D FDTD simulation with an additive source. */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Utilities.cuh"
#include "TimingCPU.h"
#include "TimingGPU.cuh"

#define BLOCKSIZE	512
//#define DEBUG

/***********************************/
/* HOST-SIZE FIELD UPDATE FUNCTION */
/***********************************/
void updateHost(double *h_ez, double* h_hy, double imp0, double qTime, const int source, const int N) {

	/* update magnetic field */
	for (int mm = 0; mm < N - 1; mm++)
		h_hy[mm] = h_hy[mm] + (h_ez[mm + 1] - h_ez[mm]) / imp0;
	
	/* update electric field */
	for (int mm = 1; mm < N; mm++)
		h_ez[mm] = h_ez[mm] + (h_hy[mm] - h_hy[mm - 1]) * imp0;

	/* use additive source at node 50 */
	h_ez[source] += exp(-(qTime - 30.) * (qTime - 30.) / 100.);

}

/********************************************************/
/* DEVICE-SIZE FIELD UPDATE FUNCTION - NO SHARED MEMORY */
/********************************************************/
__global__ void updateDevice_v0(double *d_ez, double* d_hy, double imp0, double qTime, const int source, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	/* update magnetic field */
	if (tid < N-1) d_hy[tid] = d_hy[tid] + (d_ez[tid + 1] - d_ez[tid]) / imp0;

	__threadfence();

	/* update electric field */
	if ((tid < N)&&(tid > 0)) d_ez[tid] = d_ez[tid] + (d_hy[tid] - d_hy[tid - 1]) * imp0;

	/* use additive source at node 50 */
	if (tid == source) d_ez[tid] += exp(-(qTime - 30.) * (qTime - 30.) / 100.);

}

/**************************************************************/
/* DEVICE-SIZE MAGNETIC FIELD UPDATE FUNCTION - SHARED MEMORY */
/**************************************************************/
__global__ void updateDevice_hy(double *d_ez, double* d_hy, double imp0, double qTime, const int source, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	__shared__ double hy_temp[BLOCKSIZE+1], ez_temp[BLOCKSIZE+1];

	hy_temp[threadIdx.x] = d_hy[tid];
	ez_temp[threadIdx.x] = d_ez[tid];
	
	if ((threadIdx.x == 0)&&((tid + BLOCKSIZE) < N)) {
		ez_temp[BLOCKSIZE] = d_ez[tid + BLOCKSIZE];
		hy_temp[BLOCKSIZE] = d_hy[tid + BLOCKSIZE];
	}

	__syncthreads();

	/* update magnetic field */
	if (tid < N-1) d_hy[tid] = hy_temp[threadIdx.x] + (ez_temp[threadIdx.x + 1] - ez_temp[threadIdx.x]) / imp0;

}

/**************************************************************/
/* DEVICE-SIZE ELECTRIC FIELD UPDATE FUNCTION - SHARED MEMORY */
/**************************************************************/
__global__ void updateDevice_ez(double *d_ez, double* d_hy, double imp0, double qTime, const int source, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	__shared__ double hy_temp[BLOCKSIZE+1], ez_temp[BLOCKSIZE+1];

	hy_temp[threadIdx.x + 1] = d_hy[tid];
	ez_temp[threadIdx.x + 1] = d_ez[tid];
	
	if ((threadIdx.x == 0)&&(tid >= 1)) {
		ez_temp[0] = d_ez[tid - 1];
		hy_temp[0] = d_hy[tid - 1];
	}

	__syncthreads();

	/* update electric field */
	ez_temp[threadIdx.x] = ez_temp[threadIdx.x + 1] + (hy_temp[threadIdx.x + 1] - hy_temp[threadIdx.x]) * imp0;

	/* use additive source at node 50 */
	if (tid == source) ez_temp[threadIdx.x] += exp(-(qTime - 30.) * (qTime - 30.) / 100.);

	if ((tid < N)&&(tid > 0)) d_ez[tid] = ez_temp[threadIdx.x];

}

/********/
/* MAIN */
/********/
int main() {

	// --- Problem size
	const int SIZE = 10000000;

	// --- Free-space wave impedance
	double imp0 = 377.0;

	// --- Maximum number of iterations (must be less than the problem size)
	int maxTime = 100;

	// --- Source location
	int source = SIZE / 2;
	
	// --- Host side memory allocations and initializations
	double *h_ez = (double*)calloc(SIZE, sizeof(double));
	double *h_hy = (double*)calloc(SIZE, sizeof(double));

	// --- Device side memory allocations and initializations
	double *d_ez; gpuErrchk(cudaMalloc((void**)&d_ez, SIZE * sizeof(double)));
	double *d_hy; gpuErrchk(cudaMalloc((void**)&d_hy, SIZE * sizeof(double)));
	gpuErrchk(cudaMemset(d_ez, 0, SIZE * sizeof(double)));
	gpuErrchk(cudaMemset(d_hy, 0, SIZE * sizeof(double)));
	
	// --- Host side memory allocations for debugging purposes
#ifdef DEBUG
	double *h_ez_temp = (double*)calloc(SIZE, sizeof(double));
	double *h_hy_temp = (double*)calloc(SIZE, sizeof(double));
#endif

	// --- Host-side time-steppings
#ifndef DEBUG
	TimingCPU timerCPU;
	timerCPU.StartCounter();
	for (int qTime = 0; qTime < maxTime; qTime++) {
		updateHost(h_ez, h_hy, imp0, qTime, source, SIZE);
	}
	printf("CPU elapsed time = %3.3f ms\n", timerCPU.GetCounter());
#endif

	TimingGPU timerGPU;
	timerGPU.StartCounter();
	// --- Device-side time-steppings
	for (int qTime = 0; qTime < maxTime; qTime++) {

		updateDevice_v0<<<iDivUp(SIZE, BLOCKSIZE), BLOCKSIZE>>>(d_ez, d_hy, imp0, qTime, source, SIZE);
		//updateDevice_hy<<<iDivUp(SIZE, BLOCKSIZE), BLOCKSIZE>>>(d_ez, d_hy, imp0, qTime, source, SIZE);
		//updateDevice_ez<<<iDivUp(SIZE, BLOCKSIZE), BLOCKSIZE>>>(d_ez, d_hy, imp0, qTime, source, SIZE);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaMemcpy(h_ez_temp, d_ez, SIZE * sizeof(double), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(h_hy_temp, d_hy, SIZE * sizeof(double), cudaMemcpyDeviceToHost));

		updateHost(h_ez, h_hy, imp0, qTime, source, SIZE);
		for (int i=0; i<SIZE; i++) {
			printf("%f %f %f %f\n",h_ez_temp[i], h_ez[i], h_hy_temp[i], h_hy[i]);
		}
		printf("\n");
#endif
	}
	printf("GPU elapsed time = %3.3f ms\n", timerGPU.GetCounter());

	return 0;
}
