#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iomanip>
#include <fstream>

#include "Utilities.cuh"
#include "Matlab_like.cuh"
#include "InputOutput.cuh"

#define PI_f			3.14159265358979323846  

#define BLOCKSIZE		256

#define DEBUG

/**********************************/
/* ADVECTIVE PROPAGATING FUNCTION */
/**********************************/
__host__ __device__  float propagatingFunction(const float x) { return exp(-x * x / (2.f * (PI_f / 4.f) * (PI_f / 4.f))); }

/**********************************/
/* SET INITIAL CONDITIONS KERNELS */
/**********************************/
__global__ void setInitialConditionsKernel0(float * __restrict__ d_u, const float * __restrict__ d_t, const float * __restrict__ d_x, const float v, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N + 1) return;

	d_u[tid] = propagatingFunction(d_x[tid] - v * d_t[0]);     // --- Initial condition

}

__global__ void setInitialConditionsKernel1(float * __restrict__ d_u, const float * __restrict__ d_t, const float * __restrict__ d_x, const float v, const float alpha, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid > N + 1) return;

	// --- Matsuno initial condition
	if (tid == 0) d_u[N + 1] = propagatingFunction(d_x[tid] - v * d_t[1]);			// --- Enforcing boundary condition (left boundary)
	else d_u[tid + N + 1] = d_u[tid] - 0.5 * alpha * (d_u[tid + 1] - d_u[tid - 1]);

}

/*****************/
/* UPDATE KERNEL */
/*****************/
// --- Leapfrog scheme
__global__ void updateKernel(float * __restrict__ d_u, const float * __restrict__ d_t, const float * __restrict__ d_x, const float v,
	const float alpha, const float Q, const int l, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N + 1) return;

	if ((tid > 0) && (tid < N))

		// --- Update equation
		d_u[tid + (l + 1) * (N + 1)] = d_u[tid + (l - 1) * (N + 1)] - alpha * (d_u[tid + 1 + l * (N + 1)] - d_u[tid - 1 + l * (N + 1)]);

	else if (tid == 0) {

		// --- Enforcing boundary condition (left boundary)
		d_u[tid + (l + 1) * (N + 1)] = propagatingFunction(d_x[tid] - v * d_t[l + 1]);

	}
	else if (tid == N) {

		d_u[tid + (l + 1) * (N + 1)] = (1.f - Q) * d_u[tid - 1 + l * (N + 1)] + Q * d_u[tid + l * (N + 1)];

	}

}

/********/
/* MAIN */
/********/
int main() {

	const float t_0 = 0.f;								// --- Initial time
	const float t_f = 15.f;								// --- Final time
	const float x_0 = 0.f;
	const float x_f = 2.f * PI_f;
	const int	M = 200;								// --- Number of time steps
	const int	N = 165;								// --- Number of space mesh points
	const float	v = 0.5;								// --- Wave speed

	/************************/
	/* SPACE DISCRETIZATION */
	/************************/
	const float	dx = 2.f * PI_f / (float)N;			// --- Discretization step in space
	float *d_x = d_colon(x_0, dx, x_f);			// --- Discretization points

	/***********************/
	/* TIME DISCRETIZATION */
	/***********************/
	const float dt = (t_f - t_0) / (float)M;           // --- Discretization time
	float *d_t = d_colon(t_0, dt, t_f);			// --- Discretization points

	const float alpha = v * dt / dx;					// --- Courant number

	/**************************************/
	/* DEFINE AND INITIALIZE THE SOLUTION */
	/**************************************/
	// --- u(u, t); First row is for initial condition, first column is for boundary condition
	float *d_u;		gpuErrchk(cudaMalloc((void**)&d_u, (N + 1) * (M + 1) * sizeof(float)));
	gpuErrchk(cudaMemset(d_u, 0, (N + 1) * (M + 1) * sizeof(float)));

	setInitialConditionsKernel0 << <iDivUp(N + 1, BLOCKSIZE), BLOCKSIZE >> >(d_u, d_t, d_x, v, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	setInitialConditionsKernel1 << <iDivUp(N + 1, BLOCKSIZE), BLOCKSIZE >> >(d_u, d_t, d_x, v, alpha, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	const float Q = (1.f - alpha) / (1.f + alpha);
	for (int l = 1; l < M - 1; l++) {			// --- Time steps

		updateKernel << <iDivUp(N + 1, BLOCKSIZE), BLOCKSIZE >> >(d_u, d_t, d_x, v, alpha, Q, l, N);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

	saveGPUrealtxt(d_u, "D:\\MDC2\\Advection-Equation\\Matlab\\d_u.txt", (M + 1) * (N + 1));

	return 0;
}
