/* 1D FDTD acoustic wave simulation */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Utilities.cuh"
#include "Matlab_like.cuh"
#include "InputOutput.cuh"

#define BLOCKSIZE		256

#define DEBUG

#define REALTYPE		1			// --- 0 for float and 1 for double

#if (REALTYPE == 1)
#define REAL			double
#define PI				3.1415926535897932384626433832795028841971693993751058209749445923078164062
#else	
#define REAL			float 
#define PI				3.14159265358979323846  			
#endif

/*******************************************/
/* PROPAGATING FUNCTION AND ITS DERIVATIVE */
/*******************************************/
template<class T>
__host__ __device__  T propagatingFunctionStationary(const T x, const T x1, const T x2) { return cos((3 * PI / (x2 - x1)) * x); }

template<class T>
__host__ __device__  T propagatingFunctionStationaryDerivative(const T x, const T x1, const T x2) { return -(3 * PI / (x2 - x1)) * sin((3 * PI / (x2 - x1)) * x); }

/**********************************/
/* SET INITIAL CONDITIONS KERNELS */
/**********************************/
template<class T>
__global__ void setStep0Kernel(T * __restrict__ d_u1, T * __restrict__ d_u, const T * __restrict__ d_t, const T * __restrict__ d_x, const T v, const T x1, const T x2, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N + 1) return;

	d_u1[tid] = ((T)0.5) * (propagatingFunctionStationary(d_x[tid] - x1 - v * d_t[0], x1, x2) -
		propagatingFunctionStationary(d_x[tid] - x1 + v * d_t[0], x1, x2));   // --- It includes also the two boundary conditions
	d_u[tid] = d_u1[tid];

}

template<class T>
__global__ void setStep1Kernel(T * __restrict__ d_u1, T * __restrict__ d_u2, T * __restrict__ d_u, const T * __restrict__ d_t, const T * __restrict__ d_x, const T v, const T x1, const T x2, const T alpha, const T dt,
	const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N + 1) return;

	// --- Left boundary condition
	if (tid == 0) 	{
		d_u2[0] = ((T)0.5) * (propagatingFunctionStationary(-v * d_t[1], x1, x2) - propagatingFunctionStationary(v * d_t[1], x1, x2));
		d_u[N + 1] = d_u2[0];
		return;
	}
	// --- Right boundary condition
	if (tid == N) {
		d_u2[N] = ((T)0.5) * (propagatingFunctionStationary(x2 - x1 - v * d_t[1], x1, x2) - propagatingFunctionStationary(x2 - x1 + v * d_t[1], x1, x2));
		d_u[2 * N + 1] = d_u2[N];
		return;
	}
	d_u2[tid] = alpha * alpha * d_u1[tid + 1] / ((T)2) + ((T)1 - alpha * alpha) * d_u1[tid]
		+ alpha * alpha * d_u1[tid - 1] / ((T)2) - v * dt * ((T)0.5) * (
		propagatingFunctionStationaryDerivative(d_x[tid] - x1 - v * d_t[0], x1, x2) +
		propagatingFunctionStationaryDerivative(d_x[tid] - x1 + v * d_t[0], x1, x2));

	d_u[tid + N + 1] = d_u2[tid];

}

/************************************/
/* UPDATE KERNEL - NO SHARED MEMORY */
/************************************/
// --- This kernel function will not work since the update of d_u3[tid] relies on d_u2[tid + 1] and d_u2[tid - 1], which are subsequently updated. The updates of d_u3[tid], d_u2[tid + 1] and d_u2[tid - 1] can be deal with 
//     by different threads, so d_u3 can be updated according to already (and wrongly) updated values of d_u2[tid + 1] and d_u2[tid - 1].
template<class T>
__global__ void updateKernelNotSharedNotWorking(T * __restrict__ d_u1, T * __restrict__ d_u2, T * __restrict__ d_u3, T * __restrict__ d_u, const T * __restrict__ d_t, const T * __restrict__ d_x, const T v, const T x1,
	const T x2, const T alpha, const int l, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N + 1) return;

	// --- Left boundary condition
	if (tid == 0) {
		d_u3[0] = ((T)0.5) * (propagatingFunctionStationary(-v * d_t[l + 1], x1, x2) - propagatingFunctionStationary(v * d_t[l + 1], x1, x2));
		d_u[(l + 1) * (N + 1)] = d_u3[0];
		d_u1[0] = d_u2[0];
		d_u2[0] = d_u3[0];
		return;
	}

	// --- Update equation
	if (tid < N) {
		d_u3[tid] = alpha * alpha * d_u2[tid + 1] + ((T)2) * ((T)1 - alpha * alpha) * d_u2[tid] + alpha * alpha * d_u2[tid - 1] - d_u1[tid];
		d_u[(l + 1) * (N + 1) + tid] = d_u3[tid];
		d_u1[tid] = d_u2[tid];
		d_u2[tid] = d_u3[tid];
		return;
	}

	// --- Right boundary condition
	d_u3[N] = ((T)0.5) * (propagatingFunctionStationary(x2 - x1 - v * d_t[l + 1], x1, x2) - propagatingFunctionStationary(x2 - x1 + v * d_t[l + 1], x1, x2));
	d_u[(l + 1) * (N + 1) + N] = d_u3[N];
	d_u1[N] = d_u2[N];
	d_u2[N] = d_u3[N];

}

template<class T>
__global__ void updateKernelNotShared(T * __restrict__ d_u1, T * __restrict__ d_u2, T * __restrict__ d_u3, T * __restrict__ d_u, const T * __restrict__ d_t, const T * __restrict__ d_x, const T v, const T x1,
	const T x2, const T alpha, const int l, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N + 1) return;

	// --- Left boundary condition
	if (tid == 0) {
		d_u3[0] = ((T)0.5) * (propagatingFunctionStationary(-v * d_t[l + 1], x1, x2) - propagatingFunctionStationary(v * d_t[l + 1], x1, x2));
		d_u[(l + 1) * (N + 1)] = d_u3[0];
		return;
	}

	// --- Update equation
	if (tid < N) {
		d_u3[tid] = alpha * alpha * d_u2[tid + 1] + ((T)2) * ((T)1 - alpha * alpha) * d_u2[tid] + alpha * alpha * d_u2[tid - 1] - d_u1[tid];
		d_u[(l + 1) * (N + 1) + tid] = d_u3[tid];
		return;
	}

	// --- Right boundary condition
	d_u3[N] = ((T)0.5) * (propagatingFunctionStationary(x2 - x1 - v * d_t[l + 1], x1, x2) - propagatingFunctionStationary(x2 - x1 + v * d_t[l + 1], x1, x2));
	d_u[(l + 1) * (N + 1) + N] = d_u3[N];

}

/*********************************/
/* UPDATE KERNEL - SHARED MEMORY */
/*********************************/
template<class T>
__global__ void updateKernelShared(T * __restrict__ d_u1, T * __restrict__ d_u2, T * __restrict__ d_u3, T * __restrict__ d_u, const T * __restrict__ d_t, const T * __restrict__ d_x, const T v, const T x1,
	const T x2, const T alpha, const int l, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N + 1) return;

	__shared__ T d_u2_temp[BLOCKSIZE + 2];

	d_u2_temp[threadIdx.x + 1] = d_u2[tid];

	if ((threadIdx.x == 0) && ((tid >= 1))) { d_u2_temp[0] = d_u2[tid - 1]; }										// --- Left halo region

	if ((threadIdx.x == 0) && ((tid + BLOCKSIZE) < (N + 1))) { d_u2_temp[BLOCKSIZE + 1] = d_u[tid + BLOCKSIZE]; }	// --- Right halo region

	__syncthreads();

	// --- Left boundary condition
	if (tid == 0) {
		d_u3[0] = ((T)0.5) * (propagatingFunctionStationary(-v * d_t[l + 1], x1, x2) - propagatingFunctionStationary(v * d_t[l + 1], x1, x2));
		d_u[(l + 1) * (N + 1)] = d_u3[0];
		return;
	}

	// --- Update equation
	if (tid < N) {
		d_u3[tid] = alpha * alpha * d_u2_temp[threadIdx.x + 2] + ((T)2) * ((T)1 - alpha * alpha) * d_u2_temp[threadIdx.x + 1] + alpha * alpha * d_u2_temp[threadIdx.x] - d_u1[tid];
		d_u[(l + 1) * (N + 1) + tid] = d_u3[tid];
		return;
	}

	// --- Right boundary condition
	d_u3[N] = ((T)0.5) * (propagatingFunctionStationary(x2 - x1 - v * d_t[l + 1], x1, x2) - propagatingFunctionStationary(x2 - x1 + v * d_t[l + 1], x1, x2));
	d_u[(l + 1) * (N + 1) + N] = d_u3[N];

}

/********/
/* MAIN */
/********/
int main() {

	const REAL t_0 = 0.;								// --- Initial time
	const REAL t_f = 15.;								// --- Final time
	const REAL x_0 = 0.;								// --- Left boundary of the computational domain
	const REAL x_f = 2. * PI;							// --- Right boundary of the computational domain
	const int	M = 200;								// --- Number of time steps
	const int	N = 100;								// --- Number of space mesh points
	const REAL	v = 0.5;								// --- Wave speed

	/************************/
	/* SPACE DISCRETIZATION */
	/************************/
	const REAL	dx = 2. * PI / (REAL)N;					// --- Discretization step in space
	REAL *d_x = d_colon(x_0, dx, x_f);					// --- Discretization points

	/***********************/
	/* TIME DISCRETIZATION */
	/***********************/
	const REAL dt = (t_f - t_0) / (REAL)M;           // --- Discretization time
	REAL *d_t = d_colon(t_0, dt, t_f);					// --- Discretization points

	const REAL alpha = v * dt / dx;					// --- Courant number

	/********************************************************************/
	/* DEFINE AND INITIALIZE THE SOLUTION AND OLDER SOLUTIONS u1 AND u2 */
	/********************************************************************/
	// --- u(u, t); First row is for initial condition, first column is for boundary condition
	REAL *d_u;		gpuErrchk(cudaMalloc((void**)&d_u, (N + 1) * (M + 1) * sizeof(REAL)));
	gpuErrchk(cudaMemset(d_u, 0, (N + 1) * (M + 1) * sizeof(REAL)));

	REAL *d_u1;		gpuErrchk(cudaMalloc((void**)&d_u1, (N + 1) * sizeof(REAL)));
	REAL *d_u2;		gpuErrchk(cudaMalloc((void**)&d_u2, (N + 1) * sizeof(REAL)));
	REAL *d_u3;		gpuErrchk(cudaMalloc((void**)&d_u3, (N + 1) * sizeof(REAL)));
	gpuErrchk(cudaMemset(d_u1, 0, (N + 1) * sizeof(REAL)));
	gpuErrchk(cudaMemset(d_u2, 0, (N + 1) * sizeof(REAL)));
	gpuErrchk(cudaMemset(d_u3, 0, (N + 1) * sizeof(REAL)));

	setStep0Kernel << <iDivUp(N + 1, BLOCKSIZE), BLOCKSIZE >> >(d_u1, d_u, d_t, d_x, v, x_0, x_f, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	setStep1Kernel << <iDivUp(N + 1, BLOCKSIZE), BLOCKSIZE >> >(d_u1, d_u2, d_u, d_t, d_x, v, x_0, x_f, alpha, dt, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	/**************/
	/* ITERATIONS */
	/**************/
	for (int l = 1; l < M; l++) {			// --- Time steps

		//updateKernelNotSharedNotWorking << <iDivUp(N + 1, BLOCKSIZE), BLOCKSIZE >> >(d_u1, d_u2, d_u3, d_u, d_t, d_x, v, x_0, x_f, alpha, l, N);
		//updateKernelNotShared << <iDivUp(N + 1, BLOCKSIZE), BLOCKSIZE >> >(d_u1, d_u2, d_u3, d_u, d_t, d_x, v, x_0, x_f, alpha, l, N);
		updateKernelShared << <iDivUp(N + 1, BLOCKSIZE), BLOCKSIZE >> >(d_u1, d_u2, d_u3, d_u, d_t, d_x, v, x_0, x_f, alpha, l, N);
#ifdef DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
		gpuErrchk(cudaMemcpy(d_u1, d_u2, (N + 1) * sizeof(REAL), cudaMemcpyDeviceToDevice));
		gpuErrchk(cudaMemcpy(d_u2, d_u3, (N + 1) * sizeof(REAL), cudaMemcpyDeviceToDevice));

	}

	saveGPUrealtxt(d_u, "D:\\PDEs\\Wave-Equation\\1D\\Matlab\\d_u.txt", (M + 1) * (N + 1));

	return 0;
}
