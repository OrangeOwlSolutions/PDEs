#include <stdio.h>

#include <cusparse_v2.h>

#include "Utilities.cuh"
#include "InputOutput.cuh"

#define pi	3.141592653589793238463

#define BLOCKSIZE	256

#define DEBUG

/****************************************/
/* IMPLICIT EULER INITIALIZATION KERNEL */
/****************************************/
__global__ void heat1DImplicitEulerInit(float * __restrict__ d_x, float * __restrict__ d_Ad, float * __restrict__ d_Ald, float * __restrict__ d_Aud, 
	                                    float * __restrict__ d_T, const float dx, const float F, const int N) {

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N) {
		
		// --- Discretization of the computational domain
		d_x[tid] = dx * tid;

		// --- Filling the matrix diagonal
		d_Ad[tid]  = 1.f + 2.f * F;
		
		// --- Filling the matrix lower diagonal. The cuSPARSE convention is that the first element is zero.
		if (tid > 0)		d_Ald[tid]		= -F;
		else				d_Ald[tid]		=  0.f;
		
		// --- Filling the matrix upper diagonal. The cuSPARSE convention is that the last element is zero.
		if (tid < N - 1)	d_Aud[tid]		= -F;
		else				d_Aud[N - 1]	=  0;

		// --- Initial temperature profile
		d_T[tid] = (sin(pi * d_x[tid]) + 0.1f * sin(100.f * pi * d_x[tid]));

	}

}

/********/
/* MAIN */
/********/
int main() {

	// --- Algorithm parameters
	const int maxIterNumber     = 100;                          // --- Number of overall time steps
	const int N				    = 512;                          // --- Number of grid points

	const float k               = 0.19f;                        // --- Thermal conductivity [W / (m * K)]
	const float rho             = 930.f;                        // --- Density [kg / m^3]
	const float cp              = 1340.f;                       // --- Specific heat capacity [J / (kg * K)]
	const float alpha           = k / (rho * cp);               // --- Thermal diffusivity [m^2 / s]
	const float len             = 1.f;                          // --- Total len of the domain [m]
	const float dx              = len / (float)(N - 1);         // --- Discretization step [m]
	const float dt              = dx * dx / (4. * alpha);       // --- Time step [s]
	const float T0              = 0.f;                          // --- Temperature at the first end of the domain [C]
	const float T_N_1           = 0.f;                          // --- Temperature at the second end of the domain [C]
	const float F               = alpha * dt / (dx * dx);       // --- Mesh Fourier number

	// --- Initialize cuSPARSE
	cusparseHandle_t handle;	cusparseSafeCall(cusparseCreate(&handle));

	// --- Lower diagonal, diagonal and upper diagonal of the system matrix
	float *d_Ald;	gpuErrchk(cudaMalloc(&d_Ald, N * sizeof(float)));
	float *d_Ad;	gpuErrchk(cudaMalloc(&d_Ad,  N * sizeof(float)));
	float *d_Aud;	gpuErrchk(cudaMalloc(&d_Aud, N * sizeof(float)));
	
	// --- Domain discretization
	float *d_x;	gpuErrchk(cudaMalloc(&d_x, N * sizeof(float)));

	// --- Temperature profile (host and device)
	float *h_T = (float *)malloc(N * sizeof(float));
	float *d_T;	gpuErrchk(cudaMalloc(&d_T, N * sizeof(float)));

	heat1DImplicitEulerInit<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_x, d_Ad, d_Ald, d_Aud, d_T, dx, F, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	for (int t = 0; t < maxIterNumber; t++) {

		// --- Enforcing the boundary conditions
		gpuErrchk(cudaMemcpy(&d_T[0],	  &T0,    sizeof(float), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(&d_T[N - 1], &T_N_1, sizeof(float), cudaMemcpyHostToDevice));

		cusparseSafeCall(cusparseSgtsv(handle, N, 1, d_Ald, d_Ad, d_Aud, d_T, N));
	}

	// --- Showing the result
	gpuErrchk(cudaMemcpy(h_T, d_T, N  *sizeof(float), cudaMemcpyDeviceToHost));
	for (int k = 0; k < N; k++) printf("h_T[%i] = %f\n", k, h_T[k]);

	saveGPUrealtxt(d_T, "C:\\Users\\angelo\\Documents\\CEM\\ParticleSwarm\\ParticleSwarmSynthesis\\ParticleSwarmSynthesis\\d_T.txt", N);

	return 0;
}
