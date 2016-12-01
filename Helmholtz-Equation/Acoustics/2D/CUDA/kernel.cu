// CUDA Implementation of the 2D wave equation

#include <math.h>

#include <cuda.h>

#include "InputOutput.h"
#include "Utilities.cuh"
#include "Matlab_like.cuh"
#include "TimingGPU.cuh"

#define BLOCKSIZEX		16
#define BLOCKSIZEY		16

#define DEBUG

/***********************************/
/* HOST-SIZE FIELD UPDATE FUNCTION */
/***********************************/
void updateHost(const double * __restrict__ h_uold, const double * __restrict__ h_u, double * __restrict__ h_unew, const double alphaSquared, const int Nx, const int Ny) {

	for (int j = 1; j < Ny - 1; j++)
		for (int i = 1; i < Nx - 1; i++)
			h_unew[j * Nx + i] = 2. * h_u[j * Nx + i] - h_uold[j * Nx + i] + alphaSquared * (h_u[j * Nx + i - 1] + h_u[j * Nx + i + 1] + h_u[(j + 1) * Nx + i] + h_u[(j - 1) * Nx + i] - 4. * h_u[j * Nx + i]);

}

/********************************************************/
/* DEVICE-SIZE FIELD UPDATE FUNCTION - NO SHARED MEMORY */
/********************************************************/
__global__ void updateDevice_v0(const double * __restrict__ d_uold, const double * __restrict__ d_u, double * __restrict__ d_unew, const double alphaSquared,
								const int Nx, const int Ny) {

	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= Nx - 1) || (tidx == 0) || (tidy >= Ny - 1) || (tidy == 0)) return;

	d_unew[tidy * Nx + tidx] = 2. * d_u[tidy * Nx + tidx]  - d_uold[tidy * Nx + tidx] + alphaSquared * (d_u[tidy * Nx + tidx - 1] +
									d_u[tidy * Nx + tidx + 1] +
									d_u[(tidy + 1) * Nx + tidx] +
									d_u[(tidy - 1) * Nx + tidx] -
							   4. * d_u[tidy * Nx + tidx]);
}

/********************************************************/
/* DEVICE-SIZE FIELD UPDATE FUNCTION - SHARED MEMORY v1 */
/********************************************************/
__global__ void updateDevice_v1(const double * __restrict__ d_uold, const double * __restrict__ d_u, double * __restrict__ d_unew, const double alphaSquared,
								const int Nx, const int Ny) {

	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= Nx) || (tidy >= Ny)) return;

	__shared__ double d_u_sh[BLOCKSIZEX][BLOCKSIZEY];

	// --- Load data to shared memory. Halo regions are NOT loaded.
	d_u_sh[threadIdx.x][threadIdx.y] = d_u[tidy * Nx + tidx];
	__syncthreads();

	if ((threadIdx.x > 0) && (threadIdx.x < (BLOCKSIZEX - 1)) && (threadIdx.y > 0) && (threadIdx.y < (BLOCKSIZEY ‐ 1)) && (tidx < Nx - 1) && (tidy < Ny - 1))

		// --- If we do not need halo region elements (we are "inside" a thread block, not on the border), then use shared memory.
		d_unew[tidy * Nx + tidx] = 2. * d_u_sh[threadIdx.x][threadIdx.y] -
									    d_uold[tidy * Nx + tidx] +
						 alphaSquared * (d_u_sh[threadIdx.x - 1][threadIdx.y] +
										 d_u_sh[threadIdx.x + 1][threadIdx.y] +
										 d_u_sh[threadIdx.x][threadIdx.y - 1] +
										 d_u_sh[threadIdx.x][threadIdx.y + 1] -
									4. * d_u_sh[threadIdx.x][threadIdx.y]);

	else if (tidx > 0 && tidx < Nx - 1 && tidy > 0 && tidy < Ny - 1)  // --- Only update "interior" (not boundary) node points

		// --- If we need halo region elements, then use global memory.
		d_unew[tidy * Nx + tidx] = 2. * d_u[tidy * Nx + tidx] - d_uold[tidy * Nx + tidx] + alphaSquared * (d_u[tidy * Nx + tidx - 1] +
										d_u[tidy * Nx + tidx + 1] +
										d_u[(tidy + 1) * Nx + tidx] +
										d_u[(tidy - 1) * Nx + tidx] -
								   4. * d_u[tidy * Nx + tidx]);
}

/********************************************************/
/* DEVICE-SIZE FIELD UPDATE FUNCTION - SHARED MEMORY v2 */
/********************************************************/
__global__ void updateDevice_v2(const double * __restrict__ d_uold, const double * __restrict__ d_u, double * __restrict__ d_unew, const double alphaSquared,
							    const int Nx, const int Ny) {

	const int tidx = blockIdx.x * (BLOCKSIZEX - 2) + threadIdx.x;
	const int tidy = blockIdx.y * (BLOCKSIZEY - 2) + threadIdx.y;

	if ((tidx >= Nx) || (tidy >= Ny)) return;

	__shared__ double d_u_sh[BLOCKSIZEX][BLOCKSIZEY];

	// --- Load data to shared memory. Halo regions ARE loaded.
	d_u_sh[threadIdx.x][threadIdx.y] = d_u[tidy * Nx + tidx];
	__syncthreads();

	if (((threadIdx.x > 0) && (threadIdx.x < (BLOCKSIZEX - 1)) && (threadIdx.y > 0) && (threadIdx.y < (BLOCKSIZEY ‐ 1))) &&
		(tidx < Nx - 1 && tidy < Ny - 1))

		d_unew[tidy * Nx + tidx] = 2. * d_u_sh[threadIdx.x][threadIdx.y] -
									    d_uold[tidy * Nx + tidx] +
					    alphaSquared * (d_u_sh[threadIdx.x - 1][threadIdx.y] +
										d_u_sh[threadIdx.x + 1][threadIdx.y] +
										d_u_sh[threadIdx.x][threadIdx.y - 1] +
										d_u_sh[threadIdx.x][threadIdx.y + 1] -
								   4. * d_u_sh[threadIdx.x][threadIdx.y]);
}

/********************************************************/
/* DEVICE-SIZE FIELD UPDATE FUNCTION - SHARED MEMORY v3 */
/********************************************************/
__global__ void updateDevice_v3(const double * __restrict__ d_uold, const double * __restrict__ d_u, double * __restrict__ d_unew, const double alphaSquared,
								const int Nx, const int Ny) {

	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	if ((tidx >= Nx) || (tidy >= Ny)) return;

	const int tid_block = threadIdx.y * BLOCKSIZEX + threadIdx.x;		// --- Flattened thread index within a block

	const int tidx1 = tid_block % (BLOCKSIZEX + 2);
	const int tidy1 = tid_block / (BLOCKSIZEY + 2);

	const int tidx2 = (BLOCKSIZEX * BLOCKSIZEY + tid_block) % (BLOCKSIZEX + 2);
	const int tidy2 = (BLOCKSIZEX * BLOCKSIZEY + tid_block) / (BLOCKSIZEY + 2);

	__shared__ double d_u_sh[BLOCKSIZEX + 2][BLOCKSIZEY + 2];

	if (((blockIdx.x * BLOCKSIZEX - 1 + tidx1) < Nx) && ((blockIdx.x * BLOCKSIZEX - 1 + tidx1) >= 0) &&
		((blockIdx.y * BLOCKSIZEY - 1 + tidy1) < Ny) && ((blockIdx.y * BLOCKSIZEY - 1 + tidy1) >= 0))
		d_u_sh[tidx1][tidy1] = d_u[(blockIdx.x * BLOCKSIZEX - 1 + tidx1) + (blockIdx.y * BLOCKSIZEY - 1 + tidy1) * Nx];

	if (((tidx2 < (BLOCKSIZEX + 2)) && (tidy2 < (BLOCKSIZEY + 2))) &&
		((blockIdx.x * BLOCKSIZEX - 1 + tidx2) < Nx) && ((blockIdx.x * BLOCKSIZEX - 1 + tidx2) >= 0) &&
		((blockIdx.y * BLOCKSIZEY - 1 + tidy2) < Ny) && ((blockIdx.y * BLOCKSIZEY - 1 + tidy2) >= 0))
		d_u_sh[tidx2][tidy2] = d_u[(blockIdx.x * BLOCKSIZEX - 1 + tidx2) + (blockIdx.y * BLOCKSIZEY - 1 + tidy2) * Nx];

	__syncthreads();

	if ((tidx > 0 && tidx < Nx - 1 && tidy > 0 && tidy < Ny - 1))

		d_unew[tidy * Nx + tidx] = 2. * d_u_sh[threadIdx.x + 1][threadIdx.y + 1] -
									    d_uold[tidy * Nx + tidx] +
					    alphaSquared * (d_u_sh[threadIdx.x][threadIdx.y + 1] +
										d_u_sh[threadIdx.x + 2][threadIdx.y + 1] +
										d_u_sh[threadIdx.x + 1][threadIdx.y] +
										d_u_sh[threadIdx.x + 1][threadIdx.y + 2] -
								        4. * d_u_sh[threadIdx.x + 1][threadIdx.y + 1]);
}

/********/
/* MAIN */
/********/
int main()
{

	const int		Nx = 512;															// --- Number of mesh points along x
	const int		Ny = 512;															// --- Number of mesh points along y
	const double	Lx = 200.;															// --- Length of the domain along x
	const double	Ly = 200;															// --- Length of the domain along y
	double   *h_x = h_linspace(0., Lx, Nx);												// --- Mesh points along x
	double   *h_y = h_linspace(0., Ly, Ny);												// --- Mesh points along y
	const double	dx = h_x[2] - h_x[1];												// --- Mesh step along x
	const double	dy = h_y[2] - h_y[1];												// --- Mesh step along y
	const double	v = 5.;																// --- Wave speed
	const double	p = 0.0;															// --- Wave decay factor
	const double	dt = 0.25 / (v * sqrt((1. / dx) * (1. / dx) + (1. / dy) * (1. / dy)));
	// --- Time - Step matching the Courant - Friedrichs - Lewy condition
	const int		T = floor((3. * sqrt(Lx * Lx + Ly * Ly) / v) / dt);					// --- Total number of time steps

	double   *h_u = (double *)calloc(Nx * Ny, sizeof(double));							// --- Current solution u(x, y, t)		- host
	double   *h_uold = (double *)calloc(Nx * Ny, sizeof(double));						// --- Solution at the previous step	- host
	double   *h_unew = (double *)calloc(Nx * Ny, sizeof(double));						// --- Solution at the next step		- host

	double	*d_u; gpuErrchk(cudaMalloc((void**)&d_u, Nx * Ny * sizeof(double)));		// --- Current solution u(x, y, t)		- device
	double	*d_uold; gpuErrchk(cudaMalloc((void**)&d_uold, Nx * Ny * sizeof(double)));	// --- Solution at the previous step	- device
	double	*d_unew; gpuErrchk(cudaMalloc((void**)&d_unew, Nx * Ny * sizeof(double)));	// --- Solution at the next step		- device

	gpuErrchk(cudaMemset(d_unew, 0, Nx * Ny * sizeof(double)));

	// --- Initial conditions
	const int		indxc = floor(Nx / 3) - 1;											// --- Index for the source location along x
	const int		indyc = floor(Ny / 2) - 1;											// --- Index for the source location along y
	const double	xc = h_x[indxc];													// --- x - coordinate of source
	const double	yc = h_y[indyc];													// --- y - coordinate of source
	const int		indRc = 50;
	const double	Rc = Lx / indRc;
	for (int j = 0; j < Ny; j++)
		for (int i = 0; i < Nx; i++) {
		if (sqrt((h_x[i] - xc) * (h_x[i] - xc) + (h_y[j] - yc) * (h_y[j] - yc)) <= Rc)
			h_u[j * Nx + i] = exp(-indRc * ((h_x[i] - xc) * (h_x[i] - xc) + (h_y[j] - yc) * (h_y[j] - yc)) / Lx);
		h_uold[j * Nx + i] = h_u[j * Nx + i];
		}

	// --- Transfering the initial condition from host to device 
	gpuErrchk(cudaMemcpy(d_uold, h_uold, Nx * Ny * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_u, h_u, Nx * Ny * sizeof(double), cudaMemcpyHostToDevice));

	/*********************/
	/* ITERATIONS - HOST */
	/*********************/
	const double	alphaSquared = dt * dt * v * v / (dx * dx);									// --- CFL number

	for (int tt = 0; tt < T; tt++) {
		updateHost(h_uold, h_u, h_unew, alphaSquared, Nx, Ny);
		h_uold = h_u;																	// --- Curent solution becomes old
		h_u = h_unew;																	// --- New solution becomes current
		h_unew = h_uold;
	}

	/***********************/
	/* ITERATIONS - DEVICE */
	/***********************/
	// --- For the cases of no shared memory and shared memory v1 and v3
	dim3	Grid(iDivUp(Nx, BLOCKSIZEX), iDivUp(Ny, BLOCKSIZEY));
	dim3	Block(BLOCKSIZEX, BLOCKSIZEY);

	// --- For the case of shared memory v2 only
	//dim3	Grid(iDivUp(Nx, BLOCKSIZEX - 2), iDivUp(Ny, BLOCKSIZEY - 2));
	//dim3	Block(BLOCKSIZEX, BLOCKSIZEY);

	TimingGPU timerGPU;
	timerGPU.StartCounter();
	for (int tt = 0; tt < T; tt++) {
		//updateDevice_v0<<<Grid, Block>>>(d_uold, d_u, d_unew, alphaSquared, Nx, Ny);
		//updateDevice_v1<<<Grid, Block>>>(d_uold, d_u, d_unew, alphaSquared, Nx, Ny);
		//updateDevice_v2 << <Grid, Block >> >(d_uold, d_u, d_unew, alphaSquared, Nx, Ny);
		updateDevice_v3 << <Grid, Block >> >(d_uold, d_u, d_unew, alphaSquared, Nx, Ny);
#ifdef DEBUG 
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
		d_uold = d_u;																	// --- Curent solution becomes old
		d_u = d_unew;																	// --- New solution becomes current
		d_unew = d_uold;
	}
	printf("GPU timing %f\n", timerGPU.GetCounter());

	saveCPUrealtxt(h_u, "D:\\PDEs\\Wave-Equation\\2D\\Matlab\\FDTD2D_hostResult.txt", Nx * Ny);

	double *h_uDevice = (double *)malloc(Nx * Ny * sizeof(double));
	gpuErrchk(cudaMemcpy(h_uDevice, d_u, Nx * Ny * sizeof(double), cudaMemcpyDeviceToHost));
	saveCPUrealtxt(h_uDevice, "D:\\PDEs\\Wave-Equation\\2D\\Matlab\\FDTD2D_deviceResult.txt", Nx * Ny);

	return 0;
}
