#include <stdio.h>
#include <fstream>
#include <iomanip>

// --- Greek pi
#define _USE_MATH_DEFINES
#include <math.h>

#include <cufft.h>

#define BLOCKSIZEX		16
#define BLOCKSIZEY		16

#define prec_save 10

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}

void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**************************************************/
/* COMPUTE RIGHT HAND SIDE OF 2D POISSON EQUATION */
/**************************************************/
__global__ void computeRHS(const float * __restrict__ d_x, const float * __restrict__ d_y,
	                       float2 * __restrict__ d_r, const float Lx, const float Ly, const float sigma, 
						   const int M, const int N) {

	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= M) || (tidy >= N)) return;

	const float sigmaSquared = sigma * sigma;
	
	const float rSquared = (d_x[tidx] - 0.5f * Lx) * (d_x[tidx] - 0.5f * Lx) +
		                   (d_y[tidy] - 0.5f * Ly) * (d_y[tidy] - 0.5f * Ly);
	
	d_r[tidy * M + tidx].x = expf(-rSquared / (2.f * sigmaSquared)) * (rSquared - 2.f * sigmaSquared) / (sigmaSquared * sigmaSquared);
	d_r[tidy * M + tidx].y = 0.f;

}

/****************************************************/
/* SOLVE 2D POISSON EQUATION IN THE SPECTRAL DOMAIN */
/****************************************************/
__global__ void solvePoisson(const float * __restrict__ d_kx, const float * __restrict__ d_ky, 
	                          float2 * __restrict__ d_r, const int M, const int N)
{
	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= M) || (tidy >= N)) return;

	float scale = -(d_kx[tidx] * d_kx[tidx] + d_ky[tidy] * d_ky[tidy]);

	if (tidx == 0 && tidy == 0) scale = 1.f;

	scale = 1.f / scale;
	d_r[M * tidy + tidx].x *= scale;
	d_r[M * tidy + tidx].y *= scale;
	
}

/****************************************************************************/
/* SOLVE 2D POISSON EQUATION IN THE SPECTRAL DOMAIN - SHARED MEMORY VERSION */
/****************************************************************************/
__global__ void solvePoissonShared(const float * __restrict__ d_kx, const float * __restrict__ d_ky,
	float2 * __restrict__ d_r, const int M, const int N)
{
	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= M) || (tidy >= N)) return;

	// --- Use shared memory to minimize multiple access to same spectral coordinate values
	__shared__ float kx_s[BLOCKSIZEX], ky_s[BLOCKSIZEY];

	kx_s[threadIdx.x] = d_kx[tidx];
	ky_s[threadIdx.y] = d_ky[tidy];
	__syncthreads();

	float scale = -(kx_s[threadIdx.x] * kx_s[threadIdx.x] + ky_s[threadIdx.y] * ky_s[threadIdx.y]);

	if (tidx == 0 && tidy == 0) scale = 1.f;

	scale = 1.f / scale;
	d_r[M * tidy + tidx].x *= scale;
	d_r[M * tidy + tidx].y *= scale;

}

/******************************/
/* COMPLEX2REAL SCALED KERNEL */
/******************************/
__global__ void complex2RealScaled(float2 * __restrict__ d_r, float * __restrict__ d_result, const int M, const int N, float scale)
{
	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= M) || (tidy >= N)) return;

	d_result[tidy * M + tidx] = scale * (d_r[tidy * M + tidx].x - d_r[0].x);
}

/******************************************/
/* COMPLEX2REAL SCALED KERNEL - OPTIMIZED */
/******************************************/
__global__ void complex2RealScaledOptimized(float2 * __restrict__ d_r, float * __restrict__ d_result, const int M, const int N, float scale)
{
	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= M) || (tidy >= N)) return;

	__shared__ float d_r_0[1];

	if (threadIdx.x == 0) d_r_0[0] = d_r[0].x;
	
	volatile float2 c;
	c.x = d_r[tidy * M + tidx].x;
	c.y = d_r[tidy * M + tidx].y;

	d_result[tidy * M + tidx] = scale * (c.x - d_r_0[0]);
}

/**************************************/
/* SAVE FLOAT2 ARRAY FROM GPU TO FILE */
/**************************************/
void saveGPUcomplextxt(const float2 * d_in, const char *filename, const int M) {

	float2 *h_in = (float2 *)malloc(M * sizeof(float2));

	gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(float2), cudaMemcpyDeviceToHost));

	std::ofstream outfile;
	outfile.open(filename);
	for (int i = 0; i < M; i++) {
		outfile << std::setprecision(prec_save) << h_in[i].x << "\n"; outfile << std::setprecision(prec_save) << h_in[i].y << "\n";
	}
	outfile.close();

}

/*************************************/
/* SAVE FLOAT ARRAY FROM GPU TO FILE */
/*************************************/
template <class T>
void saveGPUrealtxt(const T * d_in, const char *filename, const int M) {

	T *h_in = (T *)malloc(M * sizeof(T));

	gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));

	std::ofstream outfile;
	outfile.open(filename);
	for (int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n";
	outfile.close();

}

/********/
/* MAIN */
/********/
int main()
{
	const int	M		= 32;              // --- Number of Fourier harmonics along x (should be a multiple of 2)
	const int	N		= 128;              // --- Number of Fourier harmonics along y (should be a multiple of 2)
	const float Lx		= 3.f;             // --- Domain size along x
	const float Ly		= 1.5f;            // --- Domain size along y
	const float sigma	= 0.1f;            // --- Characteristic width of f (make << 1)
		
	// --- Wavenumbers on the host
	float *h_kx = (float *)malloc(M * sizeof(float));
	float *h_ky = (float *)malloc(N * sizeof(float));
	for (int k = 0; k < M / 2; k++)  h_kx[k]		= (2.f * M_PI / Lx) * k;
	for (int k = -M / 2; k < 0; k++) h_kx[k + M]	= (2.f * M_PI / Lx) * k;
	for (int k = 0; k < N / 2; k++)  h_ky[k]		= (2.f * M_PI / Ly) * k;
	for (int k = -N / 2; k < 0; k++) h_ky[k + N]	= (2.f * M_PI / Ly) * k;

	// --- Wavenumbers on the device
	float *d_kx;	gpuErrchk(cudaMalloc(&d_kx, M * sizeof(float)));
	float *d_ky;	gpuErrchk(cudaMalloc(&d_ky, N * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_kx, h_kx, M * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_ky, h_ky, N * sizeof(float), cudaMemcpyHostToDevice));

	// --- Domain discretization on the host
	float *h_x = (float *)malloc(M * sizeof(float));
	float *h_y = (float *)malloc(N * sizeof(float));
	for (int k = 0; k < M; k++)  h_x[k] = Lx / (float)M * k;
	for (int k = 0; k < N; k++)  h_y[k] = Ly / (float)N * k;

	// --- Domain discretization on the device
	float *d_x;		gpuErrchk(cudaMalloc(&d_x, M * sizeof(float)));
	float *d_y;		gpuErrchk(cudaMalloc(&d_y, N * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_x, h_x, M * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

	// --- Compute right-hand side of the differential equation on the device
	float2 *d_r;	gpuErrchk(cudaMalloc(&d_r, M * N * sizeof(float2)));
	dim3 dimBlock(BLOCKSIZEX, BLOCKSIZEY);
	dim3 dimGrid(iDivUp(M, BLOCKSIZEX), iDivUp(N, BLOCKSIZEY));
	computeRHS << <dimGrid, dimBlock >> >(d_x, d_y, d_r, Lx, Ly, sigma, M, N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// --- Create plan for CUDA FFT
	cufftHandle plan;
	cufftPlan2d(&plan, N, M, CUFFT_C2C);
	
	// --- Compute in place forward FFT of right-hand side
	cufftExecC2C(plan, d_r, d_r, CUFFT_FORWARD);
	
	// --- Solve Poisson equation in Fourier space 
	//solvePoisson << <dimGrid, dimBlock >> > (d_kx, d_ky, d_r, M, N);
	solvePoissonShared << <dimGrid, dimBlock >> > (d_kx, d_ky, d_r, M, N);

	// --- Compute in place inverse FFT
	cufftExecC2C(plan, d_r, d_r, CUFFT_INVERSE);

	//saveGPUcomplextxt(d_r, ".\d_r.txt", M * N);

	// --- With cuFFT, an FFT followed by an IFFT will return the same array times the size of the transform
	// --- Accordingly, we need to scale the result.
	const float scale = 1.f / ((float)M * (float)N);
	float *d_result;	gpuErrchk(cudaMalloc(&d_result, M * N * sizeof(float)));
	//complex2RealScaled << <dimGrid, dimBlock >> > (d_r, d_result, M, N, scale);
	complex2RealScaledOptimized << <dimGrid, dimBlock >> > (d_r, d_result, M, N, scale);

	saveGPUrealtxt(d_result, ".\\d_result.txt", M * N);

	return 0;

}

