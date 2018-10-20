#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iomanip>

// --- Greek pi
#define _USE_MATH_DEFINES
#include <math.h>

#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

#define prec_save 10

//#define REAL_TYPE	float
#define REAL_TYPE	double

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

/***********************************/
/* JACOBI ITERATOR KERNEL FUNCTION */
/***********************************/
template <class T>
__global__ void jacobiIteratorGPU(const T * __restrict__ d_u, T * __restrict__ d_u_old, const T * __restrict__ d_f, const T dxdy, const int M, const int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < M - 1 && j < N - 1 && i > 0 && j > 0)
		d_u_old[j * M + i] = (d_u[j * M + (i - 1)] + d_u[j * M + (i + 1)] + d_u[(j - 1) * M + i] + d_u[(j + 1) * M + i] + (dxdy * d_f[j * M + i])) * 0.25;
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
	const int		M			= 80;							// --- Number of discretization points along x
	const int		N			= 100;							// --- Number of discretization points along y
	const int		numIter		= 2000;							// --- Number of iterations
	const REAL_TYPE dx			= 1.f / ((REAL_TYPE)M - 1.f);	// --- Discretization step along x
	const REAL_TYPE	dy			= 1.f / ((REAL_TYPE)N - 1.f);	// --- Discretization step along y

	/****************************************************/
	/* DISCRETIZATION POINTS ALONG X ON HOST AND DEVICE */
	/****************************************************/
	REAL_TYPE *h_x = (REAL_TYPE *)malloc(M * sizeof(REAL_TYPE));
	REAL_TYPE *h_y = (REAL_TYPE *)malloc(N * sizeof(REAL_TYPE));
	for (int k = 0; k < M; k++) h_x[k] = k * dx;
	for (int k = 0; k < N; k++) h_y[k] = k * dy;

	int deviceCount;
	gpuErrchk(cudaGetDeviceCount(&deviceCount));

	REAL_TYPE *d_x; gpuErrchk(cudaMalloc(&d_x, M * sizeof(REAL_TYPE)));
	REAL_TYPE *d_y; gpuErrchk(cudaMalloc(&d_y, N * sizeof(REAL_TYPE)));
	gpuErrchk(cudaMemcpy(d_x, h_x, M * sizeof(REAL_TYPE), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_y, h_y, N * sizeof(REAL_TYPE), cudaMemcpyHostToDevice));

	/**********************************/
	/* SOURCE TERM ON HOST AND DEVICE */
	/**********************************/
	REAL_TYPE *h_f = (REAL_TYPE *)malloc(M * N * sizeof(REAL_TYPE));
	for (int n = 0; n < N; n++)
		for (int m = 0; m < M; m++)
			h_f[n * M + m] = h_x[m] * h_x[m] + h_y[n] * h_y[n];

	REAL_TYPE *d_f; gpuErrchk(cudaMalloc(&d_f, M * N * sizeof(REAL_TYPE)));
	gpuErrchk(cudaMemcpy(d_f, h_f, M * N * sizeof(REAL_TYPE), cudaMemcpyHostToDevice));

	/*************************************************************************/
	/* INITIALIZING THE SOLUTION WITH BOUNDARY CONDITIONS ON HOST AND DEVICE */
	/*************************************************************************/
	REAL_TYPE *h_u = (REAL_TYPE *)calloc(M * N, sizeof(REAL_TYPE));
	REAL_TYPE *d_u;		gpuErrchk(cudaMalloc(&d_u, M * N * sizeof(REAL_TYPE)));
	REAL_TYPE *d_u_old; gpuErrchk(cudaMalloc(&d_u_old, M * N * sizeof(REAL_TYPE)));

	for (int m = 0; m < M; m++) {
		h_u[(N - 1) * M + m] = h_x[m] * h_x[m] / 2.f;	// --- Upper boundary condition
		h_u[0 * M + m]		 = 0.f;						// --- Lower boundary condition
	}
	for (int n = 0; n < N; n++) {
		h_u[n * M + 0]		 = sin(M_PI * h_y[n]);						// --- Left boundary condition
		h_u[n * M + M - 1]   = exp(M_PI) * sin(M_PI * h_y[n]) + h_y[n] * h_y[n] / 2.f;							
																		// --- Right boundary condition
	}

	gpuErrchk(cudaMemcpy(d_u,	  h_u, M * N * sizeof(REAL_TYPE), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_u_old, d_u, M * N * sizeof(REAL_TYPE), cudaMemcpyDeviceToDevice));

	/**************/
	/* ITERATIONS */
	/**************/
	// --- Device iterations
	dim3 DimBlock(BLOCKSIZEX, BLOCKSIZEY);
	dim3 DimGrid(iDivUp(M, BLOCKSIZEX), iDivUp(N, BLOCKSIZEY));

	for (int k = 0; k < numIter; k++)
	{
		jacobiIteratorGPU << <DimGrid, DimBlock >> >(d_u, d_u_old, d_f, dx * dx, M, N);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// --- Pointers swap
		REAL_TYPE *temp = d_u;
		d_u = d_u_old;
		d_u_old = temp;
	}

	// --- Save results from GPU to file
	saveGPUrealtxt(d_u, ".\\d_result.txt", M * N);

	return 0;
}
