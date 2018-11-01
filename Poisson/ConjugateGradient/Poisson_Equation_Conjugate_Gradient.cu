#include <iostream>
#include <fstream>
#include <iomanip>
#include <assert.h>

#include "cublas.h"

#define prec_save 10

using namespace std;

#include "cublasWrappers.cuh"

#define BLOCKSIZEMEMSET			256
#define BLOCKSIZEMULTIPLY		256

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int d_f) { return ((a % d_f) != 0) ? (a / d_f + 1) : (a / d_f); }

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

/*************************/
/* CUBLAS ERROR CHECKING */
/*************************/
static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
	switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";

	case CUBLAS_STATUS_NOT_SUPPORTED:
		return "CUBLAS_STATUS_NOT_SUPPORTED";

	case CUBLAS_STATUS_LICENSE_ERROR:
		return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "<unknown>";
}

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
	if (CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d, error: %s\nterminating!\n", __FILE__, __LINE__, \
			_cublasGetErrorEnum(err)); \
			assert(0); \
	}
}

extern "C" void cublasSafeCall(cublasStatus_t err) { __cublasSafeCall(err, __FILE__, __LINE__); }

/*****************/
/* DEVICE MEMSET */
/*****************/
template<class T>
__global__ void deviceMemsetKernel(T * const devPtr, T const value, size_t const N) {

	size_t const tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= N)	return;

	devPtr[tid] = value;
}

template<class T>
void deviceMemset(T * const devPtr, T const value, size_t const N) {
	deviceMemsetKernel<T> << <iDivUp(N, BLOCKSIZEMEMSET), BLOCKSIZEMEMSET >> >(devPtr, value, N);
}

/****************************/
/* Ax DEVICE MULTIPLICATION */
/****************************/
template<class T>
__global__ void AxKernel(int const M, T const * const d_x, T * const d_b) {
	
	int M2 = M * M;
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid >= M2) return;

	T value = (T(4)) * d_x[tid];

	if ( tid      % M != 0) value -= d_x[tid - 1];
	if ((tid + 1) % M != 0)	value -= d_x[tid + 1];
	if ( tid + M < M2) 		value -= d_x[tid + M];
	if (tid - M >= 0)		value -= d_x[tid - M];
	
	d_b[tid] = value;
}

/*****************************/
/* CONJUGATE GRADIENT SOLVER */
/*****************************/
template<class T>
int conjugateGradientPoisson(cublasHandle_t const cublasHandle, int const M, T const * const d_b, T * const d_x,
	int maxIter, T tol) {

	T *d_p;		gpuErrchk(cudaMalloc(&d_p, M * M * sizeof(T)));
	T *d_r;		gpuErrchk(cudaMalloc(&d_r, M * M * sizeof(T)));
	T *d_h;		gpuErrchk(cudaMalloc(&d_h, M * M * sizeof(T)));
	T *d_Ax;	gpuErrchk(cudaMalloc(&d_Ax, M * M * sizeof(T)));
	T *d_q;		gpuErrchk(cudaMalloc(&d_q, M * M * sizeof(T)));

	T beta, c, ph;

	T const T_ONE(1);
	T const T_MINUS_ONE(-1);

	// --- Solution initialization d_x = 0
	deviceMemset<T>(d_x, T(0), M * M);

	// --- d_Ax = A * d_x
	AxKernel<T> << <iDivUp(M * M, 1024), 1024 >> >(M, d_x, d_Ax);

	// --- d_r = d_b
	cublasSafeCall(cublasTcopy(cublasHandle, M * M, d_b, 1, d_r, 1));

	// --- d_r = d_r - d_Ax = d_b - d_Ax
	cublasSafeCall(cublasTaxpy(cublasHandle, M * M, &T_MINUS_ONE, d_Ax, 1, d_r, 1));

	// --- norm0 = ||d_r||
	T norm0;
	cublasSafeCall(cublasTnrm2(cublasHandle, M * M, d_r, 1, &norm0));

	// --- d_p = d_r
	cublasSafeCall(cublasTcopy(cublasHandle, M * M, d_r, 1, d_p, 1));

	int numIter;
	for (numIter = 1; numIter <= maxIter; ++numIter) {

		// --- beta = <d_r, d_r>
		cublasSafeCall(cublasTdot(cublasHandle, M * M, d_r, 1, d_r, 1, &beta));

		// --- d_h = Ap
		AxKernel<T> << <iDivUp(M * M, BLOCKSIZEMULTIPLY), BLOCKSIZEMULTIPLY >> >(M, d_p, d_h);

		// --- ph = <d_p, d_h>
		cublasSafeCall(cublasTdot(cublasHandle, M * M, d_p, 1, d_h, 1, &ph));

		c = beta / ph;

		// --- d_x = d_x + c * d_p
		cublasSafeCall(cublasTaxpy(cublasHandle, M * M, &c, d_p, 1, d_x, 1));

		// --- d_r = d_r - c * d_h
		T minus_c = -c;
		cublasSafeCall(cublasTaxpy(cublasHandle, M * M, &minus_c, d_h, 1, d_r, 1));

		T norm;
		cublasSafeCall(cublasTnrm2(cublasHandle, M * M, d_r, 1, &norm));
		if (norm <= tol * norm0) break;

		// --- rr = <d_r, d_r>
		T rr;
		cublasSafeCall(cublasTdot(cublasHandle, M * M, d_r, 1, d_r, 1, &rr));

		beta = rr / beta;

		// --- d_p = beta * d_p
		cublasSafeCall(cublasTscal(cublasHandle, M * M, &beta, d_p, 1));

		cublasSafeCall(cublasTaxpy(cublasHandle, M * M, &T_ONE, d_r, 1, d_p, 1));
	}

	gpuErrchk(cudaFree(d_p));
	gpuErrchk(cudaFree(d_r));
	gpuErrchk(cudaFree(d_h));
	gpuErrchk(cudaFree(d_Ax));
	gpuErrchk(cudaFree(d_q));

	return numIter;
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
int main() {

	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	const int		M = 128;						// --- Number of discretization points along d_u and y
	const int		maxIter = 10000;				// --- Maximum number of iterations
	const double    tol = 0.0000001;				// --- Conjugate gradient convergence tol

	// --- Equation right-hand side
	double *d_f; gpuErrchk(cudaMalloc(&d_f, (M * M) * sizeof(double)));
	deviceMemset<double>(d_f, (double)1.0, M * M);

	// --- Equation unknown
	double *d_u; gpuErrchk(cudaMalloc(&d_u, (M *M) * sizeof(double)));

	int numIter = conjugateGradientPoisson<double>(cublasHandle, M, d_f, d_u, maxIter, tol);
	cout << "Number of performed iterations performed " << numIter << endl;

	saveGPUrealtxt(d_u, ".\\d_result_x.txt", M * M);
	saveGPUrealtxt(d_f, ".\\d_result_b.txt", M * M);

}
