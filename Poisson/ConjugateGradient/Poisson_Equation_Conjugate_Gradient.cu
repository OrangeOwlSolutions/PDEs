#include <iostream>
#include <fstream>
#include <iomanip>

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
template<class FT>
//__global__ void multiply_by_A(int const M, FT const alpha, FT const * const x, FT * const b) {
__global__ void multiply_by_A(int const M, FT const * const x, FT * const b) {
	int n = M * M;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n) {
		return;
	}

	//FT value = (alpha + FT(4))*x[tid];
	FT value = (FT(4))*x[tid];

	if (tid % M != 0) {
		value -= x[tid - 1];
	}
	if ((tid + 1) % M != 0) {
		value -= x[tid + 1];
	}
	if (tid + M < n) {
		value -= x[tid + M];
	}
	if (tid - M >= 0) {
		value -= x[tid - M];
	}
	b[tid] = value;
}

/*****************************/
/* CONJUGATE GRADIENT SOLVER */
/*****************************/
template<class T>
int conjugateGradientPoisson(cublasHandle_t const cublasHandle, int const M, T const * const d_b, T * const d_x, 
							 int maxIter, T tol) {

	//int const n = M*M;

	T *d_p;		gpuErrchk(cudaMalloc(&d_p,  M * M * sizeof(T)));
	T *d_r;		gpuErrchk(cudaMalloc(&d_r,  M * M * sizeof(T)));
	T *d_h;		gpuErrchk(cudaMalloc(&d_h,  M * M * sizeof(T)));
	T *d_Ax;	gpuErrchk(cudaMalloc(&d_Ax, M * M * sizeof(T)));
	T *d_q;		gpuErrchk(cudaMalloc(&d_q,  M * M * sizeof(T)));

	T beta, c, ph;

	T const T_ONE(1);
	T const T_MINUS_ONE(-1);

	// --- Solution initialization d_x = 0
	deviceMemset<T>(d_x, T(0), M * M); 

	// --- d_Ax = A * d_x
	multiply_by_A<T> << <iDivUp(M * M, 1024), 1024 >> >(M, d_x, d_Ax);
	
	// d_r = d_b
	//cublas_copy(cublasHandle, n, d_b, d_r);
	cublasTcopy(cublasHandle, M * M, d_b, 1, d_r, 1);

	// --- d_r = d_r - d_Ax = d_b - d_Ax
	//cublas_axpy(cublasHandle, n, &T_MINUS_ONE, d_Ax, d_r);
	cublasTaxpy(cublasHandle, M * M, &T_MINUS_ONE, d_Ax, 1, d_r, 1);

	// --- norm0 = ||d_r||
	T norm0;
	//cublas_nrm2(cublasHandle, n, d_r, &norm0);
	cublasTnrm2(cublasHandle, M * M, d_r, 1, &norm0);

	// d_p = d_r
	//cublas_copy(cublasHandle, n, d_r, d_p);
	cublasTcopy(cublasHandle, M * M, d_r, 1, d_p, 1);

	int numIter;
	for (numIter = 1; numIter <= maxIter; ++numIter) {
		
		// --- beta = <d_r, d_r>
		//cublas_dot(cublasHandle, n, d_r, d_r, &beta);
		cublasTdot(cublasHandle, M * M, d_r, 1, d_r, 1, &beta);

		// --- d_h = Ap
		multiply_by_A<T> << <iDivUp(M * M, BLOCKSIZEMULTIPLY), BLOCKSIZEMULTIPLY >> >(M, d_p, d_h);

		// --- ph = <d_p, d_h>
		//cublas_dot(cublasHandle, n, d_p, d_h, &ph);
		cublasTdot(cublasHandle, M * M, d_p, 1, d_h, 1, &ph);

		c = beta / ph;

		// --- d_x = d_x + c * d_p
		//cublas_axpy(cublasHandle, n, &c, d_p, d_x);
		cublasTaxpy(cublasHandle, M * M, &c, d_p, 1, d_x, 1);

		// --- d_r = d_r - c * d_h
		T minus_c = -c;
		//cublas_axpy(cublasHandle, n, &minus_c, d_h, d_r);
		cublasTaxpy(cublasHandle, M * M, &minus_c, d_h, 1, d_r, 1);

		T norm;
		//cublas_nrm2(cublasHandle, n, d_r, &norm);
		cublasTnrm2(cublasHandle, M * M, d_r, 1, &norm);
		if (norm <= tol * norm0) break;

		// --- rr = <d_r, d_r>
		T rr;
		//cublas_dot(cublasHandle, n, d_r, d_r, &rr);
		cublasTdot(cublasHandle, M * M, d_r, 1, d_r, 1, &rr);

		beta = rr / beta;

		// --- d_p = beta * d_p
		//cublas_scal(cublasHandle, n, &beta, d_p);
		cublasTscal(cublasHandle, M * M, &beta, d_p, 1);

		//cublas_axpy(cublasHandle, n, &T_ONE, d_r, d_p);
		cublasTaxpy(cublasHandle, M * M, &T_ONE, d_r, 1, d_p, 1);
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

	const int		M = 128;					// --- Number of discretization points along d_u and y
	const int		maxIter = 10000;				// --- Maximum number of iterations
	const double    tol = 0.0000001;			// --- Conjugate gradient convergence tol

													// --- Equation right-hand side
	double *d_f; gpuErrchk(cudaMalloc(&d_f, (M * M) * sizeof(double)));
	deviceMemset<double>(d_f, (double)1.0, M * M);

	// --- Equation unknown
	double *d_u; gpuErrchk(cudaMalloc(&d_u, (M *M) * sizeof(double)));

	int numIter = conjugateGradientPoisson<double>(cublasHandle, M, d_f, d_u, maxIter, tol);
	cout << "Number of performed iterations performed " << numIter << endl;

	saveGPUrealtxt(d_u, ".\\d_result_x.txt", M * M);
	saveGPUrealtxt(d_f, ".\\d_result_b.txt", M * M);

	//double *b_3d, *x_3d;
	//int m_3d = 128;

	//cudaMalloc((void **) &b_3d, (m_3d*m_3d*m_3d)*sizeof(double));
	//cudaMalloc((void **) &x_3d, (m_3d*m_3d*m_3d)*sizeof(double));
	//deviceMemset<double>(b_3d, 1.0, M*M);

	//ThomasPreconditioner3D<double> preconditioner_3d;
	//numIter = solve_with_conjugate_gradient3D<double>(cublasHandle, cusparse_handle, m_3d, alpha, b_3d, x_3d, maxIter, tol, &preconditioner_3d);
	//cout << numIter << " iterations" << endl;


}
