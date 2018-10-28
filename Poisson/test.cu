#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <numeric>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse_v2.h"

#define BLOCKSIZEMEMSET			256

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int d_f){ return ((a % d_f) != 0) ? (a / d_f + 1) : (a / d_f); }

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


#include "cublas_wrapper.h"
#include "solver.h"
#include <fstream>
#include <iomanip>


#define prec_save 10

/*
 *      test.cu
 *
 *      This file serves as demonstration how to use the given code and runs tests for the different implementations.
 *
 *      @author Simon Schoelly
*/


using namespace std;

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
	
	cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

 	const int		M			= 128;					// --- Number of discretization points along d_u and y
    const int		maxIter		= 10000;				// --- Maximum number of iterations
    const double    tolerance	= 0.0000001;			// --- Conjugate gradient convergence tolerance
        
	// --- Equation right-hand side
	double *d_f; gpuErrchk(cudaMalloc(&d_f, (M * M) * sizeof(double)));
	deviceMemset<double>(d_f, (double)1.0, M * M);

	// --- Equation unknown
	double *d_u; gpuErrchk(cudaMalloc(&d_u, (M *M) * sizeof(double)));

	int numIter = conjugateGradientPoisson<double>(cublas_handle, M, d_f, d_u, maxIter, tolerance);
	cout << "Number of performed iterations performed " << numIter << endl;

	saveGPUrealtxt(d_u, ".\\d_result_x.txt", M * M);
	saveGPUrealtxt(d_f, ".\\d_result_b.txt", M * M);

        //double *b_3d, *x_3d;
        //int m_3d = 128;

        //cudaMalloc((void **) &b_3d, (m_3d*m_3d*m_3d)*sizeof(double));
        //cudaMalloc((void **) &x_3d, (m_3d*m_3d*m_3d)*sizeof(double));
        //deviceMemset<double>(b_3d, 1.0, M*M);

        //ThomasPreconditioner3D<double> preconditioner_3d;
        //numIter = solve_with_conjugate_gradient3D<double>(cublas_handle, cusparse_handle, m_3d, alpha, b_3d, x_3d, maxIter, tolerance, &preconditioner_3d);
        //cout << numIter << " iterations" << endl;
         

}
