#include <fstream>
#include <iomanip>
#include <math.h>       /* log */

#include <cusparse.h>

#include <thrust\device_ptr.h>
#include <thrust\sequence.h>

#include "TimingGPU.cuh"

#define BLOCKSIZE 256

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

/***************************/
/* CUSPARSE ERROR CHECKING */
/***************************/
static const char *_cusparseGetErrorEnum(cusparseStatus_t error)
{
	switch (error)
	{

	case CUSPARSE_STATUS_SUCCESS:
		return "CUSPARSE_STATUS_SUCCESS";

	case CUSPARSE_STATUS_NOT_INITIALIZED:
		return "CUSPARSE_STATUS_NOT_INITIALIZED";

	case CUSPARSE_STATUS_ALLOC_FAILED:
		return "CUSPARSE_STATUS_ALLOC_FAILED";

	case CUSPARSE_STATUS_INVALID_VALUE:
		return "CUSPARSE_STATUS_INVALID_VALUE";

	case CUSPARSE_STATUS_ARCH_MISMATCH:
		return "CUSPARSE_STATUS_ARCH_MISMATCH";

	case CUSPARSE_STATUS_MAPPING_ERROR:
		return "CUSPARSE_STATUS_MAPPING_ERROR";

	case CUSPARSE_STATUS_EXECUTION_FAILED:
		return "CUSPARSE_STATUS_EXECUTION_FAILED";

	case CUSPARSE_STATUS_INTERNAL_ERROR:
		return "CUSPARSE_STATUS_INTERNAL_ERROR";

	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

	case CUSPARSE_STATUS_ZERO_PIVOT:
		return "CUSPARSE_STATUS_ZERO_PIVOT";
	}

	return "<unknown>";
}

inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
	if (CUSPARSE_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUSPARSE error in file '%s', line %d, error %s\nterminating!\n", __FILE__, __LINE__, \
			_cusparseGetErrorEnum(err)); \
			assert(0); \
	}
}

extern "C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }

/***************/
/* COLON - GPU */
/***************/
template <class T>
T * d_colon(const T a, const T step, const T b) {

	int N = (int)((b - a) / step) + 1;

	T *out_array; gpuErrchk(cudaMalloc(&out_array, N * sizeof(T)));

	thrust::device_ptr<T> d = thrust::device_pointer_cast(out_array);

	thrust::sequence(d, d + N, a, step);

	return out_array;
}

template float  * d_colon<float>(const float  a, const float  step, const float  b);
template double * d_colon<double>(const double a, const double step, const double b);

/****************************/
/* RIGHT-HAND SIDE FUNCTION */
/****************************/
__host__ __device__ double rhsFunction(const double x, const double alpha) {

	return (2. * alpha * exp(-alpha * (x * x)) * (1. - 2. * alpha * (x * x)));

}

/***************************************/
/* KERNEL FOR FILLING MATRIX DIAGONALS */
/***************************************/
__global__ void diagonalsKernel(double * __restrict__ d_ld, double * __restrict__ d_d, double * __restrict__ d_ud, const double Deltax, const int Ninner) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	const double Deltax2 = Deltax * Deltax;
	
	if (tid >= Ninner) return;

	d_d[tid] = 2. / Deltax2;

	if (tid > 0) d_ld[tid] = -1. / Deltax2;
	else d_ld[tid] = 0.;

	if (tid < (Ninner - 1)) d_ud[tid] = -1. / Deltax2;
	else d_ud[tid] = 0.;

}

/******************************************/
/* KERNEL FOR FILLING THE RIGHT-HAND SIDE */
/******************************************/
__global__ void rhsKernel(double * __restrict__ d_brhs, const double * __restrict__ d_x, const double Ta, 
	                      const double Tb, const double Deltax, const double alpha, const int Ninner) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= Ninner) return;

	d_brhs[tid] = rhsFunction(d_x[tid + 1], alpha);
	
	if (tid == 0) d_brhs[tid] = d_brhs[tid] + Ta / (Deltax * Deltax);
	
	if (tid == Ninner - 1) d_brhs[tid] = d_brhs[tid] + Tb / (Deltax * Deltax);
	
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

	// --- Initialize cuSPARSE
	cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));			

	TimingGPU timerGPU;
	
	// --- Resolution domain
	const double a			= -1;													// --- Left border of the resolution domain
	const double b			= 1;													// --- Right border of the resolution domain

	// --- Dirichlet boundary conditions
	const double Ta			= 0.1;													// --- Boundary conditions at the left border of the resolution domain
	const double Tb			= 0.1;													// --- Boundary conditions at the right border of the resolution domain

	const int    Nintervals = 8194;													// --- Number of discretization intervals
	const int    Ninner		= Nintervals - 1;										// --- Number of inner mesh nodes
	const double Deltax		= (b - a) / (double)Nintervals;							// --- Discretization step
		
	double *d_x				= d_colon<double>(a, Deltax, b);						// --- Node coordinates

	double alpha			= -log(Ta) / (b * b);									// --- Parameter for the rhs function

	// --- Device space for diagonal, super-diagonal and lower diagonal
	double *d_ld;   gpuErrchk(cudaMalloc(&d_ld, Ninner * sizeof(double)));
	double *d_d;    gpuErrchk(cudaMalloc(&d_d,  Ninner * sizeof(double)));
	double *d_ud;   gpuErrchk(cudaMalloc(&d_ud, Ninner * sizeof(double)));

	// --- Defining the tridiagonal matrix by its diagonals
	diagonalsKernel << <iDivUp(Ninner, BLOCKSIZE), BLOCKSIZE >> >(d_ld, d_d, d_ud, Deltax, Ninner);

	// --- Device space for the (dense) rhs (on input) and solution (on output) of the linear system
	double *d_brhs;		gpuErrchk(cudaMalloc(&d_brhs, Ninner * sizeof(double)));

	// --- Defining the (dense) rhs of the linear system
	rhsKernel << <iDivUp(Ninner, BLOCKSIZE), BLOCKSIZE >> >(d_brhs, d_x, Ta, Tb, Deltax, alpha, Ninner);

	// --- Solving the tridiagonal system - Internal storage - Pivoting
	//timerGPU.StartCounter();
	//cusparseSafeCall(cusparseDgtsv(handle, Ninner, 1, d_ld, d_d, d_ud, d_brhs, Ninner));
	//printf("Execution time = %f\n", timerGPU.GetCounter());

	// --- Solving the tridiagonal system - Internal storage - No pivoting
	//timerGPU.StartCounter();
	//cusparseSafeCall(cusparseDgtsv_nopivot(handle, Ninner, 1, d_ld, d_d, d_ud, d_brhs, Ninner));
	//printf("Execution time = %f\n", timerGPU.GetCounter());

	// --- Solving the tridiagonal system - External storage - Pivoting
	//timerGPU.StartCounter();
	//size_t *bufferSizeInBytes = (size_t *)malloc(sizeof(size_t));
	//cusparseDgtsv2_bufferSizeExt(handle, Ninner, 1, d_ld, d_d, d_ud, d_brhs, Ninner, bufferSizeInBytes);
	//double *pBuffer;	gpuErrchk(cudaMalloc(&pBuffer, bufferSizeInBytes[0] * sizeof(double)));
	//cusparseSafeCall(cusparseDgtsv2(handle, Ninner, 1, d_ld, d_d, d_ud, d_brhs, Ninner, pBuffer));
	//printf("Execution time = %f\n", timerGPU.GetCounter());

	// --- Solving the tridiagonal system - External storage - No pivoting
	timerGPU.StartCounter();
	size_t *bufferSizeInBytes = (size_t *)malloc(sizeof(size_t));
	cusparseDgtsv2_nopivot_bufferSizeExt(handle, Ninner, 1, d_ld, d_d, d_ud, d_brhs, Ninner, bufferSizeInBytes);
	double *pBuffer;	gpuErrchk(cudaMalloc(&pBuffer, bufferSizeInBytes[0] * sizeof(double)));
	cusparseSafeCall(cusparseDgtsv2_nopivot(handle, Ninner, 1, d_ld, d_d, d_ud, d_brhs, Ninner, pBuffer));
	printf("Execution time = %f\n", timerGPU.GetCounter());

	// --- Save results from GPU to file
	saveGPUrealtxt(d_brhs, ".\\d_brhs.txt", Ninner);

	return 0;
}


