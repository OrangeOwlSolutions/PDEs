// --- Attenzione: riportare qui la tecnica con cui è stata implementata computePreconditioning

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cusparse_v2.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <stdio.h>

#include "Utilities.cuh"
#include "cusparseLU.cuh"
#include "precondConjugateGradientSparse.cuh"

#define	BLOCKSIZE		256

//#define VIDEO_DEBUG										// --- Enables video printing debug
//#define DEBUG

#define LU_DECOMPOSITION
//#define PRECONDCONJUGATEGRADIENT

// --- Defining tuple type
typedef thrust::tuple<int, int> Tuple;

typedef thrust::device_vector<Tuple>::iterator  dIter1;
typedef thrust::device_vector<float>::iterator  dIter2;

cusparseHandle_t	handleMain;
cusparseMatDescr_t	descrAMain = 0;

/**************************/
/* TUPLE ORDERING FUNCTOR */
/**************************/
struct TupleComp
{
	__host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
	{
		if (t1.get<0>() < t2.get<0>())
			return true;
		if (t1.get<0>() > t2.get<0>())
			return false;
		return t1.get<1>() < t2.get<1>();
	}
};

/************************************/
/* EQUALITY OPERATOR BETWEEN TUPLES */
/************************************/
struct BinaryPredicate
{
	__host__ __device__ bool operator () (const Tuple& lhs, const Tuple& rhs)
	{
		return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) && (thrust::get<1>(lhs) == thrust::get<1>(rhs));
	}
};

/**************************************/
/* ELEMENT CONNECTIVITY MATRIX KERNEL */
/**************************************/
__global__ void computeConnectivityMatrix(int * __restrict__ d_elementConnectivityMatrix, const int Ne)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= Ne) return;

	d_elementConnectivityMatrix[tid * 2 * Ne] = tid;
	d_elementConnectivityMatrix[tid * 2 * Ne + 1] = tid + 1;

}

/*********************************/
/* GLOBAL MATRIX KERNEL - ATOMIC */
/*********************************/
#ifdef VIDEO_DEBUG
__global__ void computeGlobalMatrixAtomic(float * __restrict__ d_K, float * __restrict__ d_f, float * __restrict__ d_fe,
	const float * __restrict__ d_Ke, const int * __restrict__ d_elementConnectivityMatrix, const int Ne,
	const int Nn) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= Ne) return;

	for (int k = 0; k < 2; k++) {

		const int row = d_elementConnectivityMatrix[tid * 2 * Ne + k];

		for (int l = 0; l < 2; l++) {

			const int col = d_elementConnectivityMatrix[tid * 2 * Ne + l];
			atomicAdd(d_K + row * Nn + col, d_Ke[k * 2 + l]);

		}

		atomicAdd(d_f + row, d_fe[k]);
	}

}
#endif

/***********************************/
/* GLOBAL F VECTOR KERNEL - ATOMIC */
/***********************************/
__global__ void computefVectorAtomic(float * __restrict__ d_f, float * __restrict__ d_fe, const int * __restrict__ d_elementConnectivityMatrix,
									 const int Ne) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= Ne) return;

	for (int k = 0; k < 2; k++) {

		const int row = d_elementConnectivityMatrix[tid * 2 * Ne + k];

		atomicAdd(d_f + row, d_fe[k]);
	}

}

/**************************************/
/* GLOBAL MATRIX PREPROCESSING KERNEL */
/**************************************/
__global__ void computeGlobalMatrixPreprocessing(float * __restrict__ d_KValues, int * __restrict__ d_rowIndices, int * __restrict__ d_colIndices,
	const int * __restrict__ d_elementConnectivityMatrix, const float * __restrict__ d_Ke, const int Ne) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= Ne) return;

	for (int k = 0; k < 2; k++) {

		for (int l = 0; l < 2; l++) {

			d_rowIndices[tid * 4 + k * 2 + l] = d_elementConnectivityMatrix[tid * 2 * Ne + k];
			d_colIndices[tid * 4 + k * 2 + l] = d_elementConnectivityMatrix[tid * 2 * Ne + l];

			d_KValues[tid * 4 + k * 2 + l] = d_Ke[k * 2 + l];

		}
	}
}

/*******************************/
/* FROM DENSE TO SPARSE MATRIX */
/*******************************/
void fromDenseToSparse(const int Nrows, const int Ncols, const float * __restrict__ d_A_dense, int * __restrict__ h_A_RowIndices,
	int * __restrict__ h_A_ColIndices, float * __restrict__ h_A, int *nnz) {

	// --- Descriptor for sparse matrix A
	setUpDescriptor(descrAMain, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ONE);

	const int lda = Nrows;                      // --- Leading dimension of dense matrix

	// --- Device side number of nonzero elements per row
	int *d_nnzPerVector;    gpuErrchk(cudaMalloc(&d_nnzPerVector, Nrows * sizeof(*d_nnzPerVector)));

	// --- Compute the number of nonzero elements per row and the total number of nonzero elements in the dense d_A_dense
	cusparseSafeCall(cusparseSnnz(handleMain, CUSPARSE_DIRECTION_ROW, Nrows, Ncols, descrAMain, d_A_dense, lda, d_nnzPerVector, nnz));

	// --- Host side number of nonzero elements per row
	int *h_nnzPerVector = (int *)malloc(Nrows * sizeof(*h_nnzPerVector));
	gpuErrchk(cudaMemcpy(h_nnzPerVector, d_nnzPerVector, Nrows * sizeof(*h_nnzPerVector), cudaMemcpyDeviceToHost));

	// --- Device side sparse matrix
	float *d_A;            gpuErrchk(cudaMalloc(&d_A, nnz[0] * sizeof(*d_A)));
	int *d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (Nrows + 1) * sizeof(*d_A_RowIndices)));
	int *d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnz[0] * sizeof(*d_A_ColIndices)));

	cusparseSafeCall(cusparseSdense2csr(handleMain, Nrows, Ncols, descrAMain, d_A_dense, lda, d_nnzPerVector, d_A, d_A_RowIndices, d_A_ColIndices));

	// --- Host side sparse matrix
	gpuErrchk(cudaMemcpy(h_A, d_A, nnz[0] * sizeof(*h_A), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (Nrows + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnz[0] * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));

}

/*****************************************************/
/* DISPLAYING THE CSR-FORMATTED SPARSE MATRIX SYSTEM */
/*****************************************************/
#ifdef VIDEO_DEBUG
int displayCSRSystemMatrix(const int * __restrict__ d_colIndices_CSR, const int * __restrict__ d_rowIndices_CSR,
	const float * __restrict__ d_KValues_CSR, const float * __restrict__ d_K, const int Nn, const int nnz) {

	// --- Moving the result to the host
	int *h_sequence = (int *)malloc((Nn + 1) * sizeof(int));
	int *h_colIndices_output = (int *)malloc(nnz		* sizeof(int));
	int *h_rowIndices_fromDense = (int *)malloc((Nn + 1) * sizeof(int));
	int *h_colIndices_fromDense = (int *)malloc(nnz		* sizeof(int));
	float *h_KValues_output = (float *)malloc(nnz		* sizeof(float));
	float *h_KValues_output_fromDense = (float *)malloc(nnz		* sizeof(float));

	gpuErrchk(cudaMemcpy(h_colIndices_output, d_colIndices_CSR, nnz	 * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_sequence, d_rowIndices_CSR, (Nn + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_KValues_output, d_KValues_CSR, nnz	 * sizeof(float), cudaMemcpyDeviceToHost));

	int nnzfromDenseToSparse;
	fromDenseToSparse(Nn, Nn, d_K, h_rowIndices_fromDense, h_colIndices_fromDense, h_KValues_output_fromDense, &nnzfromDenseToSparse);

	printf("\n\n");
	printf("Number of non-zero elements = %i \t Number of non-zero elements from dense matrix = %i\n", nnz, nnzfromDenseToSparse);
	if (nnz != nnzfromDenseToSparse) return 1;

	printf("\n\n");
	for (int i = 0; i < nnz; i++) printf("Col index = %i \t From sparse = %i\n", h_colIndices_output[i], h_colIndices_fromDense[i]);

	printf("\n\n");
	for (int i = 0; i < (Nn + 1); i++) printf("Row index = %i \t From sparse = %i\n", h_sequence[i], h_rowIndices_fromDense[i]);

	printf("\n\n");
	for (int i = 0; i < nnz; i++) printf("Matrix value = %f \t From sparse = %f\n", 1.e9 * h_KValues_output[i], 1.e9 * h_KValues_output_fromDense[i]);

	return 0;

}
#endif

/**********************************************/
/* COMPUTE SPARSE SYSTEM MATRIX IN CSR FORMAT */
/**********************************************/
void computeSparseSystemMatrixCSRFormat(const int * __restrict__ d_elementConnectivityMatrix, const float * __restrict__ d_Ke, int *nnz,
	int * d_rowIndices_CSR, int * __restrict__ d_colIndices_CSR,
	float * __restrict__ d_KValues_CSR, const int Nn, const int Ne) {

	// --- Compute all the contributions to the sparse system matrix along with the corresponding row and column indices
	int *d_rowIndices;		gpuErrchk(cudaMalloc(&d_rowIndices, 4 * Ne * sizeof(int)));
	int *d_colIndices;		gpuErrchk(cudaMalloc(&d_colIndices, 4 * Ne * sizeof(int)));

	float *d_KValues;		gpuErrchk(cudaMalloc(&d_KValues, 4 * Ne * sizeof(float)));

	computeGlobalMatrixPreprocessing << <iDivUp(Ne, BLOCKSIZE), BLOCKSIZE >> >(d_KValues, d_rowIndices, d_colIndices, d_elementConnectivityMatrix, d_Ke, Ne);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	int *d_rowIndices_temp;		gpuErrchk(cudaMalloc(&d_rowIndices_temp, 4 * Ne * sizeof(float)));

	// --- Sort the contributions to the sparse system matrix according to their respective row and column indices. Row indices first, then column
	//     indices.
	thrust::sort_by_key(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices),
																	 thrust::device_pointer_cast(d_colIndices))),
						thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices) + 4 * Ne,
																	 thrust::device_pointer_cast(d_colIndices) + 4 * Ne)),
						thrust::device_pointer_cast(d_KValues),
						TupleComp());

	// --- Sum up all the contributions to the sparse system matrix according to their respective row and column indices. Row and column indices
	//     are the key of the reduce_by_key routine.
	auto new_end = thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices),
																					  thrust::device_pointer_cast(d_colIndices))),
									     thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices) + 4 * Ne,
																					  thrust::device_pointer_cast(d_colIndices) + 4 * Ne)),
										 thrust::device_pointer_cast(d_KValues),
										 thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices_temp),
																					  thrust::device_pointer_cast(d_colIndices_CSR))),
										 thrust::device_pointer_cast(d_KValues_CSR),
										 BinaryPredicate(),
										 thrust::plus<float>());

	nnz[0] = new_end.first - thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices_temp),
																		  thrust::device_pointer_cast(d_colIndices_CSR)));

	// --- Convert the row indices to row indices for the CSR format
	thrust::reduce_by_key(thrust::device_pointer_cast(d_rowIndices_temp),
						  thrust::device_pointer_cast(d_rowIndices_temp) + nnz[0],
						  thrust::counting_iterator<int>(0),
						  thrust::device_pointer_cast(d_rowIndices_temp),
						  thrust::device_pointer_cast(d_rowIndices_CSR),
						  thrust::equal_to<int>(),
						  thrust::minimum<int>());

	// --- Setting the last element of the row matrix
	int lastElement;
	gpuErrchk(cudaMemcpy(&lastElement, d_rowIndices_CSR, sizeof(int), cudaMemcpyDeviceToHost));
	lastElement = lastElement + nnz[0];
	gpuErrchk(cudaMemcpy(d_rowIndices_CSR + Nn, &lastElement, sizeof(int), cudaMemcpyHostToDevice));

}

/************************************************************/
/* LAST STEP IN MATRIX AND COEFFICIENTS VECTOR CONSTRUCTION */
/************************************************************/
__global__ void lastStep(const int * __restrict__ d_A_RowIndices, const int * __restrict__ d_A_ColIndices, float * __restrict__ d_A,
	                     float * __restrict__ d_f, const float Va, const float Vb, const int Nn, const int nnz) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= nnz) return;

	if ((tid > 0) && (tid <= Nn - 1) && (d_A_ColIndices[d_A_RowIndices[tid]] == 0)) d_f[tid] = d_f[tid] - d_A[d_A_RowIndices[tid]] * Va;

	const int col_index = d_A_ColIndices[tid];
	if ((col_index == 0) || (col_index == (Nn - 1))) d_A[tid] = 0.f;

	if (tid == 0) {

		for (int k = d_A_RowIndices[tid]; k < d_A_RowIndices[tid + 1]; k++) d_A[k] = 0.f;
		for (int k = d_A_RowIndices[Nn - 1]; k < nnz; k++) d_A[k] = 0.f;

		d_A[0] = 1.f;
		d_A[nnz - 1] = 1.f;

		d_f[0] = Va;
	}

	if ((tid > 0) && (tid < Nn - 1) && (d_A_ColIndices[d_A_RowIndices[tid + 1] - 1] == (Nn - 1))) d_f[tid] = d_f[tid] - d_A[d_A_RowIndices[tid + 1] - 1] * Vb;

	if (tid == (Nn - 1)) d_f[tid] = Vb;
	
}

/********/
/* MAIN */
/********/
int main()
{
	/**************/
	/* PARAMETERS */
	/**************/
	const float d = 8e-2;									// --- Length of the investigation domain
	const float rho0 = 1e-8;								// --- Charge density
	const float epsr = 1.0f;								// --- Relative dielectric permittivity
	const float eps = epsr * 8.85 * 1e-12;					// --- Dielectric permittivity
	const float Va = 1.f;									// --- Boundary condition : voltage value at the leftmost node
	const float Vb = 0.f;									// --- Boundary condition : voltage value at the rightmost node
	const int   Ne = 6;										// --- Number of elements
	const int   Nn = Ne + 1;								// --- Number of nodes
	const int   numElementInterpolationPoints = 50;         // --- Number of interpolation points inside each element

	/***********************/
	/* INITIALIZE CUSPARSE */
	/***********************/
	cusparseSafeCall(cusparseCreate(&handleMain));

	/********************************/
	/* ELEMENT MATRICES AND VECTORS */
	/********************************/
	const float le = d / (float)Ne;							// --- Element length

	float *h_Ke = (float *)malloc(4 * sizeof(float));
	float *h_fe = (float *)malloc(2 * sizeof(float));

	h_Ke[0] = eps / le;
	h_Ke[1] = -eps / le;
	h_Ke[2] = -eps / le;
	h_Ke[3] = eps / le;

	h_fe[0] = -le * rho0 / 2.f;
	h_fe[1] = -le * rho0 / 2.f;

	float *d_Ke, *d_fe;
	gpuErrchk(cudaMalloc(&d_Ke, 4 * sizeof(float)));		gpuErrchk(cudaMemcpy(d_Ke, h_Ke, 4 * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&d_fe, 2 * sizeof(float)));		gpuErrchk(cudaMemcpy(d_fe, h_fe, 2 * sizeof(float), cudaMemcpyHostToDevice));

	/*******************************/
	/* ELEMENT CONNECTIVITY MATRIX */
	/*******************************/
	int *d_elementConnectivityMatrix;
	gpuErrchk(cudaMalloc(&d_elementConnectivityMatrix, 2 * Ne * sizeof(int)));

	computeConnectivityMatrix << <iDivUp(Ne, BLOCKSIZE), BLOCKSIZE >> >(d_elementConnectivityMatrix, Ne);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	/*********************************************/
	/* GLOBAL COEFFICIENT MATRIX - DENSE VERSION */
	/*********************************************/
	float *d_f;			gpuErrchk(cudaMalloc(&d_f, Nn      * sizeof(float)));
#ifndef VIDEO_DEBUG
	computefVectorAtomic << <iDivUp(Ne, BLOCKSIZE), BLOCKSIZE >> >(d_f, d_fe, d_elementConnectivityMatrix, Ne);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
#endif

#ifdef VIDEO_DEBUG
	float *d_K;	float *d_f;
	gpuErrchk(cudaMalloc(&d_K, Nn * Nn * sizeof(float)));

	float *h_K = (float *)malloc(Nn * Nn * sizeof(float));
	float *h_f = (float *)malloc(Nn      * sizeof(float));

	computeGlobalMatrixAtomic << <iDivUp(Ne, BLOCKSIZE), BLOCKSIZE >> >(d_K, d_f, d_fe, d_Ke, d_elementConnectivityMatrix, Ne, Nn);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// --- Video display of the dense version of K and of the data coefficients
	gpuErrchk(cudaMemcpy(h_K, d_K, Nn * Nn * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_f, d_f, Nn *      sizeof(float), cudaMemcpyDeviceToHost));

	printf("Sparse system matrix in dense format\n");
	for (int k = 0; k < Nn; k++) {
		for (int l = 0; l < Nn; l++) printf("%f \t", 1.e9 * h_K[k * Nn + l]);
		printf("\n");
	}
	printf("\n\nCoefficients vector\n");
	for (int k = 0; k < Nn; k++) printf("%f\n", 1e9 * h_f[k]);
#endif

	/**********************************************/
	/* GLOBAL COEFFICIENT MATRIX - SPARSE VERSION */
	/**********************************************/

	// --- Declare outputs
	int *d_colIndices_CSR;		gpuErrchk(cudaMalloc(&d_colIndices_CSR, 4 * Ne * sizeof(float)));
	int *d_rowIndices_CSR;		gpuErrchk(cudaMalloc(&d_rowIndices_CSR, (Nn + 1) * sizeof(int)));

	float *d_KValues_CSR;		gpuErrchk(cudaMalloc(&d_KValues_CSR, 4 * Ne * sizeof(float)));

	int nnz;					// --- Number of nonzero values in sparse system matrix

	computeSparseSystemMatrixCSRFormat(d_elementConnectivityMatrix, d_Ke, &nnz, d_rowIndices_CSR, d_colIndices_CSR, d_KValues_CSR, Nn, Ne);

	// --- Display the CSR-formatted sparse matrix system on the host
#ifdef VIDEO_DEBUG
	int returnValue = displayCSRSystemMatrix(d_colIndices_CSR, d_rowIndices_CSR, d_KValues_CSR, d_K, Nn, nnz);
#endif

	/************************************************************/
	/* LAST STEP IN MATRIX AND COEFFICIENTS VECTOR CONSTRUCTION */
	/************************************************************/

	lastStep << <iDivUp(nnz, BLOCKSIZE), BLOCKSIZE >> >(d_rowIndices_CSR, d_colIndices_CSR, d_KValues_CSR, d_f, Va, Vb, Nn, nnz);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	/**********************************/
	/* SOLVE THE SPARSE LINEAR SYSTEM */
	/**********************************/
	// --- Allocating the result vector
	float *d_y;        gpuErrchk(cudaMalloc(&d_y, Nn * sizeof(float)));

#ifdef LU_DECOMPOSITION
	// --- Pay attention. This routine modified the input system matrix d_KValues_CSR
	solveSparseLinearSystemLU(d_rowIndices_CSR, d_colIndices_CSR, d_KValues_CSR, d_f, d_y, nnz, Nn, CUSPARSE_INDEX_BASE_ZERO);
#endif

#ifdef PRECONDCONJUGATEGRADIENT
	int iterations = 0;
	precondConjugateGradientSparse(d_rowIndices_CSR, Nn + 1, d_colIndices_CSR, d_KValues_CSR, nnz, d_f, Nn, d_y, 0, iterations);
#endif

	float *h_y = (float *)malloc(Nn * sizeof(float));
	gpuErrchk(cudaMemcpy(h_y, d_y, Nn * sizeof(float), cudaMemcpyDeviceToHost));
	for (int k = 0; k < Nn; k++) printf("%f\n", h_y[k]);

#ifdef VIDEO_DEBUG
	return returnValue;
#else
	return 0;
#endif
}

