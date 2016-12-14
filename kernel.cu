#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cusparse.h>

#include "Utilities.cuh"
#include "Matlab_like.cuh"
#include "cusparseLU.cuh"
#include "precondConjugateGradientSparse.cuh"

// --- Defining tuple type
typedef thrust::tuple<int, int> Tuple;

#define PI_d			3.141592653589793

#define DEBUG

#define VIDEO_DEBUG

#define BLOCKSIZE		256

/**************************************/
/* ELEMENT CONNECTIVITY MATRIX KERNEL */
/**************************************/
__global__ void computeConnectivityMatrix(int * __restrict__ d_elementConnectivityMatrix, const int numElements, const int numNodesPerElement)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= numElements) return;

	d_elementConnectivityMatrix[tid * numNodesPerElement * numElements]		= tid;
	d_elementConnectivityMatrix[tid * numNodesPerElement * numElements + 1] = tid + 1;

}

/***********************/
/* GAUSSIAN QUADRATURE */
/***********************/
void gaussianQuadrature(const int ngp, double * __restrict h_gaussPoints, double * __restrict h_gaussWeights) {

	// --- 1 Gauss point
	if (ngp == 1) {
		h_gaussPoints[0] = 0;
		h_gaussWeights[0] = 2;
	}

	// --- 2 Gauss points
	else if (ngp == 2) {
		h_gaussPoints[0] = -0.577350269189625764509148780502;
		h_gaussPoints[1] =  0.577350269189625764509148780502;

		h_gaussWeights[0] = 1.0;
		h_gaussWeights[1] = 1.0;
	}

	// --- 3 Gauss points
	else if (ngp == 3) {
		h_gaussPoints[0] = -0.774596669241483377035853079956;
		h_gaussPoints[1] = 0.0;
		h_gaussPoints[2] = 0.774596669241483377035853079956;

		h_gaussWeights[0] = 5.0 / 9.0;
		h_gaussWeights[1] = 8.0 / 9.0;
		h_gaussWeights[2] = 5.0 / 9.0;
	}

	// --- 4 Gauss points
	else if (ngp == 4) {
		h_gaussPoints[0] = -0.861136311594052575223946488893;
		h_gaussPoints[1] = -0.339981043584856264802665759103;
		h_gaussPoints[2] = 0.339981043584856264802665759103;
		h_gaussPoints[3] = 0.861136311594052575223946488893;

		h_gaussWeights[0] = 0.347854845137453857373063949222;
		h_gaussWeights[1] = 0.652145154862546142626936050778;
		h_gaussWeights[2] = 0.652145154862546142626936050778;
		h_gaussWeights[3] = 0.347854845137453857373063949222;
	}

	// --- 5 Gauss points
	else if (ngp == 5) {
		h_gaussPoints[0] = -0.906179845938663992797626878299;
		h_gaussPoints[1] = -0.538469310105683091036314420700;
		h_gaussPoints[2] = 0.0;
		h_gaussPoints[3] = 0.538469310105683091036314420700;
		h_gaussPoints[4] = 0.906179845938663992797626878299;

		h_gaussWeights[0] = 0.236926885056189087514264040720;
		h_gaussWeights[1] = 0.478628670499366468041291514836;
		h_gaussWeights[2] = 0.568888888888888888888888888889;
		h_gaussWeights[3] = 0.478628670499366468041291514836;
		h_gaussWeights[4] = 0.236926885056189087514264040720;
	}

	else {
		printf("\nGaussian quadrature - Fatal error!\n");
		printf("Illegal number of Gauss points = %d\n", ngp);
		printf("Legal values are 1 to 5\n");
	}
}

__device__ void LinearBasisFunctions1D(const double xi, double * __restrict__ Ne1, double * __restrict__ Ne2, double * __restrict__ dNe1, double * __restrict__ dNe2) {

	// --- Calculate the two basis functions at the natural coordinate xi
	Ne1[0] = .5 * (1. - xi);
	Ne2[0] = .5 * (xi + 1.);

	// --- Calculate the derivatives of the basis function with respect to natural coordinate xi
	dNe1[0] = -.5;
	dNe2[0] =  .5;

}

__device__ double ff(double x) { return (16. * PI_d * PI_d + 1) * sin(4. * PI_d * x); }

__global__ void assembleGlobalMatrix(const int * __restrict__ d_connectivityMatrix, const double * __restrict__ d_globalNodes, 
							         const double * __restrict__ d_gaussPoints, const double * __restrict__ d_gaussWeights,
									 double * __restrict__ d_f, int * __restrict__ d_I, int * __restrict__ d_J, double * __restrict__ d_X, 
									 const int numElements, const int numNodesPerElement, const int numberGaussPoints) {

	const int e = threadIdx.x + blockIdx.x * blockDim.x;

	if (e >= numElements) return;

	// --- Remember:
	//     1) Dynamically indexed arrays cannot be stored in registers, because the GPU register file is not dynamically addressable.
	//     2) Scalar variables are automatically stored in registers by the compiler.
	//     3) Statically - indexed(i.e.where the index can be determined at compile time), small arrays(say, less than 16 floats) may be stored in registers by the compiler.
	double Ke[4] = {0., 0., 0., 0.};										// --- Element stiffness matrix
	double fe[2] = {0., 0.};												// --- Element force(load) vector

	// --- Global node numbers corresponding to the current element
	int globalNodeNumbers[2];
	globalNodeNumbers[0] = d_connectivityMatrix[e * numNodesPerElement * numElements];
	globalNodeNumbers[1] = d_connectivityMatrix[e * numNodesPerElement * numElements + 1];

	//printf("Element = %i; globalNodeNumbers[0] = %i; globalNodeNumbers[1] = %i\n", e, d_connectivityMatrix[e * numNodesPerElement * numElements], d_connectivityMatrix[e * numNodesPerElement * numElements + 1]);
	
	// --- Global coordinates of the element nodes as a column vector
	double xe[2];
	xe[0] = d_globalNodes[globalNodeNumbers[0]];
	xe[1] = d_globalNodes[globalNodeNumbers[1]];

	//for (int k = 0; k < 6; k++) printf("%i %f\n", k, d_globalNodes[k]);

	//printf("Element = %i; xe[0] = %f; xe[1] = %f\n", e, xe[0], xe[1]);

	double Ne1, Ne2, dNe1, dNe2, B1, B2;
	
	double x, Jacobian, JacxW;
	
	// --- Calculate the element integral
	for (int k = 0; k < numberGaussPoints; k++) {						// --- Loop over all the Gauss points

		// --- Return the 1D linear basis functions along with their derivatives with respect to local coordinates xi at Gauss points
		LinearBasisFunctions1D(d_gaussPoints[k], &Ne1, &Ne2, &dNe1, &dNe2);

		x = Ne1 * xe[0] + Ne2 * xe[1];                                  // --- Global coordinate (here x) of the current integration point

		//printf("Element %i; Gauss point %i; x = %f\n", e, k, x);

		Jacobian = dNe1 * xe[0] + dNe2 * xe[1];							// --- Jacobian dx / dxi

		JacxW = Jacobian * d_gaussWeights[k];							// --- Calculate the integration weight

		// --- Calculate the derivatives of the basis functions with respect to x direction
		B1 = dNe1 / Jacobian;                                
		B2 = dNe2 / Jacobian;                                

		Ke[0] = Ke[0] + (B1* B1) * JacxW;
		Ke[1] = Ke[1] + (B1* B2) * JacxW;
		Ke[2] = Ke[2] + (B2* B1) * JacxW;
		Ke[3] = Ke[3] + (B2* B2) * JacxW;

		fe[0] = fe[0] + ff(x) * Ne1 * JacxW;
		fe[1] = fe[1] + ff(x) * Ne2 * JacxW;

	}

	int globalIndexii, globalIndexjj;
	// --- Loop over all the nodes of the e - th element
	for (int ii = 0; ii < numNodesPerElement; ii++) {
		
		// --- Global index of the ii - th local node of the e - th element
		globalIndexii = d_connectivityMatrix[e * numNodesPerElement * numElements + ii];

		//printf("Element %i; node %i; globalIndexii %i\n", e, ii, globalIndexii);

		if (fe[ii] != 0) atomicAdd(&d_f[globalNodeNumbers[ii]], fe[ii]);		// --- Assemble load

		// --- Loop over all the nodes of the e - th element
		for (int jj = 0; jj < numNodesPerElement; jj++) {

			// --- Global index of the ii - th local node of the e - th element
			globalIndexjj = d_connectivityMatrix[e * numNodesPerElement * numElements + jj];

			// --- If the element(ii, jj) of the stiffness matrix of the e - th element is different from zero, then add a triplet
			if (Ke[ii * numNodesPerElement + jj] != 0) {

				d_I[e * 4 + ii * numNodesPerElement + jj] = globalIndexii;
				d_J[e * 4 + ii * numNodesPerElement + jj] = globalIndexjj;
				d_X[e * 4 + ii * numNodesPerElement + jj] = Ke[ii * numNodesPerElement + jj];

				//printf("Element %i ii %i jj %i I %i J %i K %f\n", e, ii, jj, d_I[e * 4 + ii * numNodesPerElement + jj], d_J[e * 4 + ii * numNodesPerElement + jj], d_X[e * 4 + ii * numNodesPerElement + jj]);
				//printf("I %i J %i K %f\n", d_I[e * 4 + ii * numNodesPerElement + jj], d_J[e * 4 + ii * numNodesPerElement + jj], d_X[e * 4 + ii * numNodesPerElement + jj]);

			}

		}
	}
}

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
/* GLOBAL MATRIX PREPROCESSING KERNEL */
/**************************************/
template<class T>
__global__ void computeGlobalMatrixPreprocessing(T * __restrict__ d_KValues, int * __restrict__ d_rowIndices, int * __restrict__ d_colIndices, const int * __restrict__ d_elementConnectivityMatrix, 
												 const T * __restrict__ d_Ke, const int Ne) {

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

/**********************************************/
/* COMPUTE SPARSE SYSTEM MATRIX IN CSR FORMAT */
/**********************************************/
template<class T>
void computeSparseSystemMatrixCSRFormat(const int * __restrict__ d_elementConnectivityMatrix, const T * __restrict__ d_Ke, int *nnz, int * d_rowIndices_CSR, int * __restrict__ d_colIndices_CSR,
										T * __restrict__ d_KValues_CSR, const int totalNumNodes, const int numElements) {

	// --- Compute all the contributions to the sparse system matrix along with the corresponding row and column indices
	int *d_rowIndices;		gpuErrchk(cudaMalloc(&d_rowIndices, 4 * numElements * sizeof(int)));
	int *d_colIndices;		gpuErrchk(cudaMalloc(&d_colIndices, 4 * numElements * sizeof(int)));

	T *d_KValues;		gpuErrchk(cudaMalloc(&d_KValues, 4 * numElements * sizeof(T)));

	computeGlobalMatrixPreprocessing << <iDivUp(numElements, BLOCKSIZE), BLOCKSIZE >> >(d_KValues, d_rowIndices, d_colIndices, d_elementConnectivityMatrix, d_Ke, numElements);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	int *d_rowIndices_temp;		gpuErrchk(cudaMalloc(&d_rowIndices_temp, 4 * numElements * sizeof(T)));

	// --- Sort the contributions to the sparse system matrix according to their respective row and column indices. Row indices first, then column
	//     indices.
	thrust::sort_by_key(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices),
		thrust::device_pointer_cast(d_colIndices))),
		thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices) + 4 * numElements,
		thrust::device_pointer_cast(d_colIndices) + 4 * numElements)),
		thrust::device_pointer_cast(d_KValues),
		TupleComp());

	//d_rowIndices = d_rowIndices + 3;
	//d_colIndices = d_colIndices + 3;
	//d_KValues    = d_KValues + 3;

	//int *h_rowIndices = (int *)malloc((4 * numElements - 3) * sizeof(int));
	//int *h_colIndices = (int *)malloc((4 * numElements - 3) * sizeof(int));

	//T *h_KValues = (T *)malloc((4 * numElements - 3) * sizeof(T));

	//gpuErrchk(cudaMemcpy(h_rowIndices, d_rowIndices, (4 * numElements - 3) * sizeof(int), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_colIndices, d_colIndices, (4 * numElements - 3) * sizeof(int), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_KValues, d_KValues, (4 * numElements - 3) * sizeof(T), cudaMemcpyDeviceToHost));

	//printf("Parte prima\n");
	//for (int k = 0; k < (4 * numElements - 3); k++) printf("%i %i %f\n", h_rowIndices[k], h_colIndices[k], h_KValues[k]);

	// --- Sum up all the contributions to the sparse system matrix according to their respective row and column indices. Row and column indices
	//     are the key of the reduce_by_key routine.
	auto new_end = thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices),
		thrust::device_pointer_cast(d_colIndices))),
		thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices) + 4 * numElements,
		thrust::device_pointer_cast(d_colIndices) + 4 * numElements)),
		thrust::device_pointer_cast(d_KValues),
		thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices_temp),
		thrust::device_pointer_cast(d_colIndices_CSR))),
		thrust::device_pointer_cast(d_KValues_CSR),
		BinaryPredicate(),
		thrust::plus<T>());

	//auto new_end = thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices),
	//	thrust::device_pointer_cast(d_colIndices))),
	//	thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices) + 4 * numElements - 3,
	//	thrust::device_pointer_cast(d_colIndices) + 4 * numElements - 3)),
	//	thrust::device_pointer_cast(d_KValues),
	//	thrust::make_zip_iterator(thrust::make_tuple(thrust::device_pointer_cast(d_rowIndices_temp),
	//	thrust::device_pointer_cast(d_colIndices_CSR))),
	//	thrust::device_pointer_cast(d_KValues_CSR),
	//	BinaryPredicate(),
	//	thrust::plus<T>());

	//gpuErrchk(cudaMemcpy(h_rowIndices, d_rowIndices, (4 * numElements - 3) * sizeof(int), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_colIndices, d_colIndices, (4 * numElements - 3) * sizeof(int), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_KValues, d_KValues, (4 * numElements - 3) * sizeof(T), cudaMemcpyDeviceToHost));

	//printf("Parte seconda\n");
	//for (int k = 0; k < (4 * numElements - 3); k++) printf("%i %i %f\n", h_rowIndices[k], h_colIndices[k], h_KValues[k]);

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
	gpuErrchk(cudaMemcpy(d_rowIndices_CSR + totalNumNodes, &lastElement, sizeof(int), cudaMemcpyHostToDevice));

	int *h_rowIndices_CSR = (int *)malloc((4 * numElements - 3) * sizeof(int));
	int *h_colIndices_CSR = (int *)malloc((4 * numElements - 3) * sizeof(int));

	T *h_KValues_CSR = (T *)malloc((4 * numElements - 3) * sizeof(T));

	gpuErrchk(cudaMemcpy(h_rowIndices_CSR, d_rowIndices_CSR, (totalNumNodes + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_colIndices, d_colIndices_CSR, (4 * numElements - 3) * sizeof(int), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_KValues, d_KValues_CSR, (4 * numElements - 3) * sizeof(T), cudaMemcpyDeviceToHost));

	//for (int k = 0; k < (4 * numElements - 3); k++) printf("%i %i %f\n", h_rowIndices[k], h_colIndices[k], h_KValues[k]);
	for (int k = 0; k < totalNumNodes + 1; k++) printf("CSR %i\n", h_rowIndices_CSR[k]);

}

__global__ void leftBoundaryCondition(double * __restrict__ d_f, double * __restrict__ d_KValues_CSR, const int totalNumNodes) {

	double value = 4. * PI_d * cos(4. * PI_d);
	
	d_f[0] = 0.;
	d_f[totalNumNodes - 1] = d_f[totalNumNodes - 1] + value;

	d_KValues_CSR[0] = 1.;
	d_KValues_CSR[1] = 0.;
	d_KValues_CSR[2] = 0.;

}

/********/
/* MAIN */
/********/
int main() {
	
	const double	x0					= 0.0;									// --- Left end of the 1D domain
	const double	x1					= 1.0;									// --- Right end of the 1D domain

	const int		numElements			= 7;									// --- Total number of elements

	const int		totalNumNodes		= numElements + 1;						// --- Total number of nodes
	const int		numNodesPerElement	= 2;									// --- Number of nodes per element

	const int		numberGaussPoints	= 2;									// --- Number of Gauss points
		
	double *d_globalNodes = d_linspace(x0, x1, totalNumNodes);					// --- Global node coordinates
	//double *h_globalNodes = (double *)malloc(totalNumNodes * sizeof(double));
	//gpuErrchk(cudaMemcpy(h_globalNodes, d_globalNodes, totalNumNodes * sizeof(double), cudaMemcpyDeviceToHost));

	//for (int k = 0; k < totalNumNodes; k++) printf("%i %f\n", k, h_globalNodes[k]);
	
	/***************************************/
	/* COMPUTE ELEMENT CONNECTIVITY MATRIX */
	/***************************************/
	// --- global node number = connectivityMatrix(local node number, element number)
	//     d_elementConnectivityMatrix is a (numElements x numNodesPerElement) matrix, stored rowwise
	int *d_elementConnectivityMatrix;		gpuErrchk(cudaMalloc((void**)&d_elementConnectivityMatrix, numNodesPerElement * numElements * sizeof(int)));
	gpuErrchk(cudaMemset(d_elementConnectivityMatrix, 0, numNodesPerElement * numElements * sizeof(int)));

	computeConnectivityMatrix << <iDivUp(numElements, BLOCKSIZE), BLOCKSIZE >> >(d_elementConnectivityMatrix, numElements, numNodesPerElement);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	/********************************/
	/* VECTOR (I, J, K) DEFINITIONS */
	/********************************/
	// --- Global force vector
	double *d_f;	gpuErrchk(cudaMalloc((void**)&d_f, totalNumNodes * sizeof(double)));		gpuErrchk(cudaMemset(d_f, 0, totalNumNodes * sizeof(double)));
	// --- Global solution vector
	double *d_d;	gpuErrchk(cudaMalloc((void**)&d_d, totalNumNodes * sizeof(double)));		gpuErrchk(cudaMemset(d_d, 0, totalNumNodes * sizeof(double)));

	// --- Triplet for assembling the sparse matrix
	// --- Row indices of non - zero entries
	int *d_I;		gpuErrchk(cudaMalloc((void**)&d_I, numNodesPerElement * totalNumNodes * sizeof(int)));		gpuErrchk(cudaMemset(d_I, 0, numNodesPerElement * totalNumNodes * sizeof(int)));
	// --- Column indices of non - zero entries
	int *d_J;		gpuErrchk(cudaMalloc((void**)&d_J, numNodesPerElement * totalNumNodes * sizeof(int)));		gpuErrchk(cudaMemset(d_J, 0, numNodesPerElement * totalNumNodes * sizeof(int)));
	// --- Non - zero entries matrix	
	double *d_X;	gpuErrchk(cudaMalloc((void**)&d_X, numNodesPerElement * totalNumNodes * sizeof(double)));	gpuErrchk(cudaMemset(d_X, 0, numNodesPerElement * totalNumNodes * sizeof(double)));

	/**************************************/
	/* COMPUTE GAUSSIAN QUADRATURE POINTS */
	/**************************************/
	double *h_gaussPoints = (double *)malloc(numberGaussPoints * sizeof(double));
	double *h_gaussWeights	= (double *)malloc(numberGaussPoints * sizeof(double));
	double *d_gaussPoints;		gpuErrchk(cudaMalloc((void**)&d_gaussPoints,  numberGaussPoints * sizeof(double)));
	double *d_gaussWeights;		gpuErrchk(cudaMalloc((void**)&d_gaussWeights, numberGaussPoints * sizeof(double)));
	gaussianQuadrature(numberGaussPoints, h_gaussPoints, h_gaussWeights);				// --- Return Gauss quadrature points and weights
	gpuErrchk(cudaMemcpy(d_gaussPoints, h_gaussPoints, numberGaussPoints * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_gaussWeights, h_gaussWeights, numberGaussPoints * sizeof(double), cudaMemcpyHostToDevice));

	/************************************/
	/* ASSEMBLE GLOBAL STIFFNESS MATRIX */
	/************************************/
	assembleGlobalMatrix << <iDivUp(numElements, BLOCKSIZE), BLOCKSIZE >> >(d_elementConnectivityMatrix, d_globalNodes, d_gaussPoints, d_gaussWeights,
		d_f, d_I, d_J, d_X, numElements, numNodesPerElement, numberGaussPoints);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	/********************************************/
	/* GLOBAL STIFFNESS MATRIX - SPARSE VERSION */
	/********************************************/
	// --- Declare outputs
	int *d_colIndices_CSR;		gpuErrchk(cudaMalloc(&d_colIndices_CSR, 4 * numElements * sizeof(double)));
	int *d_rowIndices_CSR;		gpuErrchk(cudaMalloc(&d_rowIndices_CSR, (totalNumNodes + 1) * sizeof(int)));

	double *d_KValues_CSR;		gpuErrchk(cudaMalloc(&d_KValues_CSR, 4 * numElements * sizeof(double)));

	int nnz;					// --- Number of nonzero values in sparse system matrix

	computeSparseSystemMatrixCSRFormat(d_elementConnectivityMatrix, d_X, &nnz, d_rowIndices_CSR, d_colIndices_CSR, d_KValues_CSR, totalNumNodes, numElements);

	/*********************************/
	/* ENFORCING BOUNDARY CONDITIONS */
	/*********************************/
	//double *d_KValues_CSR_reduced;		gpuErrchk(cudaMalloc(&d_KValues_CSR_reduced, 4 * numElements * sizeof(double)));
	//gpuErrchk(cudaMemcpy(d_KValues_CSR_reduced, d_KValues_CSR, 4 * numElements * sizeof(double), cudaMemcpyDeviceToDevice));

	leftBoundaryCondition << <1, 1 >> >(d_f, d_KValues_CSR, totalNumNodes);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif


#ifdef VIDEO_DEBUG
	cusparseMatDescr_t	descrK = 0;
	double *d_K;		gpuErrchk(cudaMalloc((void**)&d_K, totalNumNodes * totalNumNodes * sizeof(double)));
	double *h_K = (double *)malloc(totalNumNodes * totalNumNodes * sizeof(double));
	cusparseHandle_t	cusparseHandle;
	cusparseSafeCall(cusparseCreate(&cusparseHandle));
	// --- Descriptor for sparse matrix A
	setUpDescriptor(descrK, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSafeCall(cusparseDcsr2dense(cusparseHandle, totalNumNodes, totalNumNodes, descrK, d_KValues_CSR, d_rowIndices_CSR, d_colIndices_CSR, d_K, totalNumNodes));
	gpuErrchk(cudaMemcpy(h_K, d_K, totalNumNodes * totalNumNodes * sizeof(double), cudaMemcpyDeviceToHost));
	for (int k = 0; k < totalNumNodes; k++)
		for (int l = 0; l < totalNumNodes; l++) printf("k = %i; l = %i; K = %f\n", k, l, h_K[k * totalNumNodes + l]);
	double *h_f = (double *)malloc(totalNumNodes * sizeof(double));
	gpuErrchk(cudaMemcpy(h_f, d_f, totalNumNodes * sizeof(double), cudaMemcpyDeviceToHost));
	printf("\n\n");
	for (int k = 0; k < totalNumNodes; k++) printf("i = %i; f = %f\n", k, h_f[k]);
#endif

	/**********************************/
	/* SOLVE THE SPARSE LINEAR SYSTEM */
	/**********************************/

	// --- Allocating the result vector
	double *d_y;        gpuErrchk(cudaMalloc(&d_y, totalNumNodes * sizeof(double)));
	// --- Pay attention. This routine modifies the input system matrix d_KValues_CSR
	solveSparseLinearSystemLU(d_rowIndices_CSR, d_colIndices_CSR, d_KValues_CSR, d_f, d_y, nnz, totalNumNodes, CUSPARSE_INDEX_BASE_ZERO);

	double *h_y = (double *)malloc(totalNumNodes * sizeof(double));
	gpuErrchk(cudaMemcpy(h_y, d_y, totalNumNodes * sizeof(double), cudaMemcpyDeviceToHost));

	for (int k = 0; k < totalNumNodes; k++) printf("%f\n", h_y[k]);

	// --- Allocating the result vector
	//double *d_y;        gpuErrchk(cudaMalloc(&d_y, totalNumNodes * sizeof(double)));
	//double *h_y = (double *)malloc(totalNumNodes * sizeof(double));

	//int iterations = 0;
	//precondConjugateGradientSparse(d_rowIndices_CSR, totalNumNodes + 1, d_colIndices_CSR, d_KValues_CSR, nnz, d_f, totalNumNodes, d_y, 0, iterations);

	//gpuErrchk(cudaMemcpy(h_y, d_y, totalNumNodes * sizeof(double), cudaMemcpyDeviceToHost));
	//printf("Num iterations %i\n", iterations);
	//for (int k = 0; k < totalNumNodes; k++) printf("%f\n", h_y[k]);

	return 0;

}
