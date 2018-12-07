// --- CUDA Poisson solver using optimized red-black Gauss–Seidel with Successive OverRelaxation (SOR) solver

// --- Boundary conditions:
//     T = 0 at x = 0 (left boundary condition), x = L (right boundary condition), y = 0 (bottom boundary condition)
//     T = TN at y = H (upper boundary condition)

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iomanip>

#include <thrust\device_ptr.h>
#include <thrust\reduce.h>

#include "TimingGPU.cuh"

#include <cuda.h>

// --- Problem size along one direction
 //#define NUM 8192
#define NUM 512

#define prec_save 10

#define BLOCK_SIZE 128

// --- Double precision
#define DOUBLE

#ifdef DOUBLE
#define real double
#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0

// --- SOR parameter
#define omega	1.85
#else
#define real float
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f

// --- SOR parameter
#define omega	1.85f
#endif

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

/************************************/
/* SAVE REAL ARRAY FROM GPU TO FILE */
/************************************/
template <class T>
void saveGPUrealtxt(const T * d_in, const char *filename, const int M) {

	T *h_in = (T *)malloc(M * sizeof(T));

	gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));

	std::ofstream outfile;
	outfile.open(filename);
	for (int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n";
	outfile.close();

}

/******************************************************/
/* APPLICATION OF BOUNDARY CONDITIONS DEVICE FUNCTION */
/******************************************************/
__device__ void applyBC(real &aW, real &aE, real &aS, real &aN, real &b, real &SP, const real thermalConductivity, const real dx, const real dy, const real width, const real TN, const int rowGlobal, const int colGlobal, const int rowmax, const int colmax) {

	if (colGlobal == 0) {
		// --- Left boundary condition
		aW = ZERO;
		SP = -TWO * thermalConductivity * width * dy / dx;
	}
	else {
		aW = thermalConductivity * width * dy / dx;
	}

	if (colGlobal == (colmax - 1)) {
		// --- Right boundary condition
		aE = ZERO;
		SP = -TWO * thermalConductivity * width * dy / dx;
	}
	else {
		aE = thermalConductivity * width * dy / dx;
	}

	if (rowGlobal == 0) {
		// --- Bottom boundary condition
		aS = ZERO;
		SP = -TWO * thermalConductivity * width * dx / dy;
	}
	else {
		aS = thermalConductivity * width * dx / dy;
	}

	if (rowGlobal == (rowmax - 1)) {
		// --- Top boundary condition
		aN = ZERO;
		b = TWO * thermalConductivity * width * dx * TN / dy;
		SP = -TWO * thermalConductivity * width * dx / dy;
	}
	else {
		aN = thermalConductivity * width * dx / dy;
	}
}

/**************/
/* RED KERNEL */
/**************/
__global__ void redKernel(const real * d_Tblack, real * d_Tred,	real * L2norm, real thermalConductivity, real dx, real dy, real width, real TN, const int rowmax, const int colmax) {
	
	int row = 1 + (blockIdx.x * blockDim.x) + threadIdx.x;
	int col = 1 + (blockIdx.y * blockDim.y) + threadIdx.y;

	int ind_red = col * ((NUM >> 1) + 2) + row;  					// --- Local red index
	real d_Told = d_Tred[ind_red];

	int rowGlobal = 2 * row - (col & 1) - 1;
	int colGlobal = col - 1;

	real b = ZERO, aW, aE, aS, aN, aP, SP = ZERO;

	applyBC(aW, aE, aS, aN, b, SP, thermalConductivity, dx, dy, width, TN, rowGlobal, colGlobal, rowmax, colmax);
		
	aP = aW + aE + aS + aN - SP;

	real res = b
		+ (aW * d_Tblack[row + (col - 1) * ((NUM >> 1) + 2)]
		+  aE * d_Tblack[row + (col + 1) * ((NUM >> 1) + 2)]
		+  aS * d_Tblack[row - (col & 1) + col * ((NUM >> 1) + 2)]
		+  aN * d_Tblack[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)]);

	real temp_new = d_Told * (ONE - omega) + omega * (res / aP);

	d_Tred[ind_red] = temp_new;
	res = temp_new - d_Told;

	L2norm[ind_red] = res * res;
} 

/****************/
/* BLACK KERNEL */
/****************/
__global__ void blackKernel(const real * d_Tred, real * d_Tblack, real * L2norm, real thermalConductivity, real dx, real dy, real width, real TN, const int rowmax, const int colmax)
{
	int row = 1 + (blockIdx.x * blockDim.x) + threadIdx.x;
	int col = 1 + (blockIdx.y * blockDim.y) + threadIdx.y;
	
	int ind_black = col * ((NUM >> 1) + 2) + row;  						// --- Local black index
	real d_Told = d_Tblack[ind_black];

	int rowGlobal = 2 * row - ((col + 1) & 1) - 1;
	int colGlobal = col - 1;
	
	real b = ZERO, aW, aE, aS, aN, aP, SP = ZERO;
	applyBC(aW, aE, aS, aN, b, SP, thermalConductivity, dx, dy, width, TN, rowGlobal, colGlobal, rowmax, colmax);
	aP = aW + aE + aS + aN - SP;

	real res = b
		+ (aW * d_Tred[row + (col - 1) * ((NUM >> 1) + 2)]
		+ aE * d_Tred[row + (col + 1) * ((NUM >> 1) + 2)]
		+ aS * d_Tred[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)]
		+ aN * d_Tred[row + (col & 1) + col * ((NUM >> 1) + 2)]);
	
	real temp_new = d_Told * (ONE - omega) + omega * (res / aP);

	d_Tblack[ind_black] = temp_new;
	res = temp_new - d_Told;

	L2norm[ind_black] = res * res;
} 

/****************************************/
/* REARRANGE RED AND BLACK CELLS KERNEL */
/****************************************/
__global__ void rearrangeKernel(const real * d_Tred, const real * d_Tblack, real *d_T)
{
	int row = 1 + (blockIdx.x * blockDim.x) + threadIdx.x;
	int col = 1 + (blockIdx.y * blockDim.y) + threadIdx.y;

	int ind_red			= col * ((NUM >> 1) + 2) + row;  							// --- Local red index
	int globalInd_red	= 2 * row - (col & 1) - 1 + NUM * (col - 1);				// --- Global red index
	int ind_black		= col * ((NUM >> 1) + 2) + row;  							// --- Local black index
	int globalInd_black = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1);			// --- Global index

	d_T[globalInd_red]		= d_Tred[ind_red];
	d_T[globalInd_black]	= d_Tblack[ind_black];
}

/********/
/* MAIN */
/********/
int main(void) {

	real L						= 1.0;										// --- Plate length
	real H						= 1.0;										// --- Plate height
	real width					= 0.01;										// --- Plate width

	real thermalConductivity	= 1.0;										// --- Thermal conductivity

	real TN						= 1.0;										// --- Temperature at top boundary

	real tol = 1.e-6;														// --- SOR iteration tolerance

	int Nrows = (NUM / 2) + 2;												// --- Number of cells in x direction including boundary cells
	int Ncols = NUM + 2;													// --- Number of cells in y direction including boundary cells

	real dx = L / NUM;														// --- Discretization step along the x-axis
	real dy = H / NUM;														// --- Discretization step along the y-axis

	int iter;																// --- Iterations counter
	int maxIter = 1e6;														// --- Maximum number of SOR iterations

	TimingGPU timerGPU;
	timerGPU.StartCounter();

	// --- Device memory temperature allocation
	real *d_T; gpuErrchk(cudaMalloc(&d_T, (NUM) * (NUM) * sizeof(real)));

	// --- Device memory allocation and initialization for red and black temperature arrays 
	real *d_Tred;		gpuErrchk(cudaMalloc(&d_Tred, Nrows * Ncols * sizeof(real)));
	real *d_Tblack;		gpuErrchk(cudaMalloc(&d_Tblack, Nrows * Ncols * sizeof(real)));
	gpuErrchk(cudaMemset(d_Tred,	0, Nrows * Ncols * sizeof(real)));
	gpuErrchk(cudaMemset(d_Tblack,	0, Nrows * Ncols * sizeof(real)));

	dim3 dimBlock(BLOCK_SIZE, 1);
	dim3 dimGrid(NUM / (2 * BLOCK_SIZE), NUM);

	// --- Device residuals for both red and black temperature points
	real *d_L2norm_pointwise; gpuErrchk(cudaMalloc(&d_L2norm_pointwise, Nrows * Ncols * sizeof(real)));
	gpuErrchk(cudaMemset(d_L2norm_pointwise, 0, Nrows * Ncols * sizeof(real)));

	// --- SOR loop
	for (iter = 1; iter <= maxIter; ++iter) {

		real L2norm = ZERO;

		// --- Update red cells
		redKernel << <dimGrid, dimBlock >> > (d_Tblack, d_Tred, d_L2norm_pointwise, thermalConductivity, dx, dy, width, TN, NUM, NUM);

		// --- Compute squared L2 norm (first part for red cells)
		L2norm = thrust::reduce(thrust::device_pointer_cast(d_L2norm_pointwise), thrust::device_pointer_cast(d_L2norm_pointwise) + Nrows * Ncols); 

		// --- Update black cells
		blackKernel << <dimGrid, dimBlock >> > (d_Tred, d_Tblack, d_L2norm_pointwise, thermalConductivity, dx, dy, width, TN, NUM, NUM);

		// --- Compute squared L2 norm (second part for red cells)
		L2norm = L2norm + thrust::reduce(thrust::device_pointer_cast(d_L2norm_pointwise), thrust::device_pointer_cast(d_L2norm_pointwise) + Nrows * Ncols);

		// --- Compute residual
		L2norm = sqrt(L2norm / ((real)NUM * NUM));

		if (iter % 100 == 0) printf("Iteration number: %5d; L2 norm: %0.6f\n", iter, L2norm);

		// --- Tolerance check
		if (L2norm < tol) break;
	}

	double runtime = timerGPU.GetCounter();

	printf("GPU\n");
	printf("Iterations: %i\n", iter);
	printf("Total time: %f s\n", runtime / 1000.0);

	// --- Rearranging red and black in a full temperature matrix
	rearrangeKernel << <dimGrid, dimBlock >> > (d_Tred, d_Tblack, d_T);

	// --- Saving the results
	saveGPUrealtxt(d_Tred, ".\\redMatrix.txt", Nrows * Ncols);
	saveGPUrealtxt(d_Tblack, ".\\blackMatrix.txt", Nrows * Ncols);
	saveGPUrealtxt(d_T, ".\\fullMatrix.txt", (NUM)* (NUM));

	// --- Free device memory
	gpuErrchk(cudaFree(d_Tred));
	gpuErrchk(cudaFree(d_Tblack));
	gpuErrchk(cudaFree(d_L2norm_pointwise));

	gpuErrchk(cudaDeviceReset());

	return 0;
}
