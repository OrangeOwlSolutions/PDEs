#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include "Utilities.cuh"
#include "InputOutput.h"
#include "TimingGPU.cuh"

// --- Problem size along one size. The computational domain is squared. NUM x NUM is the number of points in the
//     interior of the computational domain.
#define NUM			1024

#define BLOCKSIZEX	16
#define BLOCKSIZEY	16

// --- Double precision
#define DOUBLE

#ifdef DOUBLE
#define Real double
#define ZERO	0.0
#define ONE		1.0
#define TWO		2.0
#define FOUR	4.0

// --- SOR relaxation parameter
const Real omega = 1.85;
#else
#define Real	float
#define ZERO	0.0f
#define ONE		1.0f
#define TWO		2.0f
#define FOUR	4.0f

// --- SOR relaxation parameter
const Real omega = 1.85f;
#endif

// --- Split temperature into red and black arrays 
//#define MEMOPT

// --- Use texture memory
//#define TEXTURE

#ifdef TEXTURE
#ifdef DOUBLE
texture<int2, 1> t_aP;
texture<int2, 1> t_aW;
texture<int2, 1> t_aE;
texture<int2, 1> t_aS;
texture<int2, 1> t_aN;
texture<int2, 1> t_b;

static __inline__ __device__ double texFetch(texture<int2, 1> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
	return __hiloint2double(v.y, v.x);
}
#else
texture<float> t_aP;
texture<float> t_aW;
texture<float> t_aE;
texture<float> t_aS;
texture<float> t_aN;
texture<float> t_b;

static __inline__ __device__ float texFetch(texture<float> tex, int i)
{
	return tex1Dfetch(tex, i);
}
#endif
#endif

/*****************************/
/* SET EQUATION COEFFICIENTS */
/*****************************/
void setEquationCoefficients(const int Nrows, const int Ncols, const Real TN, Real * __restrict h_aP, Real * __restrict h_aW, Real * __restrict h_aE,
	Real * __restrict h_aS, Real * __restrict h_aN, Real * __restrict h_b)
{
	for (int col = 0; col < Ncols; ++col) {
		for (int row = 0; row < Nrows; ++row) {
			
			int ind = col * Nrows + row;

			h_b[ind]	= ZERO;

			// --- Left boundary condition: temperature is zero
			if (col == 0)			h_aW[ind] = ZERO;
			else					h_aW[ind] = ONE;

			// --- Right boundary condition: temperature is zero
			if (col == Ncols - 1)	h_aE[ind] = ZERO;
			else					h_aE[ind] = ONE;

			// --- Bottom boundary condition: temperature is zero
			if (row == 0)			h_aS[ind] = ZERO;
			else					h_aS[ind] = ONE;

			// --- Top boundary condition: temperature is TN
			if (row == Nrows - 1) {
									h_aN[ind] = ZERO;
									h_b[ind]  = FOUR * TN;
			}
			else					h_aN[ind] = ONE;

			h_aP[ind] = FOUR;
		}
	} 
} 

/********************************/
/* RED KERNEL - NO OPTIMIZATION */
/********************************/
template<class T>
__global__ void redKernelNoOptimization(const T * __restrict__ h_aP, const T * __restrict__ h_aW, const T * __restrict__ h_aE,
										const T * __restrict__ h_aS, const T * __restrict__ h_aN, const T * __restrict__ h_b,
										const T * __restrict__ h_tempBlack, T * __restrict__ h_tempRed, const T omega,
										T * __restrict__ norm_L2)
{
	// --- Addressing the interior of the (NUM + 2) x (NUM + 2) region
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x + 1;			
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if ((tidx + tidy) % 2 != 0) return;									// --- If we are not on a "red" pixel, then exit.
		
	int ind_red = ( tidy      * (NUM + 2)) + tidx;						// --- Index for the "red" image
	int ind		= ((tidy - 1) * NUM)       + tidx - 1;					// --- Index for the coefficients

	T temp_old = h_tempRed[ind_red];

	T res = h_b[ind]
			+ (h_aW[ind] * h_tempBlack[tidx     + (tidy - 1) * (NUM + 2)]
			+ h_aE[ind]  * h_tempBlack[tidx     + (tidy + 1) * (NUM + 2)]
			+ h_aS[ind]  * h_tempBlack[tidx - 1 +  tidy      * (NUM + 2)]
			+ h_aN[ind]  * h_tempBlack[tidx + 1 +  tidy      * (NUM + 2)]);

	T temp_new = temp_old * (ONE - omega) + omega * (res / h_aP[ind]);

	h_tempRed[ind_red] = temp_new;
		
	res = temp_new - temp_old;
	norm_L2[ind_red] = res * res;

}

/**********************************/
/* BLACK KERNEL - NO OPTIMIZATION */
/**********************************/
template<class T>
__global__ void blackKernelNoOptimization(const T * __restrict__ h_aP, const T * __restrict__ h_aW, const T * __restrict__ h_aE,
	const T * __restrict__ h_aS, const T * __restrict__ h_aN, const T * __restrict__ h_b,
	const T * __restrict__ h_tempRed, T * __restrict__ h_tempBlack, const T omega,
	T * __restrict__ norm_L2)
{
	// --- Addressing the interior of the (NUM + 2) x (NUM + 2) region
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if ((tidx + tidy) % 2 == 0) return;									// --- If we are not on a "black" pixel, then exit.

	int ind_black = (tidy       * (NUM + 2)) + tidx;					// --- Index for the "black" image
	int ind = ((tidy - 1) * NUM) + tidx - 1;				// --- Index for the coefficients

	T temp_old = h_tempBlack[ind_black];

	T res = h_b[ind]
		+ (h_aW[ind] * h_tempRed[tidx + (tidy - 1) * (NUM + 2)]
		+ h_aE[ind] * h_tempRed[tidx + (tidy + 1) * (NUM + 2)]
		+ h_aS[ind] * h_tempRed[tidx - 1 + tidy      * (NUM + 2)]
		+ h_aN[ind] * h_tempRed[tidx + 1 + tidy      * (NUM + 2)]);

	//T res = h_b[ind]
	//	+ (h_aW[ind] * h_tempRed[tidx + (tidy - 1) * (NUM + 2)]);

	T temp_new = temp_old * (ONE - omega) + omega * (res / h_aP[ind]);

	h_tempBlack[ind_black] = temp_new;

	//h_tempBlack[ind_black] = temp_old;

	res = temp_new - temp_old;
	norm_L2[ind_black] = res * res;

}

//template<class T>
//__global__ void kernelNoOptimization(const T * __restrict__ h_aP, const T * __restrict__ h_aW, const T * __restrict__ h_aE,
//	const T * __restrict__ h_aS, const T * __restrict__ h_aN, const T * __restrict__ h_b,
//	const T * __restrict__ h_tempBlack, T * __restrict__ h_tempRed, const T omega, const T TN,
//	T * __restrict__ norm_L2)
//{
//	// --- Addressing the interior of the (NUM + 2) x (NUM + 2) region
//	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
//	const int tidy = blockIdx.y * blockDim.y + threadIdx.y;
//
//	//if ((tidx > NUM + 1) || (tidy > NUM + 1) || (tidx == 0) || (tidy == 0)) return;
//	if ((tidx > NUM + 1) || (tidy > NUM + 1)) return;
//
//	int ind_red = ( tidy      * (NUM + 2)) + tidx;						// --- Index for the "red" image
//	int ind		= ((tidy - 1) * NUM)       + tidx - 1;					// --- Index for the coefficients
//
//	T temp_old = h_tempRed[ind_red];
//
//	T res, temp_new;
//	
//	// --- Northern boundary condition
//	//if (tidy == 0) h_tempRed[ind_red] = TN;
//	//else if ((tidx == 0) || (tidx == NUM + 1) || (tidy == NUM + 1)) h_tempRed[ind_red] = ZERO;
//	//else {
//
//	    res = h_tempBlack[tidx + (tidy - 1) * (NUM + 2)] +
//			h_tempBlack[tidx + (tidy + 1) * (NUM + 2)] +
//			h_tempBlack[tidx - 1 + tidy      * (NUM + 2)] +
//			h_tempBlack[tidx + 1 + tidy      * (NUM + 2)];
//
//		temp_new = res / FOUR;
//
//		h_tempRed[ind_red] = temp_new;
//	//}
//
//	res = temp_new - temp_old;
//	norm_L2[ind_red] = res * res;
//
//}

/************************/
/* RED KERNEL - TEXTURE */
/************************/
#ifdef TEXTURE
template<class T>
__global__ void redKernelTexture(const T * __restrict__ h_tempBlack, T * __restrict__ h_tempRed, const T omega, T * __restrict__ norm_L2)
{
	// --- Addressing the interior of the (NUM + 2) x (NUM + 2) region
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if ((tidx + tidy) % 2 != 0) return;									// --- If we are not on a "red" pixel, then exit.
	
	int ind_red = ( tidy      * (NUM + 2)) + tidx;						// --- Index for the "red" image
	int ind		= ((tidy - 1) * NUM)       + tidx - 1;					// --- Index for the coefficients

	T temp_old = h_tempRed[ind_red];

	T res =  texFetch(t_b, ind)
			 + (texFetch(t_aW, ind) * h_tempBlack[tidx +     (tidy - 1) * (NUM + 2)]
			 +  texFetch(t_aE, ind) * h_tempBlack[tidx +     (tidy + 1) * (NUM + 2)]
			 +  texFetch(t_aS, ind) * h_tempBlack[tidx - 1 +  tidy      * (NUM + 2)]
			 +  texFetch(t_aN, ind) * h_tempBlack[tidx + 1 +  tidy      * (NUM + 2)]);
 
	T temp_new = temp_old * (ONE - omega) + omega * (res / texFetch(t_aP, ind));

	h_tempRed[ind_red] = temp_new;
	
	res = temp_new - temp_old;
	norm_L2[ind_red] = res * res;

}
#endif

/************************************/
/* RED KERNEL - MEMORY OPTIMIZATION */
/************************************/
template<class T>
__global__ void redKernelMemoryOptimization(const T * __restrict__ h_aP, const T * __restrict__ h_aW, const T * __restrict__ h_aE,
											const T * __restrict__ h_aS, const T * __restrict__ h_aN, const T * __restrict__ h_b,
											const T * __restrict__ h_tempBlack, T * __restrict__ h_tempRed, const T omega,
											T * __restrict__ norm_L2)
{
	// --- Addressing the interior of the (NUM + 2) x (NUM + 2) region
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int ind_red = tidy * ((NUM >> 1) + 2) + tidx;  				// --- Local (red) index; problemSize of the matrix is  ((NUM >> 1) + 2) x ((NUM >> 1) + 2)
	int ind		= 2 * tidx - (tidy & 1) - 1 + NUM * (tidy - 1);	// --- Global index

	T temp_old = h_tempRed[ind_red];

	T res = h_b[ind]
		    + (h_aW[ind] * h_tempBlack[tidx					+ (tidy - 1) * ((NUM >> 1) + 2)]
			+  h_aE[ind] * h_tempBlack[tidx					+ (tidy + 1) * ((NUM >> 1) + 2)]
			+  h_aS[ind] * h_tempBlack[tidx - (tidy & 1)		+	tidy	 * ((NUM >> 1) + 2)]
			+  h_aN[ind] * h_tempBlack[tidx + ((tidy + 1) & 1) +   tidy     * ((NUM >> 1) + 2)]);

	T temp_new = temp_old * (ONE - omega) + omega * (res / h_aP[ind]);

	h_tempRed[ind_red] = temp_new;
	res = temp_new - temp_old;

	norm_L2[ind_red] = res * res;
} 

/************************************************/
/* RED KERNEL - MEMORY OPTIMIZATION AND TEXTURE */
/************************************************/
#ifdef TEXTURE
template<class T>
__global__ void redKernelMemoryOptimizationTexture(const T * __restrict__ h_tempBlack, T * __restrict__ h_tempRed, const T omega, T * __restrict__ norm_L2)
{
	// --- Addressing the interior of the (NUM + 2) x (NUM + 2) region
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int ind_red = tidy * ((NUM >> 1) + 2) + tidx;  				// --- Local (red) index; problemSize of the matrix is  ((NUM >> 1) + 2) x ((NUM >> 1) + 2)
	int ind = 2 * tidx - (tidy & 1) - 1 + NUM * (tidy - 1);	// --- Global index

	T temp_old = h_tempRed[ind_red];

	T res = texFetch(t_b, ind)
			+ (texFetch(t_aW, ind) * h_tempBlack[tidx                    + (tidy - 1) * ((NUM >> 1) + 2)]
			+  texFetch(t_aE, ind) * h_tempBlack[tidx                    + (tidy + 1) * ((NUM >> 1) + 2)]
			+  texFetch(t_aS, ind) * h_tempBlack[tidx - (tidy & 1)       +  tidy      * ((NUM >> 1) + 2)]
			+  texFetch(t_aN, ind) * h_tempBlack[tidx + ((tidy + 1) & 1) +  tidy      * ((NUM >> 1) + 2)]);

	T temp_new = temp_old * (ONE - omega) + omega * (res / texFetch(t_aP, ind));

	h_tempRed[ind_red] = temp_new;
	res = temp_new - temp_old;

	norm_L2[ind_red] = res * res;
}
#endif

/**************************/
/* BLACK KERNEL - TEXTURE */
/**************************/
#ifdef TEXTURE
template<class T>
__global__ void blackKernelTexture(const T * __restrict__ h_tempRed, T * __restrict__ h_tempBlack, const T omega, T * __restrict__ norm_L2)
{
	// --- Addressing the interior of the (NUM + 2) x (NUM + 2) region
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if ((tidx + tidy) % 2 == 0) return;									// --- If we are not on a "black" pixel, then exit.
	
	int ind_black = (tidy       * (NUM + 2)) + tidx;					// --- Index for the "black" image
	int ind		  = ((tidy - 1) * NUM)       + tidx - 1;				// --- Index for the coefficients

	T temp_old = h_tempBlack[ind_black];

	T res =  texFetch(t_b, ind)
			 + (texFetch(t_aW, ind) * h_tempRed[tidx     + (tidy - 1) * (NUM + 2)]
			 + texFetch (t_aE, ind) * h_tempRed[tidx     + (tidy + 1) * (NUM + 2)]
			 + texFetch (t_aS, ind) * h_tempRed[tidx - 1 +  tidy      * (NUM + 2)]
			 + texFetch (t_aN, ind) * h_tempRed[tidx + 1 +  tidy      * (NUM + 2)]);

	T temp_new = temp_old * (ONE - omega) + omega * (res / texFetch(t_aP, ind));

	h_tempBlack[ind_black] = temp_new;
	res = temp_new - temp_old;

	norm_L2[ind_black] = res * res;

}
#endif

/**************************************/
/* BLACK KERNEL - MEMORY OPTIMIZATION */
/**************************************/
template<class T>
__global__ void blackKernelMemoryOptimization(const T * __restrict__ h_aP, const T * __restrict__ h_aW, const T * __restrict__ h_aE,
											  const T * __restrict__ h_aS, const T * __restrict__ h_aN, const T * __restrict__ h_b,
											  const T * __restrict__ h_tempRed, T * __restrict__ h_tempBlack, const T omega,
											  T * __restrict__ norm_L2)
{
	// --- Addressing the interior of the (NUM + 2) x (NUM + 2) region
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int ind_black = tidy * ((NUM >> 1) + 2) + tidx;  				// --- Local (red) index; problemSize of the matrix is  ((NUM >> 1) + 2) x ((NUM >> 1) + 2)
	int ind = 2 * tidx - ((tidy + 1) & 1) - 1 + NUM * (tidy - 1);	// --- Global index

	T temp_old = h_tempBlack[ind_black];

	T res = h_b[ind]
			+ (h_aW[ind] * h_tempRed[tidx						+ (tidy - 1) * ((NUM >> 1) + 2)]
			+  h_aE[ind] * h_tempRed[tidx						+ (tidy + 1) * ((NUM >> 1) + 2)]
			+  h_aS[ind] * h_tempRed[tidx - ((tidy + 1) & 1)	+ tidy		 * ((NUM >> 1) + 2)]
			+  h_aN[ind] * h_tempRed[tidx + (tidy & 1)			+ tidy		 * ((NUM >> 1) + 2)]);

	T temp_new = temp_old * (ONE - omega) + omega * (res / h_aP[ind]);

	h_tempBlack[ind_black] = temp_new;
	res = temp_new - temp_old;

	norm_L2[ind_black] = res * res;
} 

/**************************************************/
/* BLACK KERNEL - MEMORY OPTIMIZATION AND TEXTURE */
/**************************************************/
#ifdef TEXTURE
template<class T>
__global__ void blackKernelMemoryOptimizationTexture(const T * __restrict__ h_tempRed, T * __restrict__ h_tempBlack, const T omega, T * __restrict__ norm_L2)
{
	// --- Addressing the interior of the (NUM + 2) x (NUM + 2) region
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x + 1;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int ind_black = tidy * ((NUM >> 1) + 2) + tidx;  				// --- Local (red) index; problemSize of the matrix is  ((NUM >> 1) + 2) x ((NUM >> 1) + 2)
	int ind = 2 * tidx - ((tidy + 1) & 1) - 1 + NUM * (tidy - 1);	// --- Global index

	T temp_old = h_tempBlack[ind_black];

	T res = texFetch(t_b, ind)
			+ (texFetch(t_aW, ind) * h_tempRed[tidx                    + (tidy - 1) * ((NUM >> 1) + 2)]
			+  texFetch(t_aE, ind) * h_tempRed[tidx                    + (tidy + 1) * ((NUM >> 1) + 2)]
			+  texFetch(t_aS, ind) * h_tempRed[tidx - ((tidy + 1) & 1) +  tidy      * ((NUM >> 1) + 2)]
			+  texFetch(t_aN, ind) * h_tempRed[tidx +  (tidy & 1)      +  tidy      * ((NUM >> 1) + 2)]);

	T temp_new = temp_old * (ONE - omega) + omega * (res / texFetch(t_aP, ind));

	h_tempBlack[ind_black] = temp_new;
	res = temp_new - temp_old;

	norm_L2[ind_black] = res * res;
} 
#endif

/********/
/* MAIN */
/********/
int main(void) {

	TimingGPU timerGPU;
	
	/**********************/
	/* PROBLEM PARAMETERS */
	/**********************/
	Real TN				= 1.0;								// --- Temperature at northern boundary
	Real TS				= 0.;								// --- Temperature at southern boundary
	Real TW				= 0.;								// --- Temperature at western boundary	
	Real TE				= 0.;								// --- Temperature at eastern boundary

	//Real dx = L / NUM;																	// --- Discretization step along x-axis
	//Real dy = H / NUM;																	// --- Discretization step along y-axis

	// --- Number of discretization points along x and y including boundary points
#ifdef MEMOPT
	int Nrows = (NUM / 2) + 2;
#else
	int Nrows = NUM + 2;
#endif
	int Ncols = NUM + 2;
	
	// --- Problem size and computational size
	int problemSize			= NUM * NUM;
	int computationalSize	= Nrows * Ncols;

	/*************************/
	/* ITERATIONS PARAMETERS */
	/*************************/
	Real tol = 1.e-6;										// --- SOR iteration tolerance
	//int maxIter = 1e6;
	//int maxIter = 1e3;									// --- Maximum number of iterations
	int maxIter = 1;										// --- Maximum number of iterations

	int iter;

	/***************************/
	/* HOST MEMORY ALLOCATIONS */
	/***************************/
	// --- Equation coefficients
	Real *h_aP = (Real *)calloc(problemSize, sizeof(Real));						// --- Self coefficients
	Real *h_aW = (Real *)calloc(problemSize, sizeof(Real));						// --- West neighbor coefficients
	Real *h_aE = (Real *)calloc(problemSize, sizeof(Real));						// --- East neighbor coefficients
	Real *h_aS = (Real *)calloc(problemSize, sizeof(Real));						// --- South neighbor coefficients
	Real *h_aN = (Real *)calloc(problemSize, sizeof(Real));						// --- North neighbor coefficients

	// --- Right-hand side array
	Real *h_b = (Real *)calloc(problemSize, sizeof(Real));

	Real *h_tempRed		= (Real *)calloc(computationalSize, sizeof(Real));		// --- Red-cells temperature array
	Real *h_tempBlack	= (Real *)calloc(computationalSize, sizeof(Real));		// --- Black-cells temperature array

	// --- Set equation coefficients
	//setEquationCoefficients(NUM, NUM, thConductivity, dx, dy, width, TN, h_aP, h_aW, h_aE, h_aS, h_aN, h_b);
	setEquationCoefficients(NUM, NUM, TN, h_aP, h_aW, h_aE, h_aS, h_aN, h_b);

	/****************************/
	/* SET GRID AND BLOCK SIZES */
	/****************************/
	dim3 dimBlock(BLOCKSIZEX, BLOCKSIZEY);
#ifdef MEMOPT
	dim3 dimGrid(iDivUp(NUM / 2, BLOCKSIZEX), iDivUp(NUM, BLOCKSIZEY));
#else
	dim3 dimGrid(iDivUp(NUM, BLOCKSIZEX), iDivUp(NUM, BLOCKSIZEY));
	printf("dimGrid = %d %d\n", dimGrid.x, dimGrid.y);
#endif

	printf("Problem problemSize: %d x %d \n", NUM, NUM);

	timerGPU.StartCounter();

	/*****************************/
	/* DEVICE MEMORY ALLOCATIONS */
	/*****************************/
	Real *d_aP;			gpuErrchk(cudaMalloc((void**)&d_aP, problemSize * sizeof(Real)));
	Real *d_aW;			gpuErrchk(cudaMalloc((void**)&d_aW, problemSize * sizeof(Real)));
	Real *d_aE;			gpuErrchk(cudaMalloc((void**)&d_aE, problemSize * sizeof(Real)));
	Real *d_aS;			gpuErrchk(cudaMalloc((void**)&d_aS, problemSize * sizeof(Real)));
	Real *d_aN;			gpuErrchk(cudaMalloc((void**)&d_aN, problemSize * sizeof(Real)));
	Real *d_b;			gpuErrchk(cudaMalloc((void**)&d_b,  problemSize * sizeof(Real)));
	Real *d_tempRed;	gpuErrchk(cudaMalloc((void**)&d_tempRed, computationalSize * sizeof(Real)));
#ifdef MEMOPT
	Real *d_tempBlack;	gpuErrchk(cudaMalloc((void**)&d_tempBlack, computationalSize * sizeof(Real)));
#endif

	Real *d_L2DifferenceArray;	gpuErrchk(cudaMalloc((void**)&d_L2DifferenceArray, computationalSize * sizeof(Real)));

	/*****************************/
	/* HOST-DEVICE MEMORY COPIES */
	/*****************************/
	gpuErrchk(cudaMemcpy(d_aP, h_aP, problemSize * sizeof(Real), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_aW, h_aW, problemSize * sizeof(Real), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_aE, h_aE, problemSize * sizeof(Real), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_aS, h_aS, problemSize * sizeof(Real), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_aN, h_aN, problemSize * sizeof(Real), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, problemSize * sizeof(Real), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(d_tempRed, 0, computationalSize * sizeof(Real)));
#ifdef MEMOPT
	gpuErrchk(cudaMemset(d_tempBlack, 0, computationalSize * sizeof(Real)));
#endif

	/********************/
	/* TEXTURE BINDINGS */
	/********************/
#ifdef TEXTURE
	gpuErrchk(cudaBindTexture(NULL, t_aP, d_aP, problemSize * sizeof(Real)));
	gpuErrchk(cudaBindTexture(NULL, t_aW, d_aW, problemSize * sizeof(Real)));
	gpuErrchk(cudaBindTexture(NULL, t_aE, d_aE, problemSize * sizeof(Real)));
	gpuErrchk(cudaBindTexture(NULL, t_aS, d_aS, problemSize * sizeof(Real)));
	gpuErrchk(cudaBindTexture(NULL, t_aN, d_aN, problemSize * sizeof(Real)));
	gpuErrchk(cudaBindTexture(NULL, t_b, d_b, problemSize * sizeof(Real)));
#endif

	/**************/
	/* ITERATIONS */
	/**************/
	for (iter = 0; iter < maxIter; ++iter) {

		// --- Update red cells
#if defined(TEXTURE) && defined(MEMOPT)
		redKernelMemoryOptimizationTexture << <dimGrid, dimBlock >> > (d_tempBlack, d_tempRed, omega, d_L2DifferenceArray);
#elif defined(TEXTURE) && !defined(MEMOPT)
		redKernelTexture << <dimGrid, dimBlock >> > (d_tempRed, d_tempRed, omega, d_L2DifferenceArray);
#elif !defined(TEXTURE) && defined(MEMOPT)
		redKernelMemoryOptimization << <dimGrid, dimBlock >> > (d_aP, d_aW, d_aE, d_aS, d_aN, d_b, d_tempBlack, d_tempRed, omega, d_L2DifferenceArray);
#else 
		redKernelNoOptimization << <dimGrid, dimBlock >> > (d_aP, d_aW, d_aE, d_aS, d_aN, d_b, d_tempRed, d_tempRed, omega, d_L2DifferenceArray);
#endif

		// --- Update black cells
#if defined(TEXTURE) && defined(MEMOPT)
		blackKernelMemoryOptimizationTexture << <dimGrid, dimBlock >> > (d_tempRed, d_tempBlack, omega, d_L2DifferenceArray);		
#elif defined(TEXTURE) && !defined(MEMOPT)
		blackKernelTexture << <dimGrid, dimBlock >> > (d_tempRed, d_tempRed, omega, d_L2DifferenceArray);
#elif !defined(TEXTURE) && defined(MEMOPT)
		blackKernelMemoryOptimization << <dimGrid, dimBlock >> > (d_aP, d_aW, d_aE, d_aS, d_aN, d_b, d_tempRed, d_tempBlack, omega, d_L2DifferenceArray);
#else 
		blackKernelNoOptimization << <dimGrid, dimBlock >> > (d_aP, d_aW, d_aE, d_aS, d_aN, d_b, d_tempRed, d_tempRed, omega, d_L2DifferenceArray);
#endif

		// --- Calculate residual
		Real norm_L2 = thrust::reduce(thrust::device_pointer_cast(d_L2DifferenceArray), thrust::device_pointer_cast(d_L2DifferenceArray) + computationalSize);
		norm_L2 = sqrt(norm_L2 / ((Real)problemSize));

		if (iter % 100 == 0) printf("%5d, %0.6f\n", iter, norm_L2);

		// --- If tolerance has been reached, end SOR iterations
		if (norm_L2 < tol) break;
	}

	// --- Transfer final red and black temperatures back to the host
	gpuErrchk(cudaMemcpy(h_tempRed,   d_tempRed,   computationalSize * sizeof(Real), cudaMemcpyDeviceToHost));
#ifdef MEMOPT
	//cudaMemcpy(h_tempBlack, d_tempRed, computationalSize * sizeof(Real), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tempBlack, d_tempBlack, computationalSize * sizeof(Real), cudaMemcpyDeviceToHost);
#endif

	double runtime = timerGPU.GetCounter();

	printf("GPU\n");
	printf("Iterations: %i\n", iter);
	printf("Total time: %f s\n", runtime / 1000.0);

	/****************************/
	/* SAVING THE FINAL RESULTS */
	/****************************/
	Real *h_T = (Real *)calloc(NUM * NUM, sizeof(Real));
	
	for (int row = 1; row < NUM + 1; ++row) {
		for (int col = 1; col < NUM + 1; ++col) {
			
			// --- Red cells
			if ((row + col) % 2 == 0) {
				int ind = col * Nrows + (row + (col % 2)) / 2;
				h_T[((col - 1) * NUM) + row - 1] = h_tempRed[ind];
			}
			// --- Black cells
			else {
				int ind = col * Nrows + (row + ((col + 1) % 2)) / 2;
#ifdef MEMOPT
				h_T[((col - 1) * NUM) + row - 1] = h_tempBlack[ind];
#else
				h_T[((col - 1) * NUM) + row - 1] = h_tempRed[ind];
#endif
			}
		}
	}

	printf("Saving...\n");
	saveCPUrealtxt(h_T, "D:\\Laplace\\Laplace_SOR_Red_Black\\Laplace_SOR_Red_Black\\Temp.txt", NUM * NUM);
	saveCPUrealtxt(h_tempRed, "D:\\Laplace\\Laplace_SOR_Red_Black\\Laplace_SOR_Red_Black\\Temp_red.txt", (NUM + 2) * (NUM + 2));

	// --- Free device memory
	gpuErrchk(cudaFree(d_aP));
	gpuErrchk(cudaFree(d_aW));
	gpuErrchk(cudaFree(d_aE));
	gpuErrchk(cudaFree(d_aS));
	gpuErrchk(cudaFree(d_aN));
	gpuErrchk(cudaFree(d_b));
	gpuErrchk(cudaFree(d_tempRed));
#ifdef MEMOPT
	gpuErrchk(cudaFree(d_tempBlack));
#endif

	gpuErrchk(cudaFree(d_L2DifferenceArray));

#ifdef TEXTURE
	// --- Unbind textures
	gpuErrchk(cudaUnbindTexture(t_aP));
	gpuErrchk(cudaUnbindTexture(t_aW));
	gpuErrchk(cudaUnbindTexture(t_aE));
	gpuErrchk(cudaUnbindTexture(t_aS));
	gpuErrchk(cudaUnbindTexture(t_aN));
	gpuErrchk(cudaUnbindTexture(t_b));
#endif

	free(h_aP);
	free(h_aW);
	free(h_aE);
	free(h_aS);
	free(h_aN);
	free(h_b);
	free(h_tempRed);
	free(h_tempBlack);

	gpuErrchk(cudaDeviceReset());

	return 0;
}
