#include <stdio.h>
#include <string.h>

#include "Utilities.cuh"

#define BLOCK_SIZE 32

/********************************/
/* HOST INITIALIZATION FUNCTION */
/********************************/
void init(double * __restrict U, double * __restrict U_old, double * __restrict F, const double x_min, const double x_max, const double y_min, const double y_max, double delta, int Nb)
{
    // --- Initilize grid coordinates
    double x = -1.0;
    double y = -1.0;

	for (int i = 0; i < Nb; i++) {
		for (int j = 0; j < Nb; j++) {
            
			F    [i * (Nb) + j]	= 0.0;
            U    [i * (Nb) + j]	= 0.0;
            U_old[i * (Nb) + j] = 0.0;
            
			// --- Set radiator temperature in the source box
            if (x <= x_max && x >= x_min && y <= y_max && y >= y_min) F[i * Nb + j] = 200.0;

			// --- Boundary condition on the left, right and upper walls. The boundary condition on the lower wall is vanishing temperature
            if (i == (Nb - 1) || i == 0 || j == (Nb - 1))
            {
                U    [i * (Nb) + j] = 20.0;
                U_old[i * (Nb) + j] = 20.0;
            }
            // --- Update y-grid coordinate
            y += delta;
        }
        // --- Update x-grid coordinate
        x += delta;
        y = -1.0;
    }
}

/*********************************/
/* JACOBI ITERATOR HOST FUNCTION */
/*********************************/
void jacobi_iterator_CPU(const double * __restrict__ U, double * __restrict__ U_old, const double * __restrict__ F, const double delta2, const int Nb)
{
	for (int i=1; i<Nb-1; i++)
		for (int j=1; j<Nb-1; j++)
	        U_old[j * Nb + i] = (U[j * Nb + (i - 1)] + U[j * Nb + (i + 1)] + U[(j - 1) * Nb + i] + U[(j + 1) * Nb + i] + (delta2 * F[j * Nb + i])) * 0.25;

}

/***********************************/
/* JACOBI ITERATOR KERNEL FUNCTION */
/***********************************/
__global__ void jacobi_iterator_GPU(const double * __restrict__ U, double * __restrict__ U_old, const double * __restrict__ F, const double delta2, const int Nb)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
	if (i < Nb - 1 && j < Nb - 1 && i > 0 && j > 0)
    {
        U_old[j * Nb + i] = (U[j * Nb + (i - 1)] + U[j * Nb + (i + 1)] + U[(j - 1) * Nb + i] + U[(j + 1) * Nb + i] + (delta2 * F[j * Nb + i])) * 0.25;

    }
}

/***************************/
/* DISPLAY MATRIX FUNCTION */
/***************************/
void print_matrix(int N, double *M)
{
    int Nb = N + 2;
    int i, j;
    for (i = Nb - 1; i >= 0; i--)
    {
        for (j = 0; j < Nb; j++)
        {
            printf("%.2f\t", M[j * Nb + i]);
        }
        printf("\n");
    }
}

/********/
/* MAIN */
/********/
int main()
{
	// --- The computation domain is [-1, 1] x [-1, 1]
	
	const int N			= 16;							// --- Grid size is N x N
    const int Nb		= N + 2;						// --- Grid side including the boundaries is Nb x Nb
    const int MAX_ITER	= 1000;							// --- Maximum number of iterations

    // --- Defining the source box
	double x_min = 0.0;
    double x_max = 1.0 / 3.0;
    double y_min = -2.0 / 3.0;
    double y_max = -1.0 / 3.0;

	double delta		= 2.0 / ((double)N - 1.0);		// --- Discretization step

	// --- Allocating host memory variables
    double *h_U				= (double *)malloc(Nb * Nb * sizeof(double));
    double *h_U_old			= (double *)malloc(Nb * Nb * sizeof(double));
    double *h_F				= (double *)malloc(Nb * Nb * sizeof(double));

	// --- Allocating device memory variables
	double *d_U;			gpuErrchk(cudaMalloc(&d_U,		Nb * Nb * sizeof(double)));
    double *d_U_old;		gpuErrchk(cudaMalloc(&d_U_old,	Nb * Nb * sizeof(double)));
	double *d_F;			gpuErrchk(cudaMalloc(&d_F,		Nb * Nb * sizeof(double)));

	// --- Dummy pointer for pointer swapping
	double *temp;

    // --- Host array initialization
    init(h_U, h_U_old, h_F, x_min, x_max, y_min, y_max, delta, Nb);

    // --- Copying arrays from host to device
    gpuErrchk(cudaMemcpy(d_U,		h_U,		Nb * Nb * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_U_old,	h_U_old,	Nb * Nb * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_F,		h_F,		Nb * Nb * sizeof(double), cudaMemcpyHostToDevice));
    
    // --- Host iterations
    for (int h = 0; h < MAX_ITER; h++)
    {
        jacobi_iterator_CPU(h_U, h_U_old, h_F, delta * delta, Nb);

		// --- Pointers swap
        temp = h_U;
        h_U = h_U_old;
        h_U_old = temp;
    }
	
	printf("Host results\n");
	print_matrix(N, h_U);

	// --- Device iterations
	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 DimGrid(iDivUp(N, BLOCK_SIZE), iDivUp(N, BLOCK_SIZE));

    for (int h = 0; h < MAX_ITER; h++)
    {
        jacobi_iterator_GPU<<<DimGrid, DimBlock>>>(d_U, d_U_old, d_F, delta * delta, Nb);

		// --- Pointers swap
        temp = d_U;
        d_U = d_U_old;
        d_U_old = temp;
    }

    // --- Move device result to the host
	gpuErrchk(cudaMemcpy(h_U, d_U, Nb * Nb * sizeof(double), cudaMemcpyDeviceToHost));
    
	printf("Device results\n");
	print_matrix(N, h_U);

	// --- Freeing host memory
    free(h_U);
    free(h_U_old);
    free(h_F);

	// --- Freeing device memory
    gpuErrchk(cudaFree(d_U));
    gpuErrchk(cudaFree(d_U_old));
    gpuErrchk(cudaFree(d_F));
    
	return 0;
}
