#include <stdio.h>
#include <stdlib.h>

#include <thrust\device_vector.h>

#include "Utilities.cuh"

#define BLOCKSIZE  512

/****************************/
/* CPU CALCULATION FUNCTION */
/****************************/
void HeatEquation1DCPU(float * __restrict__ h_T, int *Niter, const float T0, const float Q_N_1, const float dx, const float k, const float rho, 
					   const float cp, const float alpha, const float dt, const float maxErr, const int maxIterNumber, const int N)
{
    float *h_DeltaT = (float *)malloc(N * sizeof(float));

    // --- Enforcing boundary condition at the left end.
    *h_T = T0;
    h_DeltaT[0] = 0.f;

    float current_max;
	do {
        // --- Internal region between the two boundaries.
        for (int i = 1; i < N - 1; i++) h_DeltaT[i] = dt * alpha * ((h_T[i - 1] + h_T[i + 1] - 2.f * h_T[i]) / (dx * dx));

        // --- Enforcing boundary condition at the right end.
        h_DeltaT[N - 1] = dt * 2.f * ((k * ((h_T[N - 2] - h_T[N - 1]) / dx) + Q_N_1) / (dx * rho * cp));

		// --- Update the temperature and find the maximum DeltaT over all nodes
		current_max = h_DeltaT[0]; // --- Remember: h_DeltaT[0] = 0
        for (int i = 1; i < N; i++)
        {
            h_T[i] = h_T[i] + h_DeltaT[i]; // h_T[0] keeps
            current_max = abs(h_DeltaT[i]) > current_max ? abs(h_DeltaT[i]) : current_max;
        }

        // --- Increase iteration counter
        (*Niter)++;

    } while (*Niter < maxIterNumber && current_max > maxErr);

    delete [] h_DeltaT;
}

/**************************/
/* GPU CALCULATION KERNEL */
/**************************/
__global__ void HeatEquation1DGPU_IterationKernel(float * __restrict__ d_T, float * __restrict__ d_DeltaT, const float T0, const float Q_N_1, const float dx, const float k, const float rho, 
					   const float cp, const float alpha, const float dt, const float maxErr, const int maxIterNumber, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid < N) {
		
        // --- Internal region between the two boundaries.
		if ((tid > 0) && (tid < N - 1) ) d_DeltaT[tid]  = dt * alpha *((d_T[tid - 1] + d_T[tid + 1] - 2.f * d_T[tid]) / (dx * dx));
		// --- Enforcing boundary condition at the left end.
		if (tid == 0)					 d_DeltaT[0]	= 0.f;
		// --- Enforcing boundary condition at the right end.
 		if (tid == N - 1)				 d_DeltaT[tid]	= dt * 2.f * ((k * ((d_T[tid - 1] - d_T[tid]) / dx) + Q_N_1) / (dx * rho * cp));
		
		// --- Update the temperature
        d_T[tid] = d_T[tid] + d_DeltaT[tid]; 

		d_DeltaT[tid] = abs(d_DeltaT[tid]);
	}
	
}

__global__ void HeatEquation1DGPU_IterationSharedKernel(float * __restrict__ d_T, float * __restrict__ d_DeltaT, const float T0, const float Q_N_1, const float dx, const float k, const float rho, 
					   const float cp, const float alpha, const float dt, const float maxErr, const int maxIterNumber, const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// --- Shared memory has 0, 1, ..., BLOCKSIZE - 1, BLOCKSIZE locations, so it has BLOCKSIZE locations + 2 (left and right) halo cells.
	__shared__ float d_T_shared[BLOCKSIZE + 2];				// --- Need to know BLOCKSIZE beforehand
	
	if (tid < N) {
		
		// --- Load data from global memory to shared memory locations 1, 2, ..., BLOCKSIZE - 1
		d_T_shared[threadIdx.x + 1] = d_T[tid];					

		// --- Left halo cell
		if ((threadIdx.x == 0) && (tid > 0)) { d_T_shared[0] = d_T[tid - 1]; }			

		// --- Right halo cell
		if ((threadIdx.x == blockDim.x - 1) && (tid < N - 1)) { d_T_shared[threadIdx.x + 2] = d_T[tid + 1]; } 

		__syncthreads();
		
		// --- Internal region between the two boundaries.
		if ((tid > 0) && (tid < N - 1) ) d_DeltaT[tid]  = dt * alpha *((d_T_shared[threadIdx.x] + d_T_shared[threadIdx.x + 2] - 2.f * d_T_shared[threadIdx.x + 1]) / (dx * dx));
		
		// --- Enforcing boundary condition at the left end.
		if (tid == 0)					 d_DeltaT[0]	= 0.f;
		
		// --- Enforcing boundary condition at the right end.
 		if (tid == N - 1)				 d_DeltaT[tid]	= dt * 2.f * ((k * ((d_T_shared[threadIdx.x] - d_T_shared[threadIdx.x + 1]) / dx) + Q_N_1) / (dx * rho * cp));
		
		// --- Update the temperature
        d_T[tid] = d_T[tid] + d_DeltaT[tid]; 

		d_DeltaT[tid] = abs(d_DeltaT[tid]);
	}
	
}

/****************************/
/* GPU CALCULATION FUNCTION */
/****************************/
void HeatEquation1DGPU(float * __restrict__ d_T, int *Niter, const float T0, const float Q_N_1, const float dx, const float k, const float rho, 
					   const float cp, const float alpha, const float dt, const float maxErr, const int maxIterNumber, const int N)
{
	// --- Absolute values of DeltaT
	float *d_DeltaT;	gpuErrchk(cudaMalloc(&d_DeltaT, N * sizeof(float)));

    // --- Enforcing boundary condition at the left end.
    gpuErrchk(cudaMemcpy(d_T, &T0, sizeof(float), cudaMemcpyHostToDevice));

	float current_max = 0.f;
	do {
		//HeatEquation1DGPU_IterationKernel<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_T, d_DeltaT, T0, Q_N_1, dx, k, rho, cp, alpha, dt, maxErr, maxIterNumber, N);
		HeatEquation1DGPU_IterationSharedKernel<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_T, d_DeltaT, T0, Q_N_1, dx, k, rho, cp, alpha, dt, maxErr, maxIterNumber, N);

		thrust::device_ptr<float> d = thrust::device_pointer_cast(d_DeltaT);  
		current_max = thrust::reduce(d, d + N, current_max, thrust::maximum<float>());

        // --- Increase iteration counter
		(*Niter)++;

    } while (*Niter < maxIterNumber && current_max > maxErr);

    gpuErrchk(cudaFree(d_DeltaT));
}

/********/
/* MAIN */
/********/
int main()
{
	// --- See https://en.wikipedia.org/wiki/Thermal_diffusivity
	
	// --- Parameters of the problem
    const float k			= 0.19f;					// --- Thermal conductivity [W / (m * K)]
    const float rho			= 930.f;					// --- Density [kg / m^3]
    const float cp			= 1340.f;					// --- Specific heat capacity [J / (kg * K)]
    const float alpha		= k / (rho * cp);			// --- Thermal diffusivity [m^2 / s]
	const float length		= 1.6f;						// --- Total length of the domain [m]
	const int N				= 64 * BLOCKSIZE;			// --- Number of grid points
	const float dx			= (length / (float)(N - 1));// --- Discretization step [m]
    const float dt			= (float)(dx * dx / (4.f * alpha));
														// --- Time step [s]
	const float T0			= 0.f;						// --- Temperature at the first end of the domain [C]
    const float Q_N_1		= 10.f;						// --- Heat flux at the second end of the domain [W / m^2]
    const float maxErr		= 1.0e-5f;					// --- Maximum admitted DeltaT
    const int maxIterNumber = 10.0 / dt;				// --- Number of overall time steps

    /********************/
	/* GPU CALCULATIONS */
    /********************/
	float *h_T_final_device = (float *)malloc(N * sizeof(float));		// --- Final "host-side" result of GPU calculations
    int Niter_GPU = 0;													// --- Iteration counter for GPU calculations

    // --- Device temperature allocation and initialization
	float *d_T;		gpuErrchk(cudaMalloc(&d_T, N * sizeof(float)));
    gpuErrchk(cudaMemset(d_T, 0, N * sizeof(float))); 

    // --- GPU calculations
	HeatEquation1DGPU(d_T, &Niter_GPU, T0, Q_N_1, dx, k, rho, cp, alpha, dt, maxErr, maxIterNumber, N);

    // --- Transfer the GPU calculation results from device to host
    gpuErrchk(cudaMemcpy(h_T_final_device, d_T, N * sizeof(float), cudaMemcpyDeviceToHost));

    /********************/
	/* CPU CALCULATIONS */
    /********************/
    // --- Host temperature allocation and initialization
    float *h_T_final_host = (float *)malloc(N * sizeof(float));
    memset(h_T_final_host, 0, N * sizeof(float));
    
	int Niter_CPU = 0;

    HeatEquation1DCPU(h_T_final_host, &Niter_CPU, T0, Q_N_1, dx, k, rho, cp, alpha, dt, maxErr, maxIterNumber, N);

    /************************/
	/* CHECKING THE RESULTS */
    /************************/
	for (int i = 0; i < N; i++) {
        printf("Node = %i; T_host = %3.10f; T_device = %3.10f\n", i, h_T_final_host[i], h_T_final_device[i]);
		if (h_T_final_host[i] != h_T_final_device[i]) {
            printf("Error at i = %i; T_host = %f; T_device = %f\n", i, h_T_final_host[i], h_T_final_device[i]);
            return 0;
        }
    }

    printf("Test passed!\n");

    delete [] h_T_final_device;
    gpuErrchk(cudaFree(d_T));

    return 0;
}
