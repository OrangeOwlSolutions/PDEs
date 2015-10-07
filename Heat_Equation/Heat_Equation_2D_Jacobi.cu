#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Utilities.cuh"
#include "InputOutput.cuh"
#include "TimingGPU.cuh"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

texture<float, 2, cudaReadModeElementType>  tex_T;
texture<float, 2, cudaReadModeElementType>  tex_T_old;

/***********************************/
/* JACOBI ITERATION FUNCTION - GPU */
/***********************************/
__global__ void Jacobi_Iterator_GPU(const float * __restrict__ T_old, float * __restrict__ T_new, const int NX, const int NY)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x ;
    const int j = blockIdx.y * blockDim.y + threadIdx.y ;

                                //                         N 
    int P = i + j*NX;           // node (i,j)              |
    int N = i + (j+1)*NX;       // node (i,j+1)            |
    int S = i + (j-1)*NX;       // node (i,j-1)     W ---- P ---- E
    int E = (i+1) + j*NX;       // node (i+1,j)            |
    int W = (i-1) + j*NX;       // node (i-1,j)            |
                                //                         S 

    // --- Only update "interior" (not boundary) node points
	if (i>0 && i<NX-1 && j>0 && j<NY-1) T_new[P] = 0.25 * (T_old[E] + T_old[W] + T_old[N] + T_old[S]); 
}

/*********************************************/
/* JACOBI ITERATION FUNCTION - GPU - TEXTURE */
/*********************************************/
__global__ void Jacobi_Iterator_GPU_texture(float * __restrict__ T_new, const bool flag, const int NX, const int NY) {
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x ;
    const int j = blockIdx.y * blockDim.y + threadIdx.y ;

	float P, N, S, E, W;	
	if (flag) {
											//                         N 
        P = tex2D(tex_T_old, i,     j);		// node (i,j)              |
        N = tex2D(tex_T_old, i,     j + 1);	// node (i,j+1)            |
		S = tex2D(tex_T_old, i,     j - 1);	// node (i,j-1)     W ---- P ---- E
        E = tex2D(tex_T_old, i + 1, j);		// node (i+1,j)            |
        W = tex2D(tex_T_old, i - 1, j);		// node (i-1,j)            |
											//                         S 
	} else {
											//                         N 
        P = tex2D(tex_T,     i,     j);		// node (i,j)              |
        N = tex2D(tex_T,     i,     j + 1);	// node (i,j+1)            |
		S = tex2D(tex_T,     i,     j - 1);	// node (i,j-1)     W ---- P ---- E
        E = tex2D(tex_T,     i + 1, j);		// node (i+1,j)            |
        W = tex2D(tex_T,     i - 1, j);		// node (i-1,j)            |
											//                         S 
	}

	// --- Only update "interior" (not boundary) node points
	if (i>0 && i<NX-1 && j>0 && j<NY-1) T_new[i + j*NX] = 0.25 * (E + W + N + S);
}

/***********************************/
/* JACOBI ITERATION FUNCTION - CPU */
/***********************************/
void Jacobi_Iterator_CPU(float * __restrict T, float * __restrict T_new, const int NX, const int NY, const int MAX_ITER)
{
	for(int iter=0; iter<MAX_ITER; iter=iter+2)
    {
	    // --- Only update "interior" (not boundary) node points
        for(int j=1; j<NY-1; j++) 
			for(int i=1; i<NX-1; i++) {
                float T_E = T[(i+1) + NX*j];
                float T_W = T[(i-1) + NX*j];
                float T_N = T[i + NX*(j+1)];
                float T_S = T[i + NX*(j-1)];
                T_new[i+NX*j] = 0.25*(T_E + T_W + T_N + T_S);
            }
 
        for(int j=1; j<NY-1; j++) 
			for(int i=1; i<NX-1; i++) {
                float T_E = T_new[(i+1) + NX*j];
                float T_W = T_new[(i-1) + NX*j];
                float T_N = T_new[i + NX*(j+1)];
                float T_S = T_new[i + NX*(j-1)];
                T[i+NX*j] = 0.25*(T_E + T_W + T_N + T_S);
            }
    }
}

/******************************/
/* TEMPERATURE INITIALIZATION */
/******************************/
void Initialize(float * __restrict h_T, const int NX, const int NY)
{
    // --- Set left wall to 1
    for(int j=0; j<NY; j++) h_T[j * NX] = 1.0;
}


/********/
/* MAIN */
/********/
int main()
{
	const int NX = 1024;			// --- Number of discretization points along the x axis
	const int NY = 1024;			// --- Number of discretization points along the y axis

	const int MAX_ITER = 1;	// --- Number of Jacobi iterations

    // --- CPU temperature distributions
    float *h_T				= (float *)calloc(NX * NY, sizeof(float));
    float *h_T_old			= (float *)calloc(NX * NY, sizeof(float));
    Initialize(h_T,     NX, NY);
    Initialize(h_T_old, NX, NY);
    float *h_T_GPU_result	= (float *)malloc(NX * NY * sizeof(float));

    // --- GPU temperature distribution
    float *d_T;			gpuErrchk(cudaMalloc((void**)&d_T,			NX * NY * sizeof(float)));
    float *d_T_old;		gpuErrchk(cudaMalloc((void**)&d_T_old,		NX * NY * sizeof(float)));
    float *d_T_tex;		gpuErrchk(cudaMalloc((void**)&d_T_tex,		NX * NY * sizeof(float)));
    float *d_T_old_tex;	gpuErrchk(cudaMalloc((void**)&d_T_old_tex,	NX * NY * sizeof(float)));

    gpuErrchk(cudaMemcpy(d_T,			h_T,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_T_tex,		h_T,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_T_old,		d_T,	 NX * NY * sizeof(float), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_T_old_tex,	d_T_tex, NX * NY * sizeof(float), cudaMemcpyDeviceToDevice));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    gpuErrchk(cudaBindTexture2D(NULL, tex_T,	 d_T_tex,     desc, NX, NY, sizeof(float) * NX));
    gpuErrchk(cudaBindTexture2D(NULL, tex_T_old, d_T_old_tex, desc, NX, NY, sizeof(float) * NX));

	// --- Grid size
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid (iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));

    // --- Jacobi iterations on the host
	Jacobi_Iterator_CPU(h_T, h_T_old, NX, NY, MAX_ITER);

	// --- Jacobi iterations on the device
	TimingGPU timerGPU;
	timerGPU.StartCounter();
	for (int k=0; k<MAX_ITER; k=k+2) {
        Jacobi_Iterator_GPU<<<dimGrid, dimBlock>>>(d_T,     d_T_old, NX, NY);   // --- Update d_T_old     starting from data stored in d_T
        Jacobi_Iterator_GPU<<<dimGrid, dimBlock>>>(d_T_old, d_T    , NX, NY);   // --- Update d_T         starting from data stored in d_T_old
    }
	printf("Timing with no texture = %f ms\n", timerGPU.GetCounter());

	// --- Jacobi iterations on the device - texture case
	timerGPU.StartCounter();
	for (int k=0; k<MAX_ITER; k=k+2) {
        Jacobi_Iterator_GPU_texture<<<dimGrid, dimBlock>>>(d_T_tex,     1, NX, NY);   // --- Update d_T_old_tex     starting from data stored in d_T_tex
        Jacobi_Iterator_GPU_texture<<<dimGrid, dimBlock>>>(d_T_old_tex, 0, NX, NY);   // --- Update d_T_tex         starting from data stored in d_T_old_tex
    }
	printf("Timing with texture = %f ms\n", timerGPU.GetCounter());

	saveCPUrealtxt(h_T,     "C:\\Users\\angelo\\Documents\\Project\\Corsi\\Differential_Equations\\Heat_Equation\\2D\\DiffusionEquationJacobi\\DiffusionEquation\\CPU_result.txt",     NX * NY);
	saveGPUrealtxt(d_T_tex, "C:\\Users\\angelo\\Documents\\Project\\Corsi\\Differential_Equations\\Heat_Equation\\2D\\DiffusionEquationJacobi\\DiffusionEquation\\GPU_result_tex.txt", NX * NY);
	saveGPUrealtxt(d_T,     "C:\\Users\\angelo\\Documents\\Project\\Corsi\\Differential_Equations\\Heat_Equation\\2D\\DiffusionEquationJacobi\\DiffusionEquation\\GPU_result.txt",     NX * NY);
	
	// --- Copy result from device to host
    gpuErrchk(cudaMemcpy(h_T_GPU_result, d_T, NX * NY * sizeof(float), cudaMemcpyDeviceToHost));

	// --- Calculate percentage root mean square error between host and device results
	float sum = 0., sum_ref = 0.;
	for (int j=0; j<NY; j++)
		for (int i=0; i<NX; i++) {
			sum     = sum     + (h_T_GPU_result[j * NX + i] - h_T[j * NX + i]) * (h_T_GPU_result[j * NX + i] - h_T[j * NX + i]);
			sum_ref = sum_ref + h_T[j * NX + i]								   * h_T[j * NX + i];
		}
	printf("Percentage root mean square error = %f\n", 100.*sqrt(sum / sum_ref));
	
	// --- Release host memory 
    free(h_T);
	free(h_T_GPU_result);

    // --- Release device memory
    gpuErrchk(cudaFree(d_T));
    gpuErrchk(cudaFree(d_T_old));

    return 0;
}
