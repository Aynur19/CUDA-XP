#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../_common/helper_cuda.h"
#include "../../_common/helper.h"

#define N 1024
#define BLOCK_SIZE 16

__global__ void matrixMultiplicationGlobalKernel(float *matrixA, float *matrixB, int n, float *matrixC)
{
	int blockX = blockIdx.x;		
	int blockY = blockIdx.y;		
	int threadX = threadIdx.x;		
	int threadY = threadIdx.y;
	
	float sum = 0.0f;
	
	int indexA = n * BLOCK_SIZE * blockY + n * threadY;
	int indexB = BLOCK_SIZE * blockX + threadX;
	int indexC = n * BLOCK_SIZE * blockY + BLOCK_SIZE * blockX;

	for (int k = 0; k < n; k++) {
		sum += matrixA[indexA + k] * matrixB[indexB + k * n];
	}

	matrixC[indexC + n * threadY + threadX] = sum;
}

extern "C" void matrixMultiplicationGPU_Global(unsigned int verbose) {
	float *matrixA = new float[N * N];
	float *matrixB = new float[N * N];
	float *matrixC = new float[N * N];
	
	int nBytes = N * N * sizeof(float);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);

	matrixFillIndices(matrixA, N, N);
	matrixFillIndices(matrixB, N, N);

	if (verbose == 1) {
		matrixPrint(matrixA, N, N);
		matrixPrint(matrixB, N, N);
	}

	float *devMatixA, *devMatixB, *devMatixC;

	// allocate DRAM
	cudaMalloc((void**)&devMatixA, nBytes);
	cudaMalloc((void**)&devMatixB, nBytes);
	cudaMalloc((void**)&devMatixC, nBytes);

	// copy from CPU to DRAM
	cudaMemcpy(devMatixA, matrixA, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(devMatixB, matrixB, nBytes, cudaMemcpyHostToDevice);

	matrixMultiplicationGlobalKernel<<<blocks, threads>>>(devMatixA, devMatixB, N, devMatixC);

	cudaThreadSynchronize();
	cudaMemcpy(matrixC, devMatixC, nBytes, cudaMemcpyDeviceToHost);

	// free GPU memory
	cudaFree(devMatixA);
	cudaFree(devMatixB);
	cudaFree(devMatixC);

	if (verbose == 1) {
		matrixPrint(matrixC, N, N);
	}
}

extern "C" void matrixMultiplicationCPU(unsigned int verbose) {
	float* matrixA = new float[N * N];
	float* matrixB = new float[N * N];
	float* matrixC = new float[N * N];

	matrixFillIndices(matrixA, N, N);
	matrixFillIndices(matrixB, N, N);

	if (verbose == 1) {
		matrixPrint(matrixA, N, N);
		matrixPrint(matrixB, N, N);
	}

	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			float sum = 0;
			for (int k = 0; k < N; k++) {
				sum += matrixA[MATRIX_INDEX(row, k, N)] * matrixB[MATRIX_INDEX(k, col, N)];
			}
			matrixC[MATRIX_INDEX(row, col, N)] = sum;
		}
	}

	if (verbose == 1) {
		matrixPrint(matrixC, N, N);
	}
}

extern "C" void matrixMultiplication(unsigned int iters, unsigned int verbose) {

	printf("MATRIX MULTIPLICATION ON CPU\n");
	
	printf("****************************   CPU   ***************************\n");
	cpuComputedMethod = &matrixMultiplicationCPU;
	cpuTimeMeasuring(cpuComputedMethod, iters, verbose);
	printf("****************************   CPU   ***************************\n");

	printf("\n\n");
	
	printf("MATRIX MULTIPLICATION ON GPU\n");

	printf("********************   CUDA GLOBAL MEMORY   ********************\n");
	gpuComputedMethod = &matrixMultiplicationGPU_Global;
	gpuTimeMeasuring(gpuComputedMethod, iters, verbose);
	printf("********************   CUDA GLOBAL MEMORY   ********************\n");

	// cudaDeviceReset causes the driver to clean up all state. 
	// While not mandatory in normal operation, it is good practice.  
	// It is also needed to ensure correct operation when the application is being profiled. 
	// Calling cudaDeviceReset causes all profile data to be flushed before the application exits
	cudaDeviceReset();
}



