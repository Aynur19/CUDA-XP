// Most of the code is borrowed from the source: https://github.com/greyss-mai/Department806/tree/master/2020/Semenov%20S/CUDA

#include "../../_common/helper.cuh"
#include "squareMatrixMultiplication.h"

__global__ void squareMatrixMultiplicationGlobalKernel(float *matrixA, float *matrixB, int n, float *matrixC)
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

void squareMatrixMultiplicationGPU_Global(int argc, char* argv[]) {
	unsigned int verbose = getValueFromArgv<unsigned int>("verbose", 0, argc, argv);

	printf("BLOCK_SIZE = %d", BLOCK_SIZE);
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

	squareMatrixMultiplicationGlobalKernel<<<blocks, threads>>>(devMatixA, devMatixB, N, devMatixC);

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

