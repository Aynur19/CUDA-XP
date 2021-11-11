// Most of the code is borrowed from the source: https://github.com/greyss-mai/Department806/tree/master/2020/Semenov%20S/CUDA

#include "../../_common/helper.cuh"
#include "squareMatrixMultiplication.h"

__global__ void squareMtrixMultiplicationSharedKernel(float* matrixA, float* matrixB, int n, float* matrixC) {
	int blockX = blockIdx.x, blockY = blockIdx.y;
	int threadX = threadIdx.x, threadY = threadIdx.y;

	int beginMatrixA = n * BLOCK_SIZE * blockY;
	int endMatrixA = beginMatrixA + n - 1;

	int beginMatrixB = BLOCK_SIZE * blockX;

	int stepMatrixA = BLOCK_SIZE;
	int stepMatrixB = BLOCK_SIZE * n;

	float sum = 0.0f;

	for (int ia = beginMatrixA, ib = beginMatrixB; ia <= endMatrixA; ia += stepMatrixA, ib += stepMatrixB) {
		__shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

		sharedA[threadY][threadX] = matrixA[ia + n * threadY + threadX];
		sharedB[threadY][threadX] = matrixB[ib + n * threadY + threadX];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; k++) {
			sum += sharedA[threadY][k] * sharedB[k][threadX];
		}

		// Synchronize to make sure submatrices not needed
		__syncthreads();
	}

	matrixC[n * BLOCK_SIZE * blockY + BLOCK_SIZE * blockX + n * threadY + threadX] = sum;
}

argsVector squareMatrixMultiplicationGPU_Shared(argsVector argsIn) {
	argsVector argsOut;
	unsigned int verbose = getValueFromArgs<unsigned int>("--verbose", 0, argsIn);

	float* matrixA = new float[N * N];
	float* matrixB = new float[N * N];
	float* matrixC = new float[N * N];

	int nBytes = N * N * sizeof(float);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);

	matrixFillIndices(matrixA, N, N);
	matrixFillIndices(matrixB, N, N);

	if (verbose == 1) {
		matrixPrint(matrixA, N, N);
		matrixPrint(matrixB, N, N);
	}

	float* devMatixA, * devMatixB, * devMatixC;

	// allocate DRAM
	cudaMalloc((void**)&devMatixA, nBytes);
	cudaMalloc((void**)&devMatixB, nBytes);
	cudaMalloc((void**)&devMatixC, nBytes);

	// copy from CPU to DRAM
	cudaMemcpy(devMatixA, matrixA, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(devMatixB, matrixB, nBytes, cudaMemcpyHostToDevice);

	squareMtrixMultiplicationSharedKernel<<<blocks, threads>>>(devMatixA, devMatixB, N, devMatixC);

	cudaThreadSynchronize();
	cudaMemcpy(matrixC, devMatixC, nBytes, cudaMemcpyDeviceToHost);

	// free GPU memory
	cudaFree(devMatixA);
	cudaFree(devMatixB);
	cudaFree(devMatixC);

	if (verbose == 1) {
		matrixPrint(matrixC, N, N);
	}

	return argsOut;
}

