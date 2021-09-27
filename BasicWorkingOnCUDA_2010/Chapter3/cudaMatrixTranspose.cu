#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "chapter3.h"

__global__ void __cuMatrixSquareTranspose(float* inData, float* outData, int n) {
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int inIndex = xIndex + n * yIndex;	
	unsigned int outIndex = yIndex + n * xIndex;

	outData[outIndex] = inData[inIndex];

	/*printf("InData Item %d: %.f\tOutData Item %d: %.f\t|xIndex: %d\t|yIndex: %d\n", 
		inIndex, inData[inIndex], outIndex, outData[outIndex], xIndex, yIndex);*/
}

extern "C" void cuMatrixSquareTranspose() {
	const unsigned int n = 16;
	
	float* inData = new float[n * n];
	float* outData = new float[n * n];

	numbersInit(inData, n * n);

	printMatrix(inData, n, "Matrix A");

	float* devInData;
	float* devOutData;

	// выделение глобальной памяти под таблицу
	cudaMalloc(&devInData, n * n * sizeof(float));
	cudaMalloc(&devOutData, n * n * sizeof(float));

	cudaMemcpy(devInData, inData, n * n * sizeof(float), cudaMemcpyHostToDevice);

	// запуск ядра для вычисления значений
	__cuMatrixSquareTranspose<<<dim3(1, 1), dim3(n, n)>>>(devInData, devOutData, n);

	// копирование результата из глобальной памяти в память CPU
	cudaMemcpy(outData, devOutData, n * n * sizeof(float), cudaMemcpyDeviceToHost);

	// освобождение выделенной памяти
	cudaFree(devInData);
	cudaFree(devOutData);

	printMatrix(outData, n, "Matrix B");
}