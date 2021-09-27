// Сложение векторов через CUDA runtime API
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "chapter2.h"

__global__ void __cuVectorAdd(float* vec1, float* vec2, float* vecSum) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	vecSum[index] = vec1[index] + vec2[index];
}

extern "C" void cuVectorAdd_RAPI(const int blockSize, const int numBlocks, const int numItems)
{
	// Выбор первого GPU для работы
	cudaSetDevice(0);

	// Выделение памяти CPU
	float* vec1 = new float[numItems];
	float* vec2 = new float[numItems];
	float* vecSum = new float[numItems];

	// Инициализация входных массивов
	randomInit(vec1, numItems);
	randomInit(vec2, numItems);

	// Выделение памяти GPU
	float* vecDev1 = NULL;
	float* vecDev2 = NULL;
	float* vecSumDev = NULL;

	cudaMalloc((void**)&vecDev1, numItems * sizeof(float));
	cudaMalloc((void**)&vecDev2, numItems * sizeof(float));
	cudaMalloc((void**)&vecSumDev, numItems * sizeof(float));

	// Копирование данных из памяти CPU в память GPU
	cudaMemcpy(vecDev1, vec1, numItems * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(vecDev2, vec2, numItems * sizeof(float), cudaMemcpyHostToDevice);

	// Запуск ядра
	__cuVectorAdd<<<numBlocks, blockSize>>>(vecDev1, vecDev2, vecSumDev);

	// Копирование результата в память CPU
	cudaMemcpy((void*)vecSum, vecSumDev, numItems * sizeof(float), cudaMemcpyDeviceToHost);

	// Проверка результата
	vecAddValidate(vec1, vec2, vecSum, numItems);

	// Освобождение выделенной памяти
	delete[] vec1;
	delete[] vec2;
	delete[] vecSum;

	cudaFree(vecDev1);
	cudaFree(vecDev2);
	cudaFree(vecSumDev);
}
