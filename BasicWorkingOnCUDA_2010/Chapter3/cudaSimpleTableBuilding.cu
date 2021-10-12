#include <cassert>

#include "chapter3.h"

// ядро осуществляе заполнение массива
__global__ void tableKernel(float* devPtr, float step) {
	// получение глобального адреса нити
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// вычисление значения аргумента
	float x = step * index;

	// использование "быстрой" версии функции
	devPtr[index] = sinf(sqrtf(x));

	printf("Item %d: sin(sqrt(%.3f)) = %.3f\n", index, step * index, devPtr[index]);
}

extern "C" void cuBuildTable(float* res, int n, float step) {
	float* devPtr;

	// проверка того, что n кратно 256
	assert(n % 256 == 0);

	// выделение глобальной памяти под таблицу
	cudaMalloc(&devPtr, n * sizeof(float));

	// запуск ядра для вычисления значений
	tableKernel<<<dim3(n / 256), dim3(256)>>>(devPtr, step);

	// копирование результата из глобальной памяти в память CPU
	cudaMemcpy(res, devPtr, n * sizeof(float), cudaMemcpyDeviceToHost);

	// освобождение выделенной памяти
	cudaFree(devPtr);

	for (int i = 0; i < n; i++)
	{
		printf("Item %d: sin(sqrt(%.3f)) = %.3f\n", i, step * i, res[i]);
	}
}