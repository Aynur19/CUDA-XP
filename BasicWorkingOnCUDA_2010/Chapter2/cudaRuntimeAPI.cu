// Сложение векторов через CUDA runtime API
#include <cuda_runtime.h>

#include "vector.cu"
#include "helper.h"

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
	vectorAdd<<<numBlocks, blockSize>>>(vecDev1, vecDev2, vecSumDev);

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

//#include <stdio.h>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//
//// ядро, выполняется на большом числе нитей
//__global__ void sumKernel(float* a, float* b, float* c)
//{
//	// глобальный индекс нити
//	int idx = threadIdx.x + blockIdx.x * blockDim.x;
//	printf("Hello World! %d\n", idx);
//
//	// выполнить обработку соответствующих данной нити данных
//	c[idx] = a[idx] + b[idx];
//	printf("псмрмоь %d = %d + %d\n", c[idx], a[idx], b[idx]);
//}
//
//void sum(float* a, float* b, float* c, int n)
//{
//	int numBytes = n * sizeof(float);
//	float* aDev = 0;
//	float* bDev = 0;
//	float* cDev = 0;
//
//	cudaSetDevice(0);
//	// выделить память на GPU
//	cudaMalloc((void**)&aDev, numBytes);
//	cudaMalloc((void**)&bDev, numBytes);
//	cudaMalloc((void**)&cDev, numBytes);
//
//	// задать конфигурацию запуска n нитей
//	dim3 threads = dim3(3, 1);
//	dim3 blocks = dim3(n / threads.x, 1);
//
//	// скопировать входные данные из памяти CPU в память GPU
//	cudaMemcpy(aDev, a, numBytes, cudaMemcpyHostToDevice);
//	cudaMemcpy(bDev, b, numBytes, cudaMemcpyHostToDevice);
//
//	// вызвать ядро с заданной конфигурацией для обработки данных
//	sumKernel<<<blocks, threads>>>(aDev, bDev, cDev);
//	//sumKernel<<<1, n>>>(aDev, bDev, cDev);
//
//	cudaDeviceSynchronize();
//
//	// скопировать результаты в память CPU
//	cudaMemcpy(c, cDev, numBytes, cudaMemcpyDeviceToHost);
//
//	// освободить выделенную память
//	cudaFree(aDev);
//	cudaFree(bDev);
//	cudaFree(cDev);
//}
//
//int main() 
//{
//	const int arraySize = 5;
//	float a[arraySize] = { 1, 2, 3, 4, 5 };
//	float b[arraySize] = { 10, 20, 30, 40, 50 };
//	float c[arraySize] = { 0 };
//
//	sum(a, b, c, arraySize);
//
//	return 0;
//}

//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
