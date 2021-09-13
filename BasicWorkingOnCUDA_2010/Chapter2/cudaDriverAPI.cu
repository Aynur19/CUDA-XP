// Сложение векторов через CUDA driver API
//#include <cuda.h>
//
//#include "helper.h"
//#include "vector.cu"

//TODO: FIX ERRORS

//#pragma comment(lib, "cuda.lib" )
//extern "C" void cuVectorAdd_DAPI(const int blockSize, const int numBlocks, const int numItems)
//{
//
//	CUdevice hDevice;
//	CUcontext hContext;
//	CUmodule hModule;
//	CUfunction hFunction;
//
//	// Инициализация и выбор первого GPU
//	cuInit(0);
//	cuDeviceGet(&hDevice, 0);
//
//	// Создание для него контекста
//	cuCtxCreate(&hContext, 0, hDevice);
//
//	// Загрузка модуля и получение адреса ядра
//	cuModuleLoad(&hModule, "vector.cubin");
//	cuModuleGetFunction(&hFunction, hModule, "vectorAdd");
//
//	// Выделение памяти на CPU
//	float* vec1 = new float[numItems];
//	float* vec2 = new float[numItems];
//	float* vecSum = new float[numItems];
//
//	// Заполнение массива случайными числами
//	randomInit(vec1, numItems);
//	randomInit(vec2, numItems);
//
//	// Выделение памяти GPU
//	CUdeviceptr vecDev1;
//	CUdeviceptr vecDev2;
//	CUdeviceptr vecSumDev;
//
//	cuMemAlloc(&vecDev1, numItems * sizeof(float));
//	cuMemAlloc(&vecDev2, numItems * sizeof(float));
//	cuMemAlloc(&vecSumDev, numItems * sizeof(float));
//
//	// Копирование входных векторов в память GPU
//	cuMemcpyHtoD(vecDev1, vec1, numItems * sizeof(float));
//	cuMemcpyHtoD(vecDev2, vec2, numItems * sizeof(float));
//
//	// Настройка передачи параметров ядру
//	cuFuncSetBlockShape(hFunction, blockSize, 1, 1);
//
//	#define ALIGN_UP(offset, aligment) (offset) = ((offset)+(aligment)-1) & ~((aligment)-1);
//
//	int offset = 0;
//	void* ptr = (void*)(size_t)vecDev1;
//
//	ALIGN_UP(offset, __alignof(ptr));
//
//	cuParamSetv(hFunction, offset, &ptr, sizeof(ptr));
//
//	offset += sizeof(ptr);
//
//	cuParamSetSize(hFunction, offset);
//	
//	// Выполнение ядра
//	cuLaunchGrid(hFunction, numBlocks, 1);
//
//	// Скопировать результат в память CPU
//	cuMemcpyDtoH((void*)vecSum, vecSumDev, numItems * sizeof(float));
//
//	// Проверка результута
//	vecAddValidate(vec1, vec2, vecSum, numItems);
//
//	// Освобождение памяти
//	delete[] vec1;
//	delete[] vec2;
//	delete[] vecSum;
//
//	cuMemFree(vecDev1);
//	cuMemFree(vecDev2);
//	cuMemFree(vecSumDev);
//}