// �������� �������� ����� CUDA driver API
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
//	// ������������� � ����� ������� GPU
//	cuInit(0);
//	cuDeviceGet(&hDevice, 0);
//
//	// �������� ��� ���� ���������
//	cuCtxCreate(&hContext, 0, hDevice);
//
//	// �������� ������ � ��������� ������ ����
//	cuModuleLoad(&hModule, "vector.cubin");
//	cuModuleGetFunction(&hFunction, hModule, "vectorAdd");
//
//	// ��������� ������ �� CPU
//	float* vec1 = new float[numItems];
//	float* vec2 = new float[numItems];
//	float* vecSum = new float[numItems];
//
//	// ���������� ������� ���������� �������
//	randomInit(vec1, numItems);
//	randomInit(vec2, numItems);
//
//	// ��������� ������ GPU
//	CUdeviceptr vecDev1;
//	CUdeviceptr vecDev2;
//	CUdeviceptr vecSumDev;
//
//	cuMemAlloc(&vecDev1, numItems * sizeof(float));
//	cuMemAlloc(&vecDev2, numItems * sizeof(float));
//	cuMemAlloc(&vecSumDev, numItems * sizeof(float));
//
//	// ����������� ������� �������� � ������ GPU
//	cuMemcpyHtoD(vecDev1, vec1, numItems * sizeof(float));
//	cuMemcpyHtoD(vecDev2, vec2, numItems * sizeof(float));
//
//	// ��������� �������� ���������� ����
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
//	// ���������� ����
//	cuLaunchGrid(hFunction, numBlocks, 1);
//
//	// ����������� ��������� � ������ CPU
//	cuMemcpyDtoH((void*)vecSum, vecSumDev, numItems * sizeof(float));
//
//	// �������� ����������
//	vecAddValidate(vec1, vec2, vecSum, numItems);
//
//	// ������������ ������
//	delete[] vec1;
//	delete[] vec2;
//	delete[] vecSum;
//
//	cuMemFree(vecDev1);
//	cuMemFree(vecDev2);
//	cuMemFree(vecSumDev);
//}