#include <cassert>

#include "chapter3.h"

// ���� ����������� ���������� �������
__global__ void tableKernel(float* devPtr, float step) {
	// ��������� ����������� ������ ����
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// ���������� �������� ���������
	float x = step * index;

	// ������������� "�������" ������ �������
	devPtr[index] = sinf(sqrtf(x));

	printf("Item %d: sin(sqrt(%.3f)) = %.3f\n", index, step * index, devPtr[index]);
}

extern "C" void cuBuildTable(float* res, int n, float step) {
	float* devPtr;

	// �������� ����, ��� n ������ 256
	assert(n % 256 == 0);

	// ��������� ���������� ������ ��� �������
	cudaMalloc(&devPtr, n * sizeof(float));

	// ������ ���� ��� ���������� ��������
	tableKernel<<<dim3(n / 256), dim3(256)>>>(devPtr, step);

	// ����������� ���������� �� ���������� ������ � ������ CPU
	cudaMemcpy(res, devPtr, n * sizeof(float), cudaMemcpyDeviceToHost);

	// ������������ ���������� ������
	cudaFree(devPtr);

	for (int i = 0; i < n; i++)
	{
		printf("Item %d: sin(sqrt(%.3f)) = %.3f\n", i, step * i, res[i]);
	}
}