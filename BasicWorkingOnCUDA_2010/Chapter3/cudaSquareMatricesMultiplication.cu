#include "chapter3.h"

#define BLOCK_SIZE 9

__global__ void __cuSquareMatricesMultiplication(float* matrixA, float* matrixB, int n, float* matrixC) {
	
	// ������� �����
	int xBlock = blockIdx.x;
	int yBlock = blockIdx.y;

	// ������� ������� ������ ����� 
	int xThread = threadIdx.x;
	int yThread = threadIdx.y;

	// ����� ������������� ���������
	float sum = 0.0f;

	// �������� ��� ������� A[i][0]
	int iMatrixA = n * BLOCK_SIZE * yBlock + n * yThread;

	// �������� ��� ������� B[0][j]
	int iMatrixB = BLOCK_SIZE * xBlock + xThread;

	// ����������� � ���������
	for (int k = 0; k < n; k++)
	{
		sum += matrixA[iMatrixA + k] * matrixB[iMatrixB + k * n];
	}

	// ���������� ���������� � ���������� ������
	// �������� ��� ������������� ��������
	int iMatrixC = n * BLOCK_SIZE * yBlock + BLOCK_SIZE * xBlock;

	matrixC[iMatrixC + n * yThread + xThread] = sum;

	printf("Item %d: %.f\n", iMatrixC + n * yThread + xThread, sum);
}

extern "C" void cuSquareMatricesMultiplication() {
	const unsigned int n = 3;

	float* matrixA = new float[n * n];
	float* matrixB = new float[n * n];
	float* matrixC = new float[n * n];

	numbersInit(matrixA, n * n);
	numbersInit(matrixB, n * n);

	printMatrix(matrixA, n, "Matrix A");
	printMatrix(matrixB, n, "Matrix B");

	float* devMatrixA;
	float* devMatrixB;
	float* devMatrixC;

	// ��������� ���������� ������ ��� �������
	cudaMalloc(&devMatrixA, n * n * sizeof(float));
	cudaMalloc(&devMatrixB, n * n * sizeof(float));
	cudaMalloc(&devMatrixC, n * n * sizeof(float));

	cudaMemcpy(devMatrixA, matrixA, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devMatrixB, matrixB, n * n * sizeof(float), cudaMemcpyHostToDevice);

	// ������ ���� ��� ���������� ��������
	__cuSquareMatricesMultiplication<<<dim3(1, 1), dim3(n, n)>>>(devMatrixA, devMatrixB, n, devMatrixC);

	// ����������� ���������� �� ���������� ������ � ������ CPU
	cudaMemcpy(matrixC, devMatrixC, n * n * sizeof(float), cudaMemcpyDeviceToHost);

	// ������������ ���������� ������
	cudaFree(devMatrixA);
	cudaFree(devMatrixB);
	cudaFree(devMatrixC);

	printMatrix(matrixC, n, "Matrix C");
}