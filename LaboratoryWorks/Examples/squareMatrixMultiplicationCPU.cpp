#include "squareMatrixMultiplication.h"

void squareMatrixMultiplicationCPU(unsigned int verbose) {
	float* matrixA = new float[N * N];
	float* matrixB = new float[N * N];
	float* matrixC = new float[N * N];

	matrixFillIndices(matrixA, N, N);
	matrixFillIndices(matrixB, N, N);

	if (verbose == 1) {
		matrixPrint(matrixA, N, N);
		matrixPrint(matrixB, N, N);
	}

	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			float sum = 0;
			for (int k = 0; k < N; k++) {
				sum += matrixA[MATRIX_INDEX(row, k, N)] * matrixB[MATRIX_INDEX(k, col, N)];
			}
			matrixC[MATRIX_INDEX(row, col, N)] = sum;
		}
	}

	if (verbose == 1) {
		matrixPrint(matrixC, N, N);
	}
}

