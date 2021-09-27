#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "helper.h"

// Заполнение массива случайными числами
extern void randomInit(float* randArray, int n) {
	for (int i = 0; i < n; i++) {
		randArray[i] = rand() / (float)RAND_MAX;
	}
}
// Проверка результата
extern void vecAddValidate(float* vec1, float* vec2, float* vecSum, int numItems) {
	for (int i = 0; i < numItems; i++) {
		if (fabs(vec1[i] + vec2[i] - vecSum[i]) > EPS) {
			printf("Error at index %d\n", i);
		}
		else {
			printf("Sum at index %d: %f = %f + %f\n", i, vecSum[i], vec1[i], vec2[i]);
		}
	}
}

void numbersInit(float* nArray, int n) {
	for (int i = 0; i < n; i++) {
		nArray[i] = (float)i;
	}
}

void printMatrix(float* nArray, int n, char* matrixName) {
	printf("===============================================================\n");
	printf("%s\n", matrixName);
	for (int i = 0; i < n; i++)
	{
		for (int k = 0; k < n; k++)
		{
			printf("|%.f\t", nArray[i * n + k]);
		}
		printf("|\n");
	}
	printf("===============================================================\n");
}