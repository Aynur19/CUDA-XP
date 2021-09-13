#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define EPS 0.00001f

#pragma once
// Заполнение массива случайными числами
void randomInit(float* randArray, int n)
{
	for (int i = 0; i < n; i++)
	{
		randArray[i] = rand() / (float)RAND_MAX;
	}
}

// Проверка результута
void vecAddValidate(float* vec1, float* vec2, float* vecSum, int numItems)
{
	for (int i = 0; i < numItems; i++)
	{
		if (fabs(vec1[i] + vec2[i] - vecSum[i]) > EPS)
		{
			printf("Error at index %d\n", i);
		}
		else
		{
			printf("Sum at index %d: %f = %f + %f\n", i, vecSum[i], vec1[i], vec2[i]);
		}
	}
}