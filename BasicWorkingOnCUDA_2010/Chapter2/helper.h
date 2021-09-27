#ifndef __HELPER_H__
#define __HELPER_H__ 

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define EPS 0.00001f

void randomInit(float* randArray, int n);

void vecAddValidate(float* vec1, float* vec2, float* vecSum, int numItems);

void numbersInit(float* nArray, int n);

void printMatrix(float* nArray, int n, char* matrixName);

#endif