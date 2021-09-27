#ifndef __HELPER_H__
#define __HELPER_H__ 

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define EPS 0.00001f

extern void randomInit(float* randArray, int n);

extern void vecAddValidate(float* vec1, float* vec2, float* vecSum, int numItems);

extern void numbersInit(float* nArray, int n);

extern void printMatrix(float* nArray, int n, char* matrixName);

#endif