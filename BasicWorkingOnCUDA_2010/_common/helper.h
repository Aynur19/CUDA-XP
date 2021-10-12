#ifndef __HELPER_H__
#define __HELPER_H__ 

#include <cstdlib>
#include <cstdio>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define EPS 0.00001f

extern void randomInit(float* randArray, int n);

void randomInitF3(float3* vec, int n);

extern void vecAddValidate(float* vec1, float* vec2, float* vecSum, int numItems);

extern void numbersInit(float* nArray, int n);

extern void printMatrix(float* nArray, int n, char* matrixName);

#endif