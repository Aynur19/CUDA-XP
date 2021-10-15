#ifndef __HELPER_CUH__
#define __HELPER_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>

#include "helper_cuda.h"
#include "helper.h"

using namespace std;

#pragma region Get ThreadID
// 1D grid of 1D blocks
__device__ int getGlobalIdx_1D_1D();

// 1D grid of 2D blocks
__device__ int getGlobalIdx_1D_2D();

// 1D grid of 3D blocks
__device__ int getGlobalIdx_1D_3D();

// 2D grid of 1D blocks
__device__ int getGlobalIdx_2D_1D();

// 2D grid of 2D blocks
__device__ int getGlobalIdx_2D_2D();

// 2D grid of 3D blocks
__device__ int getGlobalIdx_2D_3D();

// 3D grid of 1D blocks
__device__ int getGlobalIdx_3D_1D();

// 3D grid of 2D blocks
__device__ int getGlobalIdx_3D_2D();

// 3D grid of 3D blocks
__device__ int getGlobalIdx_3D_3D();
#pragma endregion

argsVector gpuTimeMeasuring(argsVector (*gpuComputedMethod)(argsVector argsIn), unsigned int iters, argsVector argsIn);

#endif // !__HELPER_CUH__

