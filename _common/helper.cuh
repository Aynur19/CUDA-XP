#ifndef __HELPER_CUH__
#define __HELPER_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"

void gpuTimeMeasuring(void (*gpuComputedMethod)(int argc, char* argv[]), unsigned int iters, int argc, char* argv[]);

#endif // !__HELPER_CUH__

