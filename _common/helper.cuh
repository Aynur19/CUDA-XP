#ifndef __HELPER_CUH__
#define __HELPER_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"

void gpuTimeMeasuring(void (*gpuComputedMethod)(unsigned int verbose), unsigned int iters, unsigned int verbose);

#endif // !__HELPER_CUH__

