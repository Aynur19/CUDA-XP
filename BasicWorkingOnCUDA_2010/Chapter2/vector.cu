#include <device_launch_parameters.h>


__global__ void vectorAdd(float* vec1, float* vec2, float* vecSum)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	vecSum[index] = vec1[index] + vec2[index];
}
