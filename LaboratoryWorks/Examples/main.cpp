#include "../../_common/helper.h"
#include "../../_common/helper.cuh"
#include "squareMatrixMultiplication.h"


void (*cpuComputedMethod)(unsigned int verbose) = NULL;

void (*gpuComputedMethod)(unsigned int verbose) = NULL;

void squareMatrixMultiplication(unsigned int iters, unsigned int verbose) {
	printf("MATRIX MULTIPLICATION ON CPU\n");
	
	printf("****************************   CPU   ***************************\n");
	cpuComputedMethod = &squareMatrixMultiplicationCPU;
	cpuTimeMeasuring(cpuComputedMethod, iters, verbose);
	printf("****************************   CPU   ***************************\n");

	printf("\n\n");
	printf("MATRIX MULTIPLICATION ON GPU\n");

	printf("********************   CUDA GLOBAL MEMORY   ********************\n");
	gpuComputedMethod = &squareMatrixMultiplicationGPU_Global;
	gpuTimeMeasuring(gpuComputedMethod, iters, verbose);
	printf("********************   CUDA GLOBAL MEMORY   ********************\n");

	printf("\n\n");

	printf("********************   CUDA SHARED MEMORY   ********************\n");
	gpuComputedMethod = &squareMatrixMultiplicationGPU_Shared;
	gpuTimeMeasuring(gpuComputedMethod, iters, verbose);
	printf("********************   CUDA SHARED MEMORY   ********************\n");
}

int main(int argc, char* argv[])
{
	squareMatrixMultiplication(10, 0);

	return 0;
}