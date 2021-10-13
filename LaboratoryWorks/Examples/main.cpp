#include "../../_common/helper.h"
#include "../../_common/helper.cuh"
#include "squareMatrixMultiplication.h"


void (*cpuComputedMethod)(int argc, char* argv[]) = NULL;

void (*gpuComputedMethod)(int argc, char* argv[]) = NULL;


void squareMatrixMultiplication(unsigned int iters, unsigned int verbose) {
	char* argv[] = { "verbose", "0" };
	int argc = 2;

	printf("MATRIX MULTIPLICATION ON CPU\n");
	
	printf("****************************   CPU   ***************************\n");
	cpuComputedMethod = &squareMatrixMultiplicationCPU;
	cpuTimeMeasuring(cpuComputedMethod, 10, argc, argv);
	printf("****************************   CPU   ***************************\n");

	printf("\n\n");
	printf("MATRIX MULTIPLICATION ON GPU\n");

	printf("********************   CUDA GLOBAL MEMORY   ********************\n");
	gpuComputedMethod = &squareMatrixMultiplicationGPU_Global;
	gpuTimeMeasuring(gpuComputedMethod, 10, argc, argv);
	printf("********************   CUDA GLOBAL MEMORY   ********************\n");

	printf("\n\n");

	printf("********************   CUDA SHARED MEMORY   ********************\n");
	gpuComputedMethod = &squareMatrixMultiplicationGPU_Shared;
	gpuTimeMeasuring(gpuComputedMethod, 10, argc, argv);
	printf("********************   CUDA SHARED MEMORY   ********************\n");
}

int main(int argc, char* argv[])
{
	squareMatrixMultiplication(10, 0);

	return 0;
}