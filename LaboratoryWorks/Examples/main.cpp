#include "../../_common/helper.h"
#include "../../_common/helper.cuh"
#include "squareMatrixMultiplication.h"


argsVector(*cpuComputedMethod)(argsVector argsIn) = NULL;

argsVector(*gpuComputedMethod)(argsVector argsIn) = NULL;


void squareMatrixMultiplication(unsigned int iters, unsigned int verbose) {
	argsVector argsIn;
	argsIn.push_back("--verbose " + std::to_string(verbose));
	
	printf("MATRIX MULTIPLICATION ON CPU\n");
	
	printf("****************************   CPU   ***************************\n");
	cpuComputedMethod = &squareMatrixMultiplicationCPU;
	cpuTimeMeasuring(cpuComputedMethod, 10, argsIn);
	printf("****************************   CPU   ***************************\n");

	printf("\n\n");
	printf("MATRIX MULTIPLICATION ON GPU\n");

	printf("********************   CUDA GLOBAL MEMORY   ********************\n");
	gpuComputedMethod = &squareMatrixMultiplicationGPU_Global;
	gpuTimeMeasuring(gpuComputedMethod, 10, argsIn);
	printf("********************   CUDA GLOBAL MEMORY   ********************\n");

	printf("\n\n");

	printf("********************   CUDA SHARED MEMORY   ********************\n");
	gpuComputedMethod = &squareMatrixMultiplicationGPU_Shared;
	gpuTimeMeasuring(gpuComputedMethod, 10, argsIn);
	printf("********************   CUDA SHARED MEMORY   ********************\n");
}

int main(int argc, char* argv[])
{
	squareMatrixMultiplication(10, 0);

	return 0;
}