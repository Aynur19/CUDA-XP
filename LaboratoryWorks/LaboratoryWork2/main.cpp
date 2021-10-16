#include "rootEquationFinding.h"

argsVector(*cpuComputedMethod)(argsVector argsIn) = NULL;

argsVector(*gpuComputedMethod)(argsVector argsIn) = NULL;

void rootEquationFinding(argsVector argsIn) {
	printf("TASK: FINDING ROOT OF 'SIN(X) = 1/X' EQUATION\n");

	printf("****************************   CPU   ***************************\n");
	cpuComputedMethod = &rootEquationFindingCPU;
	cpuTimeMeasuring(cpuComputedMethod, 10, argsIn);
	printf("****************************   CPU   ***************************\n");

	printf("\n");

	printf("*******************   CUDA [GLOBAL MEMORY]   *******************\n");
	gpuComputedMethod = &rootEquationFindingGPU_Global;
	gpuTimeMeasuring(gpuComputedMethod, 10, argsIn);
	printf("*******************   CUDA [GLOBAL MEMORY]   *******************\n");

	printf("\n");

	printf("*******************   CUDA [SHARED MEMORY]   *******************\n");
	gpuComputedMethod = &rootEquationFindingGPU_Shared;
	gpuTimeMeasuring(gpuComputedMethod, 10, argsIn);
	printf("*******************   CUDA [SHARED MEMORY]   *******************\n");
}

int main(int argc, char* argv[])
{
	std::vector<std::string> args;

	args.push_back("--startX 0.0");
	args.push_back("--endX 10.0");
	args.push_back("--stepX 0.000001");

	rootEquationFinding(args);

	return 0;
}