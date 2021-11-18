  #include "labWork3.h"

argsVector(*cpuComputedMethod)(argsVector argsIn) = NULL;

argsVector(*gpuComputedMethod)(argsVector argsIn) = NULL;

void option19_rootEquationFinding(argsVector argsIn) {
	printf("TASK: FINDING ROOT OF '0.17x^(3) - 0.57x^(2) - 1.6x + 3.7 = 0' EQUATION\n");

	printf("****************************   CPU   ***************************\n");
	cpuComputedMethod = &option19_rootEquationFindingCPU;
	cpuTimeMeasuring(cpuComputedMethod, 1, argsIn);
	printf("****************************   CPU   ***************************\n");

	printf("\n");

	printf("****************************   GPU   ***************************\n");
	gpuComputedMethod = &option19_rootEquationFindingGPU_Thrust;
	gpuTimeMeasuring(gpuComputedMethod, 1, argsIn);
	printf("****************************   GPU   ***************************\n");

	printf("\n");
}

int main(int argc, char* argv[]) {
	std::vector<std::string> args;

	args.push_back("--startX -1000.0");
	args.push_back("--endX 1000.0");

	option19_rootEquationFinding(args);
}