#include "rootEquationFinding.h"

/// <summary>
/// Ядро вычисления корня уравнения (sin(x)=1/x) на GPU с использование Global Memory.
/// </summary>
__global__ void rootEquationFindingGlobalKernal(float* devArrayX, float startX, float stepX) {
	int threadId = getGlobalIdx_2D_2D();

	stepX *= threadId;
	float currentX = startX + stepX;
	float resultX = 1 / (sin(M_PI * currentX / 180));

	if (fabs(resultX - currentX) <= EPS) {
		devArrayX[0] = currentX;
	}
}

argsVector rootEquationFindingGPU_Global(argsVector argsIn) {
	argsVector argsOut;
	float startX = getValueFromArgs<float>("--startX", 0, argsIn);
	float endX = getValueFromArgs<float>("--endX", 0, argsIn);
	float stepX = getValueFromArgs<float>("--stepX", 0, argsIn);

	auto params = getOptimalParameters(startX, endX, stepX, BLOCK_DIM);
	dim3 gridDimension = std::get<0>(params);
	stepX = getSignedStep(startX, endX, std::get<1>(params));

	printf("Grid Dimension: (%d, %d, %d)\t Block Dimension: (%d, %d, %d)\tstepX: %.9f\n",
		gridDimension.x, gridDimension.y, gridDimension.z, BLOCK_DIM.x, BLOCK_DIM.y, BLOCK_DIM.z, stepX);

	int nBytes = sizeof(float);
	float* arrayX = new float[1];
	float* devArrayX;

	// allocate DRAM
	cudaMalloc((void**)&devArrayX, nBytes);
	cudaMemset(devArrayX, 0, nBytes);

	rootEquationFindingGlobalKernal<<<gridDimension, BLOCK_DIM>>>(devArrayX, startX, stepX);

	cudaThreadSynchronize();
	cudaMemcpy(arrayX, devArrayX, nBytes, cudaMemcpyDeviceToHost);

	// free GPU memory
	cudaFree(devArrayX);

	checkRootEquationFinding(arrayX[0]);
	argsOut.push_back("--root " + std::to_string(arrayX[0]));
	return argsOut;
}
