#include <curand_kernel.h>

#include "labWork3.h"

__host__ __device__ float getEquationValue_Option19(float x) {
    return 0.17 * powf(x, 3.0) - 0.57 * powf(x, 2.0) - 1.6 * x + 3.7;
}

__global__ void option19_rootEquationFindingKernel(unsigned int seed, float* result, float startX, float endX) {
    int threadId = getGlobalIdx_1D_1D();
    
    curandState_t state;
    curand_init(seed, threadId, 0, &state);

    float randNumber = curand_uniform(&state) - 0.5;

    float currentLeftX = randNumber * startX;
    float currentRightX = randNumber * endX;

    float currentLeftY = getEquationValue_Option19(currentLeftX);
    float currentRightY = getEquationValue_Option19(currentRightX);

    if (fabs(currentLeftY) < EPS) {
        result[0] = currentLeftX;
        result[1] = currentLeftY;
    }
    else if(fabs(currentRightY)<EPS) {
        result[0] = currentRightX;
        result[1] = currentRightY;
    }
}

argsVector option19_rootEquationFindingGPU(argsVector argsIn) {
    argsVector argsOut;

    float startX = getValueFromArgs<float>("--startX", -10.0, argsIn);
    float endX = getValueFromArgs<float>("--endX", 10.0, argsIn);

    int nBytes = 2;
    float* result = new float[nBytes];
    float* devResult;

    cudaMalloc((void**)&devResult, nBytes * sizeof(float));

    option19_rootEquationFindingKernel<<<1024, 1024>>>(time(NULL), devResult, startX, endX);

    cudaMemcpy(result, devResult, nBytes * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(devResult);

    printf("x = %.7f\n", result[0]);
    printf("y = %.7f\n", result[1]);
    checkRootEquationFinding(result[0]);
    
    argsOut.push_back("--root " + std::to_string(result[0]));
    return argsOut;
}