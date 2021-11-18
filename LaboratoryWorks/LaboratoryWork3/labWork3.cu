#include <curand_kernel.h>

#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

#include "labWork3.h"

#define N 4096


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
    
    argsOut.push_back("--root " + std::to_string(result[0]));
    return argsOut;
}


struct getFuncValues
{
    float leftX, deltaX;

    __host__ __device__ getFuncValues(float _leftX = 0.f, float _deltaX = 0.01f)
        : leftX(_leftX), deltaX(_deltaX) {};

    __host__ __device__ float2 operator()(const unsigned int n) const
    {
        float2 xy;

        xy.x = leftX + deltaX * n;
        xy.y = getEquationValue_Option19(xy.x);

        return xy;
    }
};

struct isRoot
{
    const float eps;
    isRoot(float _eps) : eps(_eps) {}

    __device__ bool operator()(const float2& xy) {
        if (eps >= fabs(xy.y)) {
            printf("x: %.7f \t %.7f \n", xy.x, xy.y);
            return true;
        }

        return false;
    }
};

argsVector option19_rootEquationFindingGPU_Thrust(argsVector argsIn) {
    argsVector argsOut;

    float leftX = getValueFromArgs<float>("--startX", -10.0, argsIn);
    float rightX = getValueFromArgs<float>("--endX", 10.0, argsIn);

    float deltaX = fabs(rightX - leftX) / (N * N);

    thrust::host_vector<float2> numbersXH(N * N);

    thrust::device_vector<float2> numbersXD = numbersXH;
    thrust::device_vector<float2> numbersYD(N * N);

    thrust::counting_iterator<unsigned int> index_sequence_begin(0);

    thrust::transform(index_sequence_begin, index_sequence_begin + N * N, numbersXD.begin(), getFuncValues(leftX, deltaX));
    thrust::copy_if(numbersXD.begin(), numbersXD.end(), numbersYD.begin(), isRoot(EPS));
    numbersXH = numbersYD;
    return argsOut;
}