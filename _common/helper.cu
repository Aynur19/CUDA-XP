#include "helper.cuh"
#include "helper.h"

#pragma region Get ThreadID
// 1D grid of 1D blocks
__device__ int getGlobalIdx_1D_1D() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// 1D grid of 2D blocks
__device__ int getGlobalIdx_1D_2D() {
    return blockIdx.x * blockDim.x * blockDim.y
        + threadIdx.y * blockDim.x + threadIdx.x;
}

// 1D grid of 3D blocks
__device__ int getGlobalIdx_1D_3D() {
    return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
        + threadIdx.z * blockDim.y * blockDim.x
        + threadIdx.y * blockDim.x + threadIdx.x;
}

// 2D grid of 1D blocks
__device__ int getGlobalIdx_2D_1D() {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

// 2D grid of 2D blocks
__device__ int getGlobalIdx_2D_2D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

// 2D grid of 3D blocks
__device__ int getGlobalIdx_2D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

// 3D grid of 1D blocks
__device__ int getGlobalIdx_3D_1D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

// 3D grid of 2D blocks
__device__ int getGlobalIdx_3D_2D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

// 3D grid of 3D blocks
__device__ int getGlobalIdx_3D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}
#pragma endregion


argsVector gpuTimeMeasuring(argsVector(*gpuComputedMethod)(argsVector argsIn), unsigned int iters, argsVector argsIn) {
    argsVector argsOut;
    
    //printf("====================   GPU COMPUTING   ====================\n");
    float curTimeGPU = 0.0f, timeGPU = 0.0f;

    for (int i = 0; i < iters; i++)
    {
        curTimeGPU = 0.0f; 
        timeGPU = 0.0f;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        argsOut = (*gpuComputedMethod)(argsIn);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&curTimeGPU, start, stop);

        printf("Iteration: %d \t GPU COMPUTE TIME: %.3f milliseconds \n\n", i + 1, curTimeGPU);

        timeGPU += curTimeGPU;
        curTimeGPU = 0.0f;
    }

    printf("=======================   GPU AVG   =======================\n");
    printf("  Iterations: %d\n", iters);
    printf("  GPU ALL COMPUTE TIME: %.3f milliseconds \n", timeGPU);
    printf("  GPU AVG COMPUTE TIME: %.3f milliseconds \n", timeGPU/iters);
    printf("=======================   GPU AVG   =======================\n");
    //printf("====================   GPU COMPUTING   ====================\n");

    // cudaDeviceReset causes the driver to clean up all state. 
    // While not mandatory in normal operation, it is good practice.  
    // It is also needed to ensure correct operation when the application is being profiled. 
    // Calling cudaDeviceReset causes all profile data to be flushed before the application exits
    cudaDeviceReset();

    return argsOut;
}