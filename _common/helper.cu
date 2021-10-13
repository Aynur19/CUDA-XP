#include "helper.cuh"
#include "helper.h"

void gpuTimeMeasuring(void (*gpuComputedMethod)(int argc, char* argv[]), unsigned int iters, int argc, char* argv[]) {
    printf("====================   GPU COMPUTING   ====================\n");
    float curTimeGPU = 0.0f, timeGPU = 0.0f;

    for (int i = 0; i < iters; i++)
    {
        curTimeGPU = 0.0f; 
        timeGPU = 0.0f;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        (*gpuComputedMethod)(argc, argv);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&curTimeGPU, start, stop);

        printf("Iteration: %d \t GPU COMPUTE TIME: %.3f milliseconds \n\n", i + 1, curTimeGPU);

        timeGPU += curTimeGPU;
        curTimeGPU = 0.0f;
    }

    printf("=======================   GPU AVG   +======================\n");
    printf("  Iterations: %d\n", iters);
    printf("  GPU ALL COMPUTE TIME: %.3f milliseconds \n", timeGPU);
    printf("  GPU AVG COMPUTE TIME: %.3f milliseconds \n", timeGPU/iters);
    printf("=======================   GPU AVG   =======================\n");
    printf("====================   GPU COMPUTING   ====================\n");

    // cudaDeviceReset causes the driver to clean up all state. 
    // While not mandatory in normal operation, it is good practice.  
    // It is also needed to ensure correct operation when the application is being profiled. 
    // Calling cudaDeviceReset causes all profile data to be flushed before the application exits
    cudaDeviceReset();
}