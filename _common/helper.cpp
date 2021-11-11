#include "helper.h"

/// <summary>
/// Decorator for measuring the execution time of calculations on the CPU.
/// </summary>
/// <param name="cpuComputedMethod">CPU computation method reference.</param>
/// <param name="iters">Number of passes.</param>
/// <param name="argsIn">Parameters passed to the evaluation function</param>
/// <returns>Computation results.</returns>
argsVector cpuTimeMeasuring(argsVector(*cpuComputedMethod)(argsVector argsIn), unsigned int iters, argsVector argsIn) {
    argsVector argsOut;
    int start = 0, time = 0;
    float curTimeCPU = 0.0f, timeCPU = 0.0f;

    for (int i = 0; i < iters; i++)
    {
        start = clock();

        argsOut = (*cpuComputedMethod)(argsIn);

        time = clock() - start;
        curTimeCPU = time / 1.0;

        printf("Iteration: %d \t CPU COMPUTE TIME: %.3f milliseconds \n\n", i + 1, curTimeCPU);

        timeCPU += curTimeCPU;
        curTimeCPU = 0.0f;
    }

    printf("=======================   CPU AVG   +======================\n");
    printf("  Iterations: %d\n", iters);
    printf("  CPU ALL COMPUTE TIME: %.5f milliseconds \n", timeCPU);
    printf("  CPU AVG COMPUTE TIME: %.5f milliseconds \n", timeCPU / iters);
    printf("=======================   CPU AVG   =======================\n");

    return argsOut;
}

void matrixFillIndices(float* matrix, int nRows, int nCols) {
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            matrix[i * nCols + j] = i * nCols + j;
        }
    }
}

void matrixPrint(float* matrix, int nRows, int nCols) {
    //printf("\n==================================================\n");
    for (int i = 0; i < nRows; i++) {
        printf("[");
        for (int j = 0; j < nCols; j++) {
            printf("%.3f\t", matrix[i * nCols + j]);
        }
        printf("]\n");
    }
    //printf("\n==================================================\n");
}

void arrayRandomInit(float* randArray, int n)
{
    for (int i = 0; i < n; i++)
    {
        randArray[i] = rand() / (float)RAND_MAX;
    }
}