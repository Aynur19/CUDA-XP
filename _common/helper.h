#ifndef __HELPER_H__
#define __HLEPER_H__

#include <cstdio>
#include <ctime>

#define MATRIX_INDEX(row, col, columnsInRow) ((col) + (row) * columnsInRow)

void (*gpuComputedMethod)(unsigned int verbose) = NULL;

void (*cpuComputedMethod)(unsigned int verbose) = NULL;

void gpuTimeMeasuring(void (*gpuComputedMethod)(unsigned int verbose), unsigned int iters, unsigned int verbose = 0) {
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

        (*gpuComputedMethod)(verbose);

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
}

void cpuTimeMeasuring(void (*cpuComputedMethod)(unsigned int verbose), unsigned int iters, unsigned int verbose = 0) {
    printf("====================   CPU COMPUTING   ====================\n");
    int start = 0, time = 0;
    float curTimeCPU = 0.0f, timeCPU = 0.0f;

    for (int i = 0; i < iters; i++)
    {
        start = clock();

        (*cpuComputedMethod)(verbose);

        time = clock() - start;
        curTimeCPU = time / 1.0;

        printf("Iteration: %d \t CPU COMPUTE TIME: %.3f milliseconds \n\n", i + 1, curTimeCPU);

        timeCPU += curTimeCPU;
        curTimeCPU = 0.0f;
    }

    printf("=======================   CPU AVG   +======================\n");
    printf("  Iterations: %d\n", iters);
    printf("  CPU ALL COMPUTE TIME: %.3f milliseconds \n", timeCPU);
    printf("  CPU AVG COMPUTE TIME: %.3f milliseconds \n", timeCPU / iters);
    printf("=======================   CPU AVG   =======================\n");
    printf("====================   CPU COMPUTING   ====================\n");
}

void matrixFillIndices(float *matrix, int nRows, int nCols) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			matrix[i * nCols + j] = i * nCols + j;
		}
	}
}


void matrixPrint(float *matrix, int nRows, int nCols) {
	printf("\n==================================================\n");
	for (int i = 0; i < nRows; i++) {
		printf("[");
		for (int j = 0; j < nCols; j++) {
			printf("%.3f\t", matrix[i * nCols + j]);
		}
		printf("]\n");
	}
	printf("\n==================================================\n");
}

#endif // !__HELPER_H__
