// 4. Вывести на экран числа от 1 до 65535
//#include <stdio.h>
//#include <cuda_runtime.h>

// System includes
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iterator>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cuda_profiler_api.h>
#include <ctime>

#include "..\_common\helper_cuda.h"
#include "..\_common\helper_string.h"
//#include "..\_common\deviceQuery.h"

using namespace std;


//__device__ int getGlobalIdx_1D_1D() {
//    return blockIdx.x * blockDim.x + threadIdx.x;
//}
//
//__device__ int getGlobalIdx_3D_3D() {
//    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
//    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
//    return threadId;
//}

//__global__ void testKernel2(int* vec, const int nAllThreads, const int iteration)
//{   
//    //int idx = getGlobalIdx_3D_3D();
//    int idx = getGlobalIdx_1D_1D();
//    printf("Iteration: %d \t Thread ID: %d \t Value: %d \n", iteration, idx, vec[idx + iteration * nAllThreads]);
//
//    //printf("[%d, %d]:\t\tValue is:%d\n", blockId, threadId, vec[blockId]
//    /*printf("[%d, %d]:\t\tValue is:%d\n", \
//        blockIdx.y * gridDim.x + blockIdx.x, \
//        threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x, \
//        vec[(blockIdx.y * gridDim.x + blockIdx.x) * nThreads + (threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)]);
//        */
//}

//void vecPrintCPU(int* vec, const int nItems)
//{
//    printf("\n====================   CPU   ====================\n");
//    int start, time;
//    start = clock();
//
//    for (int i = 0; i < nItems; i++)
//    {
//        printf("[%d]:\t\tValue is:%d\n", i, vec[i]);
//    }
//
//    time = clock() - start;
//    float timeCPU = time / 2.0;
//
//    printf("CPU compute time: %f milliseconds\n\n", timeCPU);
//}

//void vecPrintGPU(int* vec, const int nItems, const int nBlocks, const int nThreads, dim3 dimGrid, dim3 dimBlock)
//{
//    printf("\n====================   GPU   ====================\n");
//
//    cudaEvent_t start, stop;
//    float timeGPU = 0.0f;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start, 0);
//
//    // Выбор первого GPU для работы
//    cudaSetDevice(0);
//
//    // Выделение памяти GPU
//    int* vecDev = NULL;
//    int vecDevSize = nItems;
//    cudaMalloc((void**)&vecDev, vecDevSize * sizeof(int));
//
//    // Копирование данных из памяти CPU в память GPU
//    cudaMemcpy(vecDev, vec, vecDevSize * sizeof(int), cudaMemcpyHostToDevice);
//
//    int count = nItems / (nBlocks * nThreads);
//    for (int i = 0; i < count; i++)
//    {
//        testKernel2<<<dimGrid, dimBlock>>>(vecDev, nBlocks * nThreads, i);
//        cudaThreadSynchronize();
//    }
//    // Вызов ядра
//    //testKernel2<<<dimGrid, dimBlock>>>(nThreads, vecDev);
//
//    // Ожидание синхринизации потоков
//    //cudaThreadSynchronize();
//
//
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&timeGPU, start, stop);
//
//    printf("DEVICE GPU compute time: %.3f milliseconds \n\n", timeGPU);
//
//    cudaFree(vecDev);
//}

//const unsigned int nItems = 1<<16;
    //const unsigned int nBlocks = 256;
    //const unsigned int nThreads = 256;

    ////dim3 dimGrid(64, 1, 1);
    //dim3 dimGrid(256, 1, 1);
    //dim3 dimBlock(256, 1, 1);

    //int* vec = new int[nItems];
    //for (int i = 0; i < nItems; i++)
    //{
    //    vec[i] = i + 1;
    //}

    //vecPrintGPU(vec, nItems, nBlocks, nThreads, dimGrid, dimBlock);
    //vecPrintCPU(vec, nItems);
    

    //delete[] vec;


    //vecPrintGPU(vec, nBlocks, nThreads, dimGrid, dimBlock);


    //int devID;
    //cudaDeviceProp props;

    //// This will pick the best possible CUDA capable device
    //devID = findCudaDevice(argc, (const char**)argv);

    ////Get GPU information
    //checkCudaErrors(cudaGetDevice(&devID));
    //checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    //printf("Device %d: \"%s\" with Compute %d.%d capability\n",
    //    devID, props.name, props.major, props.minor);

    //printf("printf() is called. Output:\n\n");

    ////Kernel configuration, where a two-dimensional grid and
    ////three-dimensional blocks are configured.
    //dim3 dimGrid(2, 2);
    //dim3 dimBlock(2, 2, 2);
    //testKernel2<<<dimGrid, dimBlock>>>(numbers);
    //cudaDeviceSynchronize();
