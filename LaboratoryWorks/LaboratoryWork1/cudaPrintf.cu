#include <stdio.h>
#include <time.h>

#include <cstdlib>
#include <tuple>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../_common/deviceQuery.h"

using namespace std;

__device__ int getGlobalIdx_1D_1D() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_3D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) 
        + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__global__ void kernel(int* vec)
{
    int idx = getGlobalIdx_3D_3D();
    printf("Thread ID: %d \t Value: %d \n", idx, vec[idx]);
}

std::tuple<int, dim3, dim3, int> paramsInit()
{
    unsigned int nItems = 0;
    unsigned int gridX = 0, gridY = 0, gridZ = 0;
    unsigned int blockX = 0, blockY = 0, blockZ = 0;

    printf("Enter the maximum number: ");
    scanf("%d", &nItems);
    printf("MAximum number: %d\n", nItems);

    printf("Enter the Grid Dimension (x y z): ");
    scanf("%d%d%d: ", &gridX, &gridY, &gridZ);
    printf("Grid Dimension (x y z): (%d, %d, %d)\n", gridX, gridY, gridZ);

    printf("Enter the Block Dimension (x y z): ");
    scanf("%d%d%d: ", &blockX, &blockY, &blockZ);
    printf("Block Dimension (x y z): (%d, %d, %d)\n", blockX, blockY, blockZ);

    return std::tuple<int, dim3, dim3, int>(nItems,
                                            dim3(gridX, gridY, gridZ), 
                                            dim3(blockX, blockY, blockZ),
                                            0);
}

int* setNumbersToVec(const int nItems)
{
    int* vec = new int[nItems];
    for (int i = 0; i < nItems; i++)
    {
        vec[i] = i + 1;
    }

    return vec;
}

void vecPrintCPU(const int nItems, int nIterations, int preset)
{
    printf("\n====================   CPU BEGIN   ====================\n");
    int start = 0, time = 0;
    float curTimeCPU = 0.0f, timeCPU = 0.0f;
    int* vec = setNumbersToVec(nItems);

    printf("%d", nItems);

    for (int i = 0; i < nIterations; i++)
    {
        start = clock();

        for (int i = 0; i < nItems; i++)
        {
            printf("Index[%d]:\t\tValue is:%d\n", i, vec[i]);
        }

        time = clock() - start;
        curTimeCPU = time / 1.0;

        printf("CPU compute time: %.3f milliseconds\n", curTimeCPU);
        printf("\n=====================   CPU END   =====================\n");
        timeCPU += curTimeCPU;
    }
    timeCPU /= nIterations;

    printf("\n=========================   CPU AVG   ======================\n");
    printf("  Preset: %d\n", preset);
    printf("  Iterations: %d\n", nIterations);
    printf("  Numbers: %d\n", nItems);
    printf("  AVG Time: %.3f milliseconds", timeCPU);
    printf("\n=========================   CPU AVG   ======================\n");
}

void vecPrintGPU(const int nItems, dim3 dimGrid, dim3 dimBlock, int nIterations, int preset)
{
    printf("\n====================   GPU   ====================\n");
    cudaEvent_t start, stop;
    float curTimeGPU = 0.0f, timeGPU = 0.0f;
    int* vec = setNumbersToVec(nItems);

    for (int i = 0; i < nIterations; i++)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Выбор первого GPU для работы
        cudaSetDevice(0);

        // Выделение памяти GPU
        int* vecDev = NULL;
        int vecDevSize = nItems;
        cudaMalloc((void**)&vecDev, vecDevSize * sizeof(int));

        // Копирование данных из памяти CPU в память GPU
        cudaMemcpy(vecDev, vec, vecDevSize * sizeof(int), cudaMemcpyHostToDevice);

        // Вызов ядра
        kernel<<<dimGrid, dimBlock>>>(vecDev);

        // Ожидание синхринизации потоков
        cudaThreadSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&curTimeGPU, start, stop);

        printf("DEVICE GPU compute time: %.3f milliseconds \n\n", curTimeGPU);

        cudaFree(vecDev);
        timeGPU += curTimeGPU;
        curTimeGPU = 0.0f;
    }

    timeGPU /= nIterations;

    printf("\n=========================   GPU AVG   ======================\n");
    printf("  Preset: %d\n", preset);
    printf("  Iterations: %d\n", nIterations);
    printf("  Numbers: %d\n", nItems);
    printf("  AVG Time: %.3f milliseconds", timeGPU);
    printf("\n=========================   GPU AVG   ======================\n");
}

std::tuple<int, dim3, dim3, int> setParamsGPU()
{
    std::tuple<int, dim3, dim3, int> params;
    unsigned int cmd;

    #pragma region Dialogue with the user
    printf("The task of displaying N numbers on the screen through GPU.\n");
    printf("Choose the preset settings for GPU or set them yourself : \n");
    printf("  0: Set custom parameters\n");

    printf("  1: N = 2^2 \t Grid Dimension: (1, 1, 1) \t Block Dimension: (2, 2, 1)\n");     // N = 4        Blocks = 1      Threads of Block = 4
    printf("  2: N = 2^2 \t Grid Dimension: (2, 1, 1) \t Block Dimension: (2, 1, 1)\n");     // N = 4        Blocks = 2      Threads of Block = 2
    printf("  3: N = 2^2 \t Grid Dimension: (2, 2, 1) \t Block Dimension: (1, 1, 1)\n");     // N = 4        Blocks = 4      Threads of Block = 1

    printf("  4: N = 2^4 \t Grid Dimension: (1, 1, 1) \t Block Dimension: (4, 2, 2)\n");     // N = 16       Blocks = 1      Threads of Block = 16
    printf("  5: N = 2^4 \t Grid Dimension: (2, 2, 1) \t Block Dimension: (2, 2, 1)\n");     // N = 16       Blocks = 4      Threads of Block = 4
    printf("  6: N = 2^4 \t Grid Dimension: (4, 2, 2) \t Block Dimension: (1, 1, 1)\n");     // N = 16       Blocks = 16     Threads of Block = 1

    printf("  7: N = 2^6 \t Grid Dimension: (1, 1, 1) \t Block Dimension: (4, 4, 4)\n");     // N = 64       Blocks = 1      Threads of Block = 64
    printf("  8: N = 2^6 \t Grid Dimension: (2, 2, 2) \t Block Dimension: (2, 2, 2)\n");     // N = 64       Blocks = 8      Threads of Block = 8
    printf("  9: N = 2^6 \t Grid Dimension: (4, 4, 4) \t Block Dimension: (1, 1, 1)\n");     // N = 64       Blocks = 64     Threads of Block = 1

    printf("  10: N = 2^8 \t Grid Dimension: (1, 1, 1) \t Block Dimension: (8, 8, 4)\n");     // N = 256      Blocks = 1      Threads of Block = 256
    printf("  11: N = 2^8 \t Grid Dimension: (4, 2, 2) \t Block Dimension: (4, 2, 2)\n");     // N = 256      Blocks = 16     Threads of Block = 16
    printf("  12: N = 2^8 \t Grid Dimension: (8, 8, 4) \t Block Dimension: (1, 1, 1)\n");     // N = 256      Blocks = 256    Threads of Block = 1

    printf("  13: N = 2^10 \t Grid Dimension: (1, 1, 1) \t Block Dimension: (16, 8, 8)\n");       // N = 1024     Blocks = 1      Threads of Block = 1024
    printf("  14: N = 2^10 \t Grid Dimension: (4, 4, 2) \t Block Dimension: (4, 4, 2)\n");        // N = 1024     Blocks = 32     Threads of Block = 32
    printf("  15: N = 2^10 \t Grid Dimension: (16, 8, 8) \t Block Dimension: (1, 1, 1)\n");       // N = 1024     Blocks = 1024   Threads of Block = 1

    printf("  16: N = 2^12 \t Grid Dimension: (2, 2, 1) \t Block Dimension: (16, 8, 8)\n");     // N = 4096     Blocks = 1      Threads of Block = 4096
    printf("  17: N = 2^12 \t Grid Dimension: (4, 4, 4) \t Block Dimension: (4, 4, 4)\n");        // N = 4096     Blocks = 64     Threads of Block = 64
    printf("  18: N = 2^12 \t Grid Dimension: (16, 16, 16) \t Block Dimension: (1, 1, 1)\n");     // N = 4096     Blocks = 4096   Threads of Block = 1

    printf("  19: N = 2^14 \t Grid Dimension: (4, 2, 2) \t Block Dimension: (16, 8, 8)\n");     // N = 16384    Blocks = 1      Threads of Block = 16384
    printf("  20: N = 2^14 \t Grid Dimension: (8, 4, 4) \t Block Dimension: (4, 4, 2)\n");        // N = 16384    Blocks = 128    Threads of Block = 128
    printf("  21: N = 2^14 \t Grid Dimension: (32, 32, 16) \t Block Dimension: (1, 1, 1)\n");     // N = 16384    Blocks = 16384  Threads of Block = 1

    printf("  22: N = 2^16 \t Grid Dimension: (4, 4, 4) \t Block Dimension: (16, 8, 8)\n");     // N = 65536    Blocks = 1      Threads of Block = 65536
    printf("  23: N = 2^16 \t Grid Dimension: (8, 8, 4) \t Block Dimension: (8, 8, 4)\n");        // N = 65536    Blocks = 256    Threads of Block = 256
    printf("  24: N = 2^16 \t Grid Dimension: (64, 32, 32) \t Block Dimension: (1, 1, 1)\n");     // N = 65536    Blocks = 65536  Threads of Block = 1

    printf("Enter command: ");
    scanf("%d", &cmd);
    #pragma endregion

    switch (cmd)
    {
    case 1:
        params = std::tuple<int, dim3, dim3, int>(4, dim3(1, 1, 1), dim3(2, 2, 1), 1);
        break;
    case 2:
        params = std::tuple<int, dim3, dim3, int>(4, dim3(2, 1, 1), dim3(2, 1, 1), 2);
        break;
    case 3:
        params = std::tuple<int, dim3, dim3, int>(4, dim3(2, 2, 1), dim3(1, 1, 1), 3);
        break;
    case 4:
        params = std::tuple<int, dim3, dim3, int>(16, dim3(1, 1, 1), dim3(4, 2, 2), 4);
        break;
    case 5:
        params = std::tuple<int, dim3, dim3, int>(16, dim3(2, 2, 1), dim3(2, 2, 1), 5);
        break;
    case 6:
        params = std::tuple<int, dim3, dim3, int>(16, dim3(4, 2, 2), dim3(1, 1, 1), 6);
        break;
    case 7:
        params = std::tuple<int, dim3, dim3, int>(64, dim3(1, 1, 1), dim3(4, 4, 4), 7);
        break;
    case 8:
        params = std::tuple<int, dim3, dim3, int>(64, dim3(2, 2, 2), dim3(2, 2, 2), 8);
        break;
    case 9:
        params = std::tuple<int, dim3, dim3, int>(64, dim3(4, 4, 4), dim3(1, 1, 1), 9);
        break;
    case 10:
        params = std::tuple<int, dim3, dim3, int>(256, dim3(1, 1, 1), dim3(8, 8, 4), 10);
        break;
    case 11:
        params = std::tuple<int, dim3, dim3, int>(256, dim3(4, 2, 2), dim3(4, 2, 2), 11);
        break;
    case 12:
        params = std::tuple<int, dim3, dim3, int>(256, dim3(8, 8, 4), dim3(1, 1, 1), 12);
        break;
    case 13:
        params = std::tuple<int, dim3, dim3, int>(1024, dim3(1, 1, 1), dim3(16, 8, 8), 13);
        break;
    case 14:
        params = std::tuple<int, dim3, dim3, int>(1024, dim3(4, 4, 2), dim3(4, 4, 2), 14);
        break;
    case 15:
        params = std::tuple<int, dim3, dim3, int>(1024, dim3(16, 8, 8), dim3(1, 1, 1), 15);
        break;
    case 16:
        params = std::tuple<int, dim3, dim3, int>(4096, dim3(2, 2, 1), dim3(16, 8, 8), 16);
        break;
    case 17:
        params = std::tuple<int, dim3, dim3, int>(4096, dim3(4, 4, 4), dim3(4, 4, 4), 17);
        break;
    case 18:
        params = std::tuple<int, dim3, dim3, int>(4096, dim3(16, 16, 16), dim3(1, 1, 1), 18);
        break;
    case 19:
        params = std::tuple<int, dim3, dim3, int>(16384, dim3(4, 2, 2), dim3(16, 8, 8), 19); ///////
        break;
    case 20:
        params = std::tuple<int, dim3, dim3, int>(16384, dim3(8, 4, 4), dim3(8, 4, 4), 20);
        break;
    case 21:
        params = std::tuple<int, dim3, dim3, int>(16384, dim3(32, 32, 16), dim3(1, 1, 1), 21);
        break;
    case 22:
        params = std::tuple<int, dim3, dim3, int>(65536, dim3(4, 4, 4), dim3(16, 8, 8), 22); ////
        break;
    case 23:
        params = std::tuple<int, dim3, dim3, int>(65536, dim3(8, 8, 4), dim3(8, 8, 4), 23);
        break;
    case 24:
        params = std::tuple<int, dim3, dim3, int>(65536, dim3(64, 32, 32), dim3(1, 1, 1), 24);
        break;
    case 0:
    default:
        params = paramsInit();
        break;
    }

    return params;
}

std::tuple<int, int> setParamsCPU()
{
    std::tuple<int, int> params;
    int n, nItems;

    printf("The task of displaying N numbers on the screen through CPU.\n");
    printf("Specify the number of numbers to be printed or select one of the presets:\n");
    printf(" 1: N = 4\n");
    printf(" 2: N = 16\n");
    printf(" 3: N = 64\n");
    printf(" 4: N = 256\n");
    printf(" 5: N = 1024\n");
    printf(" 6: N = 4096\n");
    printf(" 7: N = 16384\n");
    printf(" 8: N = 65536\n");
    printf("Enter command: ");
    scanf("%d", &n);

    switch (n)
    {
        case 1: 
            params = std::tuple<int, int>(4, 1);
            break;
        case 2:
            params = std::tuple<int, int>(16, 2);
            break;
        case 3:
            params = std::tuple<int, int>(64, 3);
            break;
        case 4:
            params = std::tuple<int, int>(256, 4);
            break;
        case 5:
            params = std::tuple<int, int>(1024, 5);
            break;
        case 6:
            params = std::tuple<int, int>(4096, 6);
            break;
        case 7:
            params = std::tuple<int, int>(16384, 7);
            break;
        case 8:
            params = std::tuple<int, int>(65536, 8);
            break;
        default:
            params = std::tuple<int, int>(1024, 0);
            break;
    }

    return params;
}

extern "C" void cudaPrintNumbers()
{
    unsigned int dev = -1;
    unsigned int nIterations = 1;
    unsigned int nItems = 0;
    std::tuple<int, dim3, dim3, int> paramsGPU;
    std::tuple<int, int> paramsCPU;


    printf("The task of displaying N numbers on the screen. Select a device to complete the task:\n");
    printf("  1: CPU\n");
    printf("  2: GPU\n");
    printf("  3: View GPU Device Characteristics\n");

    printf("Enter command: ");
    scanf("%d", &dev);

    printf("Specify the number of iterations to get the average run time:");
    scanf("%d", &nIterations);



    switch (dev)
    {
        case 0:
            break;
        case 3:
            getDeviceInfo();
            break;
        case 2:
            paramsGPU = setParamsGPU();
            nItems = std::get<0>(paramsGPU);
            vecPrintGPU(nItems, std::get<1>(paramsGPU), std::get<2>(paramsGPU), nIterations, std::get<3>(paramsGPU));
            break;
        case 1:
            paramsCPU = setParamsCPU();
            nItems = std::get<0>(paramsCPU);
            vecPrintCPU(nItems, nIterations, std::get<1>(paramsCPU));
            break;
        default:
            nItems = std::get<0>(paramsCPU);
            paramsCPU = setParamsCPU();
            vecPrintCPU(nItems, nIterations, std::get<1>(paramsCPU));
            break;
    }
}