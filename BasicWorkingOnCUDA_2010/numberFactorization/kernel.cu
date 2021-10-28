#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

using namespace std;

typedef unsigned long long uint64;
#define BLOCK_SIZE 256

__global__ void cudaFermaKernel(uint64* factors, uint64 n, uint64 maxFactor, uint64 k, uint64 index) {

    uint64 threadId = blockDim.x * blockIdx.x + threadIdx.x;

    k = k + threadId;
    uint64 x = 0;
    double l = 0, y = 0, a = 0, b = 0;

    x = maxFactor + k;
    l = x * x - n;
    y = sqrt(l);

    if ((uint64)y * (uint64)y == l) {
        factors[index] = x - y;
        factors[index + 1] = x + y;
    }
}

__global__ void cudaFermaSharedKernel(uint64* factors, uint64 n, uint64 maxFactor, uint64 k, uint64 index) {

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double sharedArray[BLOCK_SIZE][2];

    uint64 x = maxFactor + k + threadId;
    sharedArray[threadIdx.x][0] = x * x - n;                            // l
    sharedArray[threadIdx.x][1] = sqrt(sharedArray[threadIdx.x][0]);    // y

    if ((uint64)sharedArray[threadIdx.x][1] * (uint64)sharedArray[threadIdx.x][1] == (uint64)sharedArray[threadIdx.x][0]) {
        factors[index] = x - sharedArray[threadIdx.x][1];
        factors[index + 1] = x + sharedArray[threadIdx.x][1];
    }
}

bool checkResult(vector<uint64> factors, uint64 n) {
    uint64 result = 0;

    if (factors.size() == 0) {
        return result == n;
    }

    result = factors[0];
    if (factors.size() == 1) {
        return result == n;
    }

    printf("%lld ", factors[0]);
    for (int i = 1; i < factors.size(); i++) {
        result *= factors[i];
        printf("* %lld ", factors[i]);
    }

    printf("= %lld => %s", result, (result == n) ? "true" : "false");

    return result == n;
}

vector<uint64> getFactors2(uint64& n) {
    vector<uint64> factors2;

    while (n % 2 == 0) {
        factors2.push_back(2);
        n /= 2;
    }

    return factors2;
}

uint64 getMaxFactor(uint64 n) {
    uint64 maxFactor = sqrt(n);

    if (maxFactor * maxFactor < n) {
        maxFactor++;
    }

    return maxFactor;
}

vector<uint64> getFactorsAll(uint64 n) {
    printf("\nCPU BEGIN (Method of Enumerating Divisors)\n");
    int start = 0, time = 0;
    float curTimeCPU = 0.0f, timeCPU = 0.0f;

    vector<uint64> factorsAll;
    uint64 currentN = n;
    uint64 i = 2;

    while (getMaxFactor(currentN) >= i) {
        while (currentN % i == 0) {
            factorsAll.push_back(i);
            currentN /= i;
        }
        i++;
    }

    if (currentN != 1) {
        factorsAll.push_back(currentN);
    }

    time = clock() - start;
    curTimeCPU = time / 1.0;

    printf("CPU compute time (Method of Enumerating Divisors): %.3f milliseconds\n", curTimeCPU);
    printf("Factors of number %lld: ", n);

    printf("%lld", factorsAll[0]);
    for (uint64 i = 1; i < factorsAll.size(); i++) {
        printf(", %lld", factorsAll[i]);
    }
    printf("\n");

    checkResult(factorsAll, n);

    printf("\nCPU END (Method of Enumerating Divisors)\n");
    return factorsAll;
}

void ferma(vector<uint64>& factorsAll, uint64 n) {
    uint64 currentN = n;

    if (n == 1) {
        return;
    }

    uint64 maxFactor = getMaxFactor(currentN);
    uint64 k = 0, x = 0, l = 0, a = 0, b = 0;
    double y = 0;

    do {
        x = maxFactor + k;
        l = x * x - currentN;
        y = sqrt(l);
        k++;

    } while ((uint64)y * (uint64)y != l);

    a = x - (uint64)y;
    b = x + (uint64)y;

    //printf("MaxFactor: %lld\t ThreadID-k: %lld\tx: %lld\tl: %.3e\ty: %.3e\ta: %lld\tb: %lld\n",
    //    maxFactor, k, x, l, y, a, b);

    if (a == 1) {
        factorsAll.push_back(b);
    }
    else if (b == 1) {
        factorsAll.push_back(a);
    }
    else
    {
        ferma(factorsAll, a);
        ferma(factorsAll, b);
    }
}

void factorsFermaCPU(uint64 n) {
    printf("\nCPU BEGIN: (Ferma method)\n");
    int start = 0, time = 0;
    float curTimeCPU = 0.0f, timeCPU = 0.0f;

    uint64 currentN = n;
    vector<uint64> factorsAll = getFactors2(currentN);
    ferma(factorsAll, currentN);

    time = clock() - start;
    curTimeCPU = time / 1.0;

    printf("CPU compute time (Ferma method): %.3f milliseconds\n", curTimeCPU);
    printf("Factors of number %lld: ", n);

    printf("%lld", factorsAll[0]);
    for (uint64 i = 1; i < factorsAll.size(); i++) {
        printf(", %lld", factorsAll[i]);
    }

    printf("\n");
    checkResult(factorsAll, n);
    printf("\nCPU END (Ferma method)\n");
}

void factorsFermaGPU(uint64 n, int kernel = 0) {
    printf("\nGPU (Ferma method):\n");
    cudaEvent_t start, stop;
    float curTimeGPU = 0.0f, timeGPU = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Выбор первого GPU для работы
    cudaSetDevice(0);

    uint64 mult = 1;
    uint64 currentN = n;
    vector<uint64> factorsAll = getFactors2(currentN);
    for (uint64 i = 0; i < factorsAll.size(); i++) {
        mult *= factorsAll[i];
    }

    uint64 maxFactor = getMaxFactor(currentN);
    uint64 factorsSize = 20;
    uint64* factors = new uint64[factorsSize];
    uint64* factorsDev;

    cudaMalloc((void**)&factorsDev, factorsSize * sizeof(uint64));

    dim3 blocks = 3;
    dim3 threads = BLOCK_SIZE;
    uint64 threadsCount = 3 * BLOCK_SIZE;

    //printf("Grid Dim: %d %d %d  =>  Blocks: %d\n", blocks.x, blocks.y, blocks.z, blocks.x * blocks.y * blocks.z);
    //printf("Block Dim: %d %d %d  =>  Threads: %d\n", threads.x, threads.y, threads.z, threads.x * threads.y * threads.z);

    uint64 k = 0;
    uint64 index = 0;
    uint64 indexPush = 0;
    bool isEnd = false;
    uint64 temp1 = 0, temp2 = 0;
    while (!isEnd) {
        if (kernel == 0) {
            cudaFermaKernel<<<blocks, threads>>>(factorsDev, currentN, getMaxFactor(currentN), k, indexPush);
        }
        else {
            cudaFermaSharedKernel<<<blocks, threads>>>(factorsDev, currentN, getMaxFactor(currentN), k, indexPush);
        }

        cudaMemcpy(factors, factorsDev, factorsSize * sizeof(uint64), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();

        if (factors[index] == 0) {
            k += threadsCount;
        }
        else if (factors[index] != 1 && factors[index + 1] != 1) {
            k = 0;
            currentN = factors[index];
            temp1 = factors[index];
            temp2 = factors[index + 1];
        }
        else if (factors[index] == 1 && factors[index + 1] == 1) {
            isEnd = true;
        }
        else {
            uint64 temp = 0;
            if (temp1 == factors[index] && temp2 == factors[index + 1]) {
                if (factors[index] == 1) {
                    temp = factors[index + 1];
                }
                else {
                    temp = factors[index];
                }

                factorsAll.push_back(temp);
                mult *= temp;

                currentN = (uint64)n / mult;
                indexPush = index + 2;
                index = indexPush;
            }
            else {
                if (factors[index] == 1) {
                    currentN = factors[index + 1];
                }
                else {
                    currentN = factors[index];
                }

                k = 0;
                temp1 = factors[index];
                temp2 = factors[index + 1];
            }

            if (mult == n) {
                isEnd = true;
            }
        }
    }

    cudaFree(factorsDev);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&curTimeGPU, start, stop);

    printf("DEVICE GPU compute time (Ferma method): %.3f milliseconds \n", curTimeGPU);
    printf("%lld", factorsAll[0]);
    for (uint64 i = 1; i < factorsAll.size(); i++) {
        printf(", %lld", factorsAll[i]);
    }

    printf("\n");
    checkResult(factorsAll, n);
    printf("\nGPU END (Ferma method)\n");
}

int main()
{
    uint64 n = 100000050500;

    getFactorsAll(n);
    factorsFermaGPU(n, 0); //Global Memory
    factorsFermaGPU(n, 1); //Shared Memory

    return 0;
}









//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdlib.h>
//#include <time.h>
//
//#include <cstdlib>
//#include <ctime>
//
//#include <curand.h>
//
//#include <stdio.h>
//#include <string>
//#include "../../_common/helper_cuda.h"
//
//using namespace std;
//
//typedef unsigned long long uint64;
//
//__host__ __device__ uint64 gcd(uint64 u, uint64 v) {
//    uint64 shift;
//    if (u == 0) return v;
//    if (v == 0) return u;
//
//    // пока u и v не станут равны 0
//    for (shift = 0; ((u | v) & 1) == 0; ++shift) {
//        // уменьшаем в 2 раза u и v
//        u >>= 1;
//        v >>= 1;
//    }   
//
//    while ((u & 1) == 0) {
//        u >>= 1;
//    }
//
//    do {
//        while ((v & 1) == 0)
//            v >>= 1;
//
//        if (u > v) {
//            uint64 t = v; v = u; u = t;
//        }
//        v = v - u;
//    } while (v != 0);
//
//    return u << shift;
//}
//
//// проверка того, что число простое
//__host__ __device__ bool isPrime(uint64 n) {
//    for (uint64 i = 2; i <= sqrt(n); i++) {
//        if (n % i == 0) {
//            return false;
//        }
//    }
//
//    return true;
//}
//
//__global__ void clearPara(uint64* devA, uint64* devC, uint64 maxFactor) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    //devA[idx] = devA[idx] % (maxFactor - 1) + 1;
//    //devC[idx] = devC[idx] % (maxFactor - 1) + 1;
//    uint64 devA_idx = devA[idx] % (maxFactor - 1) + 1;
//    uint64 devC_idx = devC[idx] % (maxFactor - 1) + 1;
//
//    //printf("maxFactor: %lld\n devA[%lld]: %lld\t -> devA[%lld]: %lld\ndevC[%lld]: %lld\t -> devC[%lld]: %lld\n",
//    //    maxFactor, idx, devA[idx], idx, devA_idx, idx, devC[idx], idx, devC_idx);
//}
//
//__global__ void pollardKernel(uint64 num, uint64* resultd, uint64* dx, uint64* dy, uint64* da, uint64* dc) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    uint64 n = num;
//    uint64 x, y, a, c;
//    x = dx[idx];
//    y = dy[idx];
//    a = da[idx];
//    c = dc[idx];
//
//    x = (a * x * x + c) % n;
//    y = a * y * y + c;
//    y = (a * y * y + c) % n;
//
//    uint64 z = x > y ? (x - y) : (y - x);
//    uint64 d = gcd(z, n);
//
//    dx[idx] = x;
//    dy[idx] = y;
//
//    if (d != 1 && d != n) *resultd = d;
//
//    printf("z: %lld\tn: %lld\td: %lld\n", z, n, d);
//}
//
//#pragma comment(lib, "curand.lib")
//uint64 pollard(uint64 num)
//{
//    // находим маскимальный делитель
//    uint64 maxFactor = sqrt(num); 
//    uint64 result = 0;
//    int nThreads = 256, nBlocks = 256;
//    
//    if (num % 2 == 0) return 2;
//    if (num % 3 == 0) return 3;
//    if (num % 5 == 0) return 5;
//    if (num % 7 == 0) return 7;
//    if (maxFactor * maxFactor == num) return maxFactor;
//    if (isPrime(num)) return num;
//
//    // инициализация указателей на массивы для GPU
//    uint64* devResult = NULL;
//    uint64* devX = NULL;
//    uint64* dexY = NULL;
//    uint64* devA = NULL;
//    uint64* devC = NULL;
//    
//    // выделение памяти на GPU
//    cudaMalloc((void**)&devResult, sizeof(uint64));
//    cudaMemset(devResult, 0, sizeof(uint64)); // заполнение массива нулями
//    cudaMalloc((void**)&devX, nBlocks * nThreads * sizeof(uint64));
//    cudaMalloc((void**)&dexY, nBlocks * nThreads * sizeof(uint64));
//    cudaMalloc((void**)&devA, nBlocks * nThreads * sizeof(uint64));
//    cudaMalloc((void**)&devC, nBlocks * nThreads * sizeof(uint64));
//
//    // инициализация рандомайзера
//    curandGenerator_t randGen;
//    curandCreateGenerator(&randGen, CURAND_RNG_QUASI_SOBOL64);
//    curandSetPseudoRandomGeneratorSeed(randGen, time(NULL)); // установка начального числа
//    
//    // генерация случайных чисел
//    curandGenerateLongLong(randGen, devA, nBlocks * nThreads);
//    curandGenerateLongLong(randGen, devC, nBlocks * nThreads);
//
//    // заполнение массивов нулями
//    cudaMemset(devX, 0, nBlocks * nThreads * sizeof(uint64));
//    cudaMemset(dexY, 0, nBlocks * nThreads * sizeof(uint64));
//    
//    printf("%lld \n", maxFactor);
//
//    // получаем массивы с остатками от деления на maxFactor
//    clearPara<<<nBlocks, nThreads >>>(devA, devC, maxFactor);
//
//    while (result == 0) {
//        pollardKernel<<<nBlocks, nThreads>>>(num, devResult, devX, dexY, devA, devC);
//        cudaMemcpy(&result, devResult, sizeof(uint64), cudaMemcpyDeviceToHost);
//    }
//
//    // освобождение памяти
//    cudaFree(devX);
//    cudaFree(dexY);
//    cudaFree(devA);
//    cudaFree(devC);
//    cudaFree(devResult);
//    curandDestroyGenerator(randGen);
//
//    return result;
//}
//
//uint64 pollardhost(uint64 num)
//{
//    uint64 upper = sqrt(num), result = 0;
//
//    if (num % 2 == 0) return 2;
//    if (num % 3 == 0) return 3;
//    if (num % 5 == 0) return 5;
//    if (num % 7 == 0) return 7;
//
//    if (upper * upper == num) return upper;
//    if (isPrime(num)) return num;
//
//    bool quit = false;
//
//    uint64 x = 0;
//    uint64 a = rand() % (upper - 1) + 1;
//    uint64 c = rand() % (upper - 1) + 1;
//    uint64 y, d;
//
//    y = x;
//    d = 1;
//
//    do {
//        x = (a * x * x + c) % num;
//        y = a * y * y + c;
//        y = (a * y * y + c) % num;
//        uint64 z = x > y ? (x - y) : (y - x);
//        d = gcd(z, num);
//    } while (d == 1 && !quit);
//
//
//    if (d != 1 && d != num) {
//        quit = true;
//        result = d;
//    }
//
//    return result;
//}
//
//uint64 pollardhost1(uint64 num)
//{
//    int result = 0;
//    while (result == 0) {
//        result = pollardhost(num);
//    }
//    return result;
//}
//
//
//int main()
//{
//tryAgain: // ýòî ëåéáë
////getTime();
//    srand(time(NULL));
//
//    //auto elapsedTimeInMsGPU = 0.0f;
//    //float elapsedTimeInMsCPU = 0.0f;
//    //StopWatchInterface* timerCPU = NULL;
//    //StopWatchInterface* timerGPU = NULL;
//
//    uint64 n = 0;
//    printf("Input num: ");
//    scanf("%d", &n);             //çàäàåì ðàçìåð
//    uint64 num = n;
//    //
//    uint64 result;
//    uint64 prevNum;
//    string res1;
//    string res2;
//    string res3;
//    string res4;
//    string res5;
//    string res6;
//    string res7;
//    uint64 rslt;
//    string resultString;
//    const char* resultStr;
//    //
//          //SDK timer
//    //sdkCreateTimer(&timerGPU);
//    //sdkStartTimer(&timerGPU);
//    //
//    result = pollard(num);
//    prevNum = num / result;
//    res1 = "Result(GPU): ";
//    res2 = to_string(num);
//    res3 = " = ";
//    res4 = to_string(result);
//    res5 = " * ";
//    resultString = res1 + res2 + res3 + res4;
//    while (!isPrime(prevNum))
//    {
//        rslt = pollard(prevNum);
//        prevNum = prevNum / rslt;
//        res6 = to_string(rslt);
//        resultString += res5 + res6;
//    }
//    res7 = to_string(prevNum);
//    resultString += res5 + res7;
//    resultString += "\n";
//    resultStr = resultString.c_str();
//    //	
//    //sdkStopTimer(&timerGPU);
//    //elapsedTimeInMsGPU = sdkGetTimerValue(&timerGPU);
//
//    //printf("Result(GPU): %lld = %lld * %lld\n", num, result, num / result);  
//    printf(resultStr);
//    //printf("Time  : %.6fs\n", elapsedTimeInMsGPU);
//
//    //SDK timer
//    //sdkCreateTimer(&timerCPU);
//    //sdkStartTimer(&timerCPU);
//
//    result = pollardhost1(num);
//    prevNum = num / result;
//    res1 = "Result(CPU): ";
//    res2 = to_string(num);
//    res3 = " = ";
//    res4 = to_string(result);
//    res5 = " * ";
//    resultString = res1 + res2 + res3 + res4;
//    while (!isPrime(prevNum))
//    {
//        rslt = pollardhost1(prevNum);
//        prevNum = prevNum / rslt;
//        res6 = to_string(rslt);
//        resultString += res5 + res6;
//    }
//    res7 = to_string(prevNum);
//    resultString += res5 + res7;
//    resultString += "\n";
//    resultStr = resultString.c_str();
//
//    //sdkStopTimer(&timerCPU);
//    //elapsedTimeInMsCPU = sdkGetTimerValue(&timerCPU);
//
//    printf(resultStr);
//    //printf("Time  : %.6fs\n", elapsedTimeInMsCPU);
//
//    goto tryAgain; // à ýòî îïåðàòîð goto
//
//    return 0;
//}
//
//
////#include "cuda.h"
////#include "cuda_runtime.h"
////#include "device_launch_parameters.h"
////#include <stdio.h>
////#include <time.h>
////#include <iostream>
////#include <chrono> 
////using namespace std;
////using namespace std::chrono;
////
////#define BLOCK_SIZE 16
//////#define N 16384
////#define N 405
////
//////----------CPU functions----------
////
////bool isPrimeCPU(int x) {
////    bool numberIsPrime = true;
////
////    if (x == 0 || x == 1) {
////        numberIsPrime = false;
////    }
////    else {
////        for (int i = 2; i <= x / 2; i++) {
////            if (x % i == 0) {
////                numberIsPrime = false;
////                break;
////            }
////        }
////    }
////        
////
////    return numberIsPrime;
////}
////
////int powerCPU(int x, int n) {
////
////    int result = x;
////
////    for (int i = 1; i < n; i++) {
////        result *= x;
////    }
////
////    return result;
////}
////
////bool isMersenneCPU(int x) {
////
////    int p;
////
////    for (int i = 0; ; i++) {
////        p = powerCPU(2, i);
////
////        if (p > x + 1)
////            return false;
////
////        else if ((p == x + 1) && isPrimeCPU(i))
////            return true;
////    }
////}
////
//////----------GPU functions----------
////
////__device__ bool isPrime(int x) {
////
////    bool flag = true;
////
////    if (x == 0 || x == 1)
////        flag = false;
////
////    else
////        for (int i = 2; i <= x / 2; i++) {
////            if (x % i == 0) {
////                flag = false;
////                break;
////            }
////        }
////
////    return flag;
////}
////
////__device__ int power(int x, int n) {
////
////    int result = x;
////
////    for (int i = 1; i < n; i++) {
////        result *= x;
////    }
////
////    return result;
////}
////
////__device__ bool isMersenne(int x) {
////
////    int p;
////
////    for (int i = 0; ; i++) {
////        p = power(2, i);
////
////        if (p > x + 1)
////            return false;
////
////        else if ((p == x + 1) && isPrime(i))
////            return true;
////    }
////}
////
////__global__ void mersenneKernel(bool* a) {
////
////    int i = threadIdx.x + blockIdx.x * BLOCK_SIZE;
////    __shared__ bool sharedA[N];
////
////    __syncthreads();
////
////    if (isMersenne(i))
////        sharedA[i] = true;
////
////    __syncthreads();
////    a[i] = sharedA[i];
////}
////
////void showArray(bool* a, int n) {
////    for (int i = 0; i < n; i++)
////        if (a[i])
////            cout << i << " ";
////}
////
////
////int main() {
////
////    bool a[N] = { false };
////    cout << N << " elements:\n";
////
////    bool* dev_a = 0;
////    cudaError_t cudaStatus;
////    cudaStatus = cudaSetDevice(0);
////    cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(bool));
////
////    auto start = high_resolution_clock::now();
////    mersenneKernel << <N / BLOCK_SIZE, BLOCK_SIZE >> > (dev_a);
////
////    cudaStatus = cudaDeviceSynchronize();
////    cudaStatus = cudaMemcpy(a, dev_a, N * sizeof(bool), cudaMemcpyDeviceToHost);
////
////    auto end = high_resolution_clock::now();
////    auto duration = duration_cast<nanoseconds>(end - start);
////
////    cudaFree(dev_a);
////    cudaStatus = cudaDeviceReset();
////
////    cout << "(GPU) Mersenne's numbers in range [0; " << N << "]: ";
////    showArray(a, N);
////    cout << "\n(GPU) Elapsed time: " << duration.count() << " nanoseconds\n";
////
////    start = high_resolution_clock::now();
////
////    for (int i = 0; i < N; i++)
////        if (isMersenneCPU(i))
////            a[i] = true;
////
////    end = high_resolution_clock::now();
////    duration = duration_cast<nanoseconds>(end - start);
////
////    cout << "(CPU) Mersenne's numbers in range [0; " << N << "]: ";
////    showArray(a, N);
////    cout << "\n(CPU) Elapsed time: " << duration.count() << " nanoseconds\n";
////
////    return 0;
////}
////
////
//////#include <cstdio>
//////#include <cstdlib>
//////#include <vector>
//////#include <cuda_runtime.h>
//////#include <device_launch_parameters.h>
//////#include <time.h>
//////
//////using namespace std;
//////
//////typedef unsigned long long uint64;
//////#define BLOCK_SIZE 256
//////
////////uint64* getSimpleNumbers() {
////////    uint64* numbers = new uint64[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
////////        107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
////////        257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
////////        421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
////////        547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
////////        661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
////////        811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
////////        947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
////////        1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
////////        1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373,
////////        1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
////////        1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657,
////////        1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811,
////////        1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
////////        1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129,
////////        2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287,
////////        2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423,
////////        2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617,
////////        2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741,
////////        2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903,
////////        2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079,
////////        3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257,
////////        3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413,
////////        3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571];
////////}
//////
////////__global__ void cudaFermaKernel(uint64* factors, uint64 n, uint64 maxFactor) {
////////
////////    uint64 threadId = threadIdx.x;
////////
////////    printf("\n", threadId);
////////
////////    uint64 x = 0;
////////    float l = 0, y = 0, a = 0, b = 0;
////////
////////    x = maxFactor + threadId;
////////    l = x * x - n;
////////    y = sqrt(l);
////////
////////    //if ((uint64)y * (uint64)y == l) {
////////    factors[threadId] = x - y;
////////    //factors[threadId + maxFactor] = x + y;
////////
////////
////////    //x = factors[threadId] + threadId;
////////    //l = x * x - n;
////////    //y = sqrt(l);
////////
////////    ////if ((uint64)y * (uint64)y == l) {
////////    a = x - y;
////////    b = x + y;
////////    //factors[threadId + maxFactor] = a;
////////    //factors[threadId + maxFactor*3] = x + y;
////////
////////    printf("tId: %lld\tx: %lld\tl: %.2f\ty: %.2f\ta: %.2f\tb: %.2f\n",
////////        threadId, x, l, y, a, b);
////////
////////    /*x = factors[threadId + maxFactor] + threadId;
////////    l = x * x - n;
////////    y = sqrt(l);*/
////////
////////    //if ((uint64)y * (uint64)y == l) {
////////    //factors[threadId + maxFactor*2] = x - y;
////////
////////    //printf("ThreadID: %lld\t[0]:%lld\t[1]:%lld\n",
////////    //    threadId, factors[threadId], factors[threadId + maxFactor]);
////////
////////    //printf("ThreadID: %lld\t[0]:%lld\t[1]:%lld\t[2]:%lld\n", 
////////        //threadId, factors[threadId], factors[threadId + maxFactor], factors[threadId + maxFactor*2]);
////////
////////    //printf("ThreadID: %lld\t[0]:%lld\t[1]:%lld\t[2]:%lld\t[3]:%lld\n", 
////////        //threadId, factors[threadId], factors[threadId + maxFactor], factors[threadId + maxFactor*2], factors[threadId + maxFactor*3]);
////////    //}
////////}
////////
////////bool checkResult(vector<uint64> factors, uint64 n) {
////////    uint64 result = 0;
////////
////////    if (factors.size() == 0) {
////////        return result == n;
////////    }
////////
////////    result = factors[0];
////////    if (factors.size() == 1) {
////////        return result == n;
////////    }
////////
////////    printf("%lld ", factors[0]);
////////    for (int i = 1; i < factors.size(); i++) {
////////        result *= factors[i];
////////        printf("* %lld ", factors[i]);
////////    }
////////
////////    printf("= %lld => %s", result, (result == n) ? "true" : "false");
////////
////////    return result == n;
////////}
////////
////////vector<uint64> getFactors2(uint64& n) {
////////    vector<uint64> factors2;
////////
////////    while (n % 2 == 0) {
////////        factors2.push_back(2);
////////        n /= 2;
////////    }
////////
////////    return factors2;
////////}
////////
////////uint64 getMaxFactor(uint64 n) {
////////    uint64 maxFactor = sqrt(n);
////////
////////    if (maxFactor * maxFactor < n) {
////////        maxFactor++;
////////    }
////////
////////    return maxFactor;
////////}
////////
////////vector<uint64> getFactorsAll(uint64 n) {
////////    printf("\nCPU BEGIN (Method of Enumerating Divisors)\n");
////////    int start = 0, time = 0;
////////    float curTimeCPU = 0.0f, timeCPU = 0.0f;
////////
////////    vector<uint64> factorsAll;
////////    uint64 currentN = n;
////////    uint64 i = 2;
////////
////////    while (getMaxFactor(currentN) >= i) {
////////        while (currentN % i == 0) {
////////            factorsAll.push_back(i);
////////            currentN /= i;
////////        }
////////        i++;
////////    }
////////
////////    if (currentN != 1) {
////////        factorsAll.push_back(currentN);
////////    }
////////
////////    time = clock() - start;
////////    curTimeCPU = time / 1.0;
////////
////////    printf("CPU compute time (Method of Enumerating Divisors): %.3f milliseconds\n", curTimeCPU);
////////    printf("Factors of number %lld: ", n);
////////
////////    printf("%lld", factorsAll[0]);
////////    for (uint64 i = 1; i < factorsAll.size(); i++) {
////////        printf(", %lld", factorsAll[i]);
////////    }
////////    printf("\n");
////////
////////    checkResult(factorsAll, n);
////////
////////    printf("\nCPU END (Method of Enumerating Divisors)\n");
////////    return factorsAll;
////////}
////////
////////void ferma(vector<uint64>& factorsAll, uint64 n) {
////////    uint64 currentN = n;
////////
////////    if (n == 1) {
////////        return;
////////    }
////////
////////    uint64 maxFactor = getMaxFactor(currentN);
////////    uint64 k = 0, x = 0, l = 0, a = 0, b = 0;
////////    double y = 0;
////////
////////    do {
////////        x = maxFactor + k;
////////        l = x * x - currentN;
////////        y = sqrt(l);
////////        k++;
////////
////////    } while ((uint64)y * (uint64)y != l);
////////
////////    a = x - (uint64)y;
////////    b = x + (uint64)y;
////////
////////    //printf("MaxFactor: %lld\t ThreadID-k: %lld\tx: %lld\tl: %.3e\ty: %.3e\ta: %lld\tb: %lld\n",
////////    //    maxFactor, k, x, l, y, a, b);
////////
////////    if (a == 1) {
////////        factorsAll.push_back(b);
////////    }
////////    else if (b == 1) {
////////        factorsAll.push_back(a);
////////    }
////////    else
////////    {
////////        ferma(factorsAll, a);
////////        ferma(factorsAll, b);
////////    }
////////}
////////
////////void factorsFermaCPU(uint64 n) {
////////    printf("\nCPU BEGIN: (Ferma method)\n");
////////    int start = 0, time = 0;
////////    float curTimeCPU = 0.0f, timeCPU = 0.0f;
////////
////////    uint64 currentN = n;
////////    vector<uint64> factorsAll = getFactors2(currentN);
////////    ferma(factorsAll, currentN);
////////
////////    time = clock() - start;
////////    curTimeCPU = time / 1.0;
////////
////////    printf("CPU compute time (Ferma method): %.3f milliseconds\n", curTimeCPU);
////////    printf("Factors of number %lld: ", n);
////////
////////    printf("%lld", factorsAll[0]);
////////    for (uint64 i = 1; i < factorsAll.size(); i++) {
////////        printf(", %lld", factorsAll[i]);
////////    }
////////
////////    printf("\n");
////////    checkResult(factorsAll, n);
////////    printf("\nCPU END (Ferma method)\n");
////////}
////////
//////////void factorsFermaGPU(uint64 n) {
//////////    //printf("\nGPU (Ferma method):\n");
//////////    //cudaEvent_t start, stop;
//////////    //float curTimeGPU = 0.0f, timeGPU = 0.0f;
//////////
//////////    //cudaEventCreate(&start);
//////////    //cudaEventCreate(&stop);
//////////    //cudaEventRecord(start, 0);
//////////
//////////    //// Выбор первого GPU для работы
//////////    //cudaSetDevice(0);
//////////
//////////    uint64 mult = 1;
//////////    uint64 currentN = n;
//////////    vector<uint64> factorsAll = getFactors2(currentN);
//////////    for (uint64 i = 0; i < factorsAll.size(); i++) {
//////////        mult *= factorsAll[i];
//////////    }
//////////
//////////    uint64 maxFactor = getMaxFactor(currentN);
//////////    uint64 factorsSize = 20;
//////////    uint64* factors = new uint64[factorsSize];
//////////    uint64* factorsDev;
//////////
//////////    cudaMalloc((void**)&factorsDev, factorsSize * sizeof(uint64));
//////////
//////////    dim3 blocks = 3;
//////////    dim3 threads = BLOCK_SIZE;
//////////    uint64 threadsCount = 3 * BLOCK_SIZE;
//////////
//////////    //printf("Grid Dim: %d %d %d  =>  Blocks: %d\n", blocks.x, blocks.y, blocks.z, blocks.x * blocks.y * blocks.z);
//////////    //printf("Block Dim: %d %d %d  =>  Threads: %d\n", threads.x, threads.y, threads.z, threads.x * threads.y * threads.z);
//////////
//////////    uint64 k = 0;
//////////    uint64 index = 0;
//////////    uint64 indexPush = 0;
//////////    bool isEnd = false;
//////////    uint64 temp1 = 0, temp2 = 0;
//////////    while (!isEnd) {
//////////        //if(threadsCount>currentN)
//////////
//////////        cudaFermaKernel << <blocks, threads >> > (factorsDev, currentN, getMaxFactor(currentN), k, indexPush);
//////////        cudaMemcpy(factors, factorsDev, factorsSize * sizeof(uint64), cudaMemcpyDeviceToHost);
//////////        cudaThreadSynchronize();
//////////
//////////        if (factors[index] == 0) {
//////////            k += threadsCount;
//////////        }
//////////        else if (factors[index] != 1 && factors[index + 1] != 1) {
//////////            k = 0;
//////////            currentN = factors[index];
//////////            temp1 = factors[index];
//////////            temp2 = factors[index + 1];
//////////        }
//////////        else if (factors[index] == 1 && factors[index + 1] == 1) {
//////////            isEnd = true;
//////////        }
//////////        else {
//////////            uint64 temp = 0;
//////////            if (temp1 == factors[index] && temp2 == factors[index + 1]) {
//////////                if (factors[index] == 1) {
//////////                    temp = factors[index + 1];
//////////                }
//////////                else {
//////////                    temp = factors[index];
//////////                }
//////////
//////////                factorsAll.push_back(temp);
//////////                mult *= temp;
//////////
//////////                currentN = (uint64)n / mult;
//////////                indexPush = index + 2;
//////////                index = indexPush;
//////////            }
//////////            else {
//////////                if (factors[index] == 1) {
//////////                    currentN = factors[index + 1];
//////////                }
//////////                else {
//////////                    currentN = factors[index];
//////////                }
//////////
//////////                k = 0;
//////////                temp1 = factors[index];
//////////                temp2 = factors[index + 1];
//////////            }
//////////
//////////            if (mult == n) {
//////////                isEnd = true;
//////////            }
//////////        }
//////////    }
//////////
//////////    cudaFree(factorsDev);
//////////
//////////   /* cudaEventRecord(stop, 0);
//////////    cudaEventSynchronize(stop);
//////////    cudaEventElapsedTime(&curTimeGPU, start, stop);
//////////
//////////    printf("DEVICE GPU compute time (Ferma method): %.3f milliseconds \n", curTimeGPU);*/
//////////    printf("%lld", factorsAll[0]);
//////////    for (uint64 i = 1; i < factorsAll.size(); i++) {
//////////        printf(", %lld", factorsAll[i]);
//////////    }
//////////
//////////    printf("\n");
//////////    checkResult(factorsAll, n);
//////////    printf("\nGPU END (Ferma method)\n");
//////////}
////////
////////void factorsFermaGPU(uint64 n) {
////////    uint64 mult = 1;
////////    uint64 currentN = n;
////////    vector<uint64> factorsAll = getFactors2(currentN);
////////    for (uint64 i = 0; i < factorsAll.size(); i++) {
////////        mult *= factorsAll[i];
////////    }
////////
////////    uint64 maxFactor = getMaxFactor(currentN);
////////
////////    printf("mult: %lld, maxFactor: %lld \n", mult, maxFactor);
////////
////////    uint64 factorsSize = maxFactor * maxFactor;
////////    uint64* factors = new uint64[factorsSize];
////////    uint64* factorsDev;
////////
////////    cudaMalloc((void**)&factorsDev, factorsSize * sizeof(uint64));
////////
////////    dim3 threads = 21;
////////    dim3 blocks = 1; maxFactor / threads.x;
////////    ////uint64 threadsCount = 3 * BLOCK_SIZE;
////////
////////    cudaFermaKernel<<<blocks, threads>>>(factorsDev, currentN, getMaxFactor(currentN));
////////    cudaMemcpy(factors, factorsDev, factorsSize * sizeof(uint64), cudaMemcpyDeviceToHost);
////////    cudaThreadSynchronize();
////////
////////    //printf("%lld", factorsAll[0]);
////////    //for (uint64 i = 1; i < factorsAll.size(); i++) {
////////    //    printf(", %lld", factorsAll[i]);
////////    //}
////////
////////    //printf("\n");
////////    //checkResult(factorsAll, n);
////////    //printf("\nGPU END (Ferma method)\n");
////////}
////////
////////int main()
////////{
////////    uint64 n = 405;
////////
////////    //getFactorsAll(n);
////////    //factorsFermaCPU(n);
////////    factorsFermaGPU(n);
////////
////////    return 0;
////////}
//////
//////
//////#include <cstdio>
//////#include <cstdlib>
//////#include <vector>
//////#include <cuda_runtime.h>
//////#include <device_launch_parameters.h>
//////#include <time.h>
//////
//////using namespace std;
//////
//////typedef unsigned long long uint64;
//////#define BLOCK_SIZE 256
//////
//////bool checkResult(vector<uint64> factors, uint64 n) {
//////    uint64 result = 0;
//////
//////    if (factors.size() == 0) {
//////        return result == n;
//////    }
//////
//////    result = factors[0];
//////    if (factors.size() == 1) {
//////        return result == n;
//////    }
//////
//////    printf("%lld ", factors[0]);
//////    for (int i = 1; i < factors.size(); i++) {
//////        result *= factors[i];
//////        printf("* %lld ", factors[i]);
//////    }
//////
//////    printf("= %lld => %s", result, (result == n) ? "true" : "false");
//////
//////    return result == n;
//////}
//////
//////vector<uint64> getFactors2(uint64& n) {
//////    vector<uint64> factors2;
//////
//////    while (n % 2 == 0) {
//////        factors2.push_back(2);
//////        n /= 2;
//////    }
//////
//////    return factors2;
//////}
//////
//////__device__ __host__ uint64 getMaxFactor(uint64 n) {
//////    uint64 maxFactor = sqrt(n);
//////
//////    if (maxFactor * maxFactor < n) {
//////        maxFactor++;
//////    }
//////
//////    return maxFactor;
//////}
//////
//////vector<uint64> getFactorsAll(uint64 n) {
//////    printf("\nCPU BEGIN (Method of Enumerating Divisors)\n");
//////    int start = 0, time = 0;
//////    float curTimeCPU = 0.0f, timeCPU = 0.0f;
//////
//////    vector<uint64> factorsAll;
//////    uint64 currentN = n;
//////    uint64 i = 2;
//////
//////    while (getMaxFactor(currentN) >= i) {
//////        while (currentN % i == 0) {
//////            factorsAll.push_back(i);
//////            currentN /= i;
//////        }
//////        i++;
//////    }
//////
//////    if (currentN != 1) {
//////        factorsAll.push_back(currentN);
//////    }
//////
//////    time = clock() - start;
//////    curTimeCPU = time / 1.0;
//////
//////    printf("CPU compute time (Method of Enumerating Divisors): %.3f milliseconds\n", curTimeCPU);
//////    printf("Factors of number %lld: ", n);
//////
//////    printf("%lld", factorsAll[0]);
//////    for (uint64 i = 1; i < factorsAll.size(); i++) {
//////        printf(", %lld", factorsAll[i]);
//////    }
//////    printf("\n");
//////
//////    checkResult(factorsAll, n);
//////
//////    printf("\nCPU END (Method of Enumerating Divisors)\n");
//////    return factorsAll;
//////}
//////
//////void del(int n) {
//////    int maxFactor = getMaxFactor(n);
//////
//////    for (int i = maxFactor; i > 1; i--) {
//////        printf("%d / %d = %d (%d)\n", n, i, int(n / i), n % i);
//////    }
//////}
//////
//////__global__ void getFactorsGlobalKernel(uint64* allFactors, uint64 n, uint64 maxFactor) {
//////    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
//////
//////    allFactors[threadId] = n / (threadId + 2);
//////    allFactors[threadId + maxFactor] = n % (threadId + 2);
//////
//////    //uint64 countN = getMaxFactor(allFactors[threadId]);
//////
//////    //for (uint64 i = 0; i < countN; i++) {
//////    //     
//////    //}
//////
//////    printf("TID: %d\tf[%d]: %lld\tf[%d]: %lld\n", threadId, threadId, allFactors[threadId],
//////        threadId + maxFactor, allFactors[threadId + maxFactor]);
//////}
//////
//////void getFactorsGPU_Global(uint64 n) {
//////
//////    uint64 maxFactor = getMaxFactor(n);
//////    maxFactor--;
//////    uint64 computeArraySize = maxFactor * 4;
//////
//////
//////    printf("maxFactor: %lld\n", maxFactor);
//////
//////    uint64* allFactors = new uint64[computeArraySize];
//////
//////    int nBytes = computeArraySize * sizeof(uint64);
//////    dim3 threads(4);
//////    dim3 blocks(maxFactor / threads.x);
//////
//////    uint64* devAllFactors;
//////
//////    // allocate DRAM
//////    cudaMalloc((void**)&devAllFactors, nBytes);
//////
//////    // copy from CPU to DRAM
//////    cudaMemcpy(devAllFactors, allFactors, nBytes, cudaMemcpyHostToDevice);
//////
//////    getFactorsGlobalKernel<<<blocks, threads>>>(devAllFactors, n, maxFactor);
//////
//////    cudaThreadSynchronize();
//////    cudaMemcpy(allFactors, devAllFactors, nBytes, cudaMemcpyDeviceToHost);
//////
//////    // free GPU memory
//////    cudaFree(devAllFactors);
//////
//////    //for (uint64 i = 0; i < maxFactor; i++) {
//////    //    printf("AllFactor[%lld]: %lld \t %lld \n", i, allFactors[i], allFactors[i + maxFactor]);
//////    //}
//////}
//////
//////int main()
//////{
//////    uint64 n = 100000050500;
//////
//////    //getFactorsAll(n);
//////    getFactorsGPU_Global(405);
//////
//////    return 0;
//////}