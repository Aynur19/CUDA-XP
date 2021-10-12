#include "chapter3.h"

#define EPS 0.0001f
#define N (16*1024)
#define BLOCK_SIZE 256

__global__ void __cuIntegrateBodies(float3* newPos, float3* newVel, 
									float3* oldPos, float3* oldVel, float dt) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float3 pos = oldPos[index];
	float3 f = make_float3(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < N; i++)
	{
		float3 pi = oldPos[i];
		float3 r;

		// вектор от текущей точки pi
		r.x = pi.x - pos.x;
		r.y = pi.y - pos.y;
		r.z = pi.z - pos.z;

		// использование ESP^2, чтобы не было деления на 0
		float invDist = 1.0f / sqrtf(r.x * r.x + r.y * r.y + r.z * r.z + EPS * EPS);
		float s = invDist * invDist * invDist;

		// добавление к сумме всех силу, вызванную i-м телом
		f.x += r.x * s;
		f.y += r.y * s;
		f.z += r.z * s;
	}

	float3 vel = oldVel[index];

	vel.x += f.x * dt;
	vel.y += f.y * dt;
	vel.z += f.z * dt;

	pos.x += vel.x * dt;
	pos.y += vel.y * dt;
	pos.z += vel.z * dt;

	newPos[index] = pos;
	newVel[index] = vel;
}

extern "C" void cuIntegrateBodies() {
	float3* pos = new float3[N];
	float3* vel = new float3[N];
	float3* posDev[2] = { NULL, NULL };
	float3* velDev[2] = { NULL, NULL };

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	int index = 0;

	randomInitF3(pos, N);
	randomInitF3(vel, N);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&posDev[0], N * sizeof(float3));
	cudaMalloc((void**)&velDev[0], N * sizeof(float3));
	cudaMalloc((void**)&posDev[1], N * sizeof(float3));
	cudaMalloc((void**)&velDev[1], N * sizeof(float3));

	for (int i = 0; i < 2; i++, index ^= 1)
	{
		__cuIntegrateBodies<<<dim3(N / BLOCK_SIZE), dim3(BLOCK_SIZE)>>>
			(posDev[index^1], velDev[index^1], 
				posDev[index], velDev[index], 0.01f);
	}

	cudaMemcpy(pos, posDev[index ^ 1], N * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(vel, velDev[index ^ 1], N * sizeof(float3), cudaMemcpyDeviceToHost);

	cudaFree(posDev[0]);
	cudaFree(velDev[0]);
	cudaFree(posDev[1]);
	cudaFree(velDev[1]);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	printf("Elapsed time: %.3f milliseconds\n", gpuTime);

	delete pos;
	delete vel;
}