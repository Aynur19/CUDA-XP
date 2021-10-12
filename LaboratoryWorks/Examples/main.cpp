extern "C" void matrixMultiplication(unsigned int iters, unsigned int verbose);
extern "C" void matrixMultiplicationCPU(unsigned int verbose);

int main(int argc, char* argv[])
{
	matrixMultiplication(10, 0);
	//matrixMultiplicationCPU();
	//gpuComputedMethod = &matrixMultiplicationGlobal;

	//gpuTimeMeasuring(gpuComputedMethod, 10);
	////matrixMultiplicationGlobal();

	//// cudaDeviceReset causes the driver to clean up all state. 
	//// While not mandatory in normal operation, it is good practice.  
	//// It is also needed to ensure correct operation when the application is being profiled. 
	//// Calling cudaDeviceReset causes all profile data to be flushed before the application exits
	//cudaDeviceReset();

	return 0;
}