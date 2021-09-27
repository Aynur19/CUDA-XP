#include <stdio.h>

extern "C" void cuVectorAdd_RAPI(const int blockSize, const int numBlocks, const int numItems);
//extern "C" void cuVectorAdd_DAPI(const int blockSize, const int numBlocks, const int numItems);

extern "C" void cuGetGpuFeatures();
extern "C" void cuBuildTable(float* res, int n, float step);
extern "C" void cuMatrixSquareTranspose();
extern "C" void cuSquareMatricesMultiplication();

int main(int argc, char* argv[]) {

	int cmd = -1;

	printf("Enter the available command to proceed:\n");
	printf("  0: End of the program\n");
	printf("  1: Vector addition\n");
	printf("  2: Derivation of the characteristics and capabilities of the GPU\n");
	printf("  3: Building a table by function values with a given step\n");
	printf("  4: Square Matrix Transposing\n");
	printf("  5: Square Matrix Multiplication\n");


	printf("Enter the command: ");
	scanf("%d", &cmd);

	while (cmd != 0)
	{

		switch (cmd)
		{
			case 0:
				break;
			case 1: {
				const unsigned int blockSize = 512;
				const unsigned int numBlocks = 3;
				const unsigned int numItems = numBlocks * blockSize;

				cuVectorAdd_RAPI(blockSize, numBlocks, numItems);
				//cuVectorAdd_DAPI(blockSize, numBlocks, numItems);
				break;
			}
			case 2: {
				cuGetGpuFeatures();
				break;
			}
			case 3: {
				int n = 256 * 2;
				float* res = new float[n];
				float step = 0.05;
				cuBuildTable(res, n, step);
				break;
			}
			case 4: {
				cuMatrixSquareTranspose();
				break;
			}
			case 5: {
				cuSquareMatricesMultiplication();
				break;
			}
			default: {
				printf("Invalid the command!");
				break;
			}
		}

		printf("Enter the command: ");
		scanf("%d", &cmd);
	}
}