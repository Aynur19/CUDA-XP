#include "chapter2.h"

int main(int argc, char* argv[]) {

	int cmd = -1;

	printf("Enter the available command to proceed:\n");
	printf("  0: End of the program\n");
	printf("  1: Vector addition\n");
	printf("  2: Derivation of the characteristics and capabilities of the GPU\n");


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

				vectorAdd_RAPI(blockSize, numBlocks, numItems);
				//cuVectorAdd_DAPI(blockSize, numBlocks, numItems);
				break;
			}
			case 2: {
				getGpuFeatures();
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