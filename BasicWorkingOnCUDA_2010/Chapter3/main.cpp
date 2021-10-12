#include "chapter3.h"

int main(int argc, char* argv[]) {

	int cmd = -1;

	printf("Enter the available command to proceed:\n");
	printf("  0: End of the program\n");
	printf("  1: Building a table by function values with a given step\n");
	printf("  2: Square Matrix Transposing\n");
	printf("  3: Square Matrix Multiplication\n");
	printf("  4: The Problem of N Bodies\n");


	printf("Enter the command: ");
	scanf("%d", &cmd);

	while (cmd != 0)
	{

		switch (cmd)
		{
			case 0:
				break;
			case 1: {
				int n = 256 * 2;
				float* res = new float[n];
				float step = 0.05;
				cuBuildTable(res, n, step);
				break;
			}
			case 2: {
				cuMatrixSquareTranspose();
				break;
			}
			case 3: {
				cuSquareMatricesMultiplication();
				break;
			}
			case 4: {
				cuIntegrateBodies();
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