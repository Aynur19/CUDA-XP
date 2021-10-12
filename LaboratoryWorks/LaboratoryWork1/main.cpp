#include <stdio.h>

extern "C" void cudaPrintNumbers();

int main(int argc, char* argv)
{
    int cmd = -1;
	//printf("Enter one of the following commands:\n");
	//printf("  0: Get information about the GPU\n");
	//printf("  4: Task 4. Print numbers\n");
	//printf("\n====================================================================\n");
	//scanf("%d", &cmd);

    //switch (cmd)
    //{
    //    case 4:
    //        printNumbers();
    //        break;
    //    case 0:
    //    default:
    //        getDeviceInfo();
    //        break;
    //}


    while (cmd != 0)
    {
        cudaPrintNumbers();

        printf("Enter number to continue the process ...");
        scanf("%d", &cmd);
    }
}