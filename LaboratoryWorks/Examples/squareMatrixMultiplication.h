#ifndef __SQUARE_MATRIX_MULTIPLICATION__
#define __SQUARE_MATRIX_MULTIPLICATION__

#include "../../_common/helper.h"

#define N 4
#define BLOCK_SIZE 2

void squareMatrixMultiplicationCPU(int argc, char* argv[]);

void squareMatrixMultiplicationGPU_Shared(int argc, char* argv[]);

void squareMatrixMultiplicationGPU_Global(int argc, char* argv[]);

#endif // !__SQUARE_MATRIX_MULTIPLICATION__
