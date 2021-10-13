#ifndef __SQUARE_MATRIX_MULTIPLICATION__
#define __SQUARE_MATRIX_MULTIPLICATION__

#include "../../_common/helper.h"

#define N 4
#define BLOCK_SIZE 2

void squareMatrixMultiplicationCPU(unsigned int verbose);

void squareMatrixMultiplicationGPU_Shared(unsigned int verbose);

void squareMatrixMultiplicationGPU_Global(unsigned int verbose);

#endif // !__SQUARE_MATRIX_MULTIPLICATION__
