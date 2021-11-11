#ifndef __SQUARE_MATRIX_MULTIPLICATION__
#define __SQUARE_MATRIX_MULTIPLICATION__

#include "../../_common/helper.h"

#define N 4
#define BLOCK_SIZE 2

argsVector squareMatrixMultiplicationCPU(argsVector argsIn);

argsVector squareMatrixMultiplicationGPU_Shared(argsVector argsIn);

argsVector squareMatrixMultiplicationGPU_Global(argsVector argsIn);

#endif // !__SQUARE_MATRIX_MULTIPLICATION__
