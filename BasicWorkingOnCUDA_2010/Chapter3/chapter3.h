#ifndef __CHAPTER3_H__
#define __CHAPTER3_H__

#include "..\_common\helper.h"

extern "C" void cuMatrixSquareTranspose();

extern "C" void cuBuildTable(float* res, int n, float step);

extern "C" void cuSquareMatricesMultiplication();

extern "C" void cuIntegrateBodies();

#endif // !__CHAPTER3_H__