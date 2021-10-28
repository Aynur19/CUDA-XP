#ifndef __CHAPTER2_H__
#define __CHAPTER2_H__

#include "../../_common/helper.h"
#include "../../_common/helper.cuh"


void vectorAdd_RAPI(const int blockSize, const int numBlocks, const int numItems);

void getGpuFeatures();

bool vectorAddValidate(float* vec1, float* vec2, float* vecSum, int numItems);
#endif // !__CHAPTER2_H__
