#ifndef __CHAPTER2_H__
#define __CHAPTER2_H__

#include "..\_common\helper.h"

extern "C" void cuVectorAdd_RAPI(const int blockSize, const int numBlocks, const int numItems);

extern "C" void cuGetGpuFeatures();
#endif // !__CHAPTER2_H__
