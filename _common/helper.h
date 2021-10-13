#ifndef __HELPER_H__
#define __HELPER_H__

#include <cstdio>
#include <ctime>

#define MATRIX_INDEX(row, col, columnsInRow) ((col) + (row) * columnsInRow)

void cpuTimeMeasuring(void (*cpuComputedMethod)(unsigned int verbose), unsigned int iters, unsigned int verbose = 0);

void matrixFillIndices(float* matrix, int nRows, int nCols);

void matrixPrint(float* matrix, int nRows, int nCols);


#endif // !__HELPER_H__
