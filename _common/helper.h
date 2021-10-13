#ifndef __HELPER_H__
#define __HELPER_H__

#include <cstdio>
#include <ctime>
#include <string>

using namespace std;

#define MATRIX_INDEX(row, col, columnsInRow) ((col) + (row) * columnsInRow)

template<typename T> T getValueFromArgv(char* paramName, T defaultValue, int argc, char* argv[]) {
    T param = defaultValue;
    for (int i = 0; i < argc; i++) {
        std::string strName(argv[i]);
        if (strName == paramName) {
            std::string str(argv[i + 1]);
            try {
                param = std::stoi(str);
            }
            catch (exception ex) {
                printf(ex.what());
            }
            break;
        }
    }

    return param;
}


void cpuTimeMeasuring(void (*cpuComputedMethod)(int argc, char* argv[]), unsigned int iters, int argc, char* argv[]);

void matrixFillIndices(float* matrix, int nRows, int nCols);

void matrixPrint(float* matrix, int nRows, int nCols);


#endif // !__HELPER_H__
