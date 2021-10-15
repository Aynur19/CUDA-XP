#ifndef __HELPER_H__
#define __HELPER_H__

#define _USE_MATH_DEFINES

#include <cstdio>
#include <ctime>
#include <string>
#include <typeinfo>
#include <vector>
#include <math.h>

using namespace std;

typedef std::vector<std::string> argsVector;

#define MATRIX_INDEX(row, col, columnsInRow) ((col) + (row) * columnsInRow)

template<typename T> char* toChars(T param) {
    char convertedParam[sizeof(T)];
    if (typeid(param).name() == "float") {
        sprintf(convertedParam, "%f", param);
    }
    else if(typeid(param).name() == "int") {
        sprintf(convertedParam, "%d", param);
    }
    //const char* convertedParam = xChar;

    return convertedParam;
}

template<typename T> T getValueFromArgs(std::string paramName, T defaultValue, argsVector args) {
	T param = defaultValue;
	for (int i = 0; i < args.size(); i++) {
		if (args[i].find(paramName) != std::string::npos) {
			string strParam = args[i].substr(paramName.size() + 1);
			string typeParam = string(typeid(defaultValue).name());
			try {
				//printf("%s\t", strParam.c_str());
				if (typeParam == "int" || typeParam == "unsigned int") {
					param = std::stoi(strParam);
				}
				else if (typeParam == "long") {
					param = std::stol(strParam);
				}
				else if (typeParam == "unsigned long") {
					param = std::stoul(strParam);
				}
				else if (typeParam == "long long") {
					param = std::stoll(strParam);
				}
				else if (typeParam == "unsigned long long") {
					param = std::stoull(strParam);
				}
				else if (typeParam == "float") {
					param = std::stof(strParam);
				}
				else if (typeParam == "double") {
					param = std::stod(strParam);
				}
				else if (typeParam == "long double") {
					param = std::stold(strParam);
				}
			}
			catch (exception ex) {
				printf(ex.what());
			}
			break;
		}
	}

	return param;
}

argsVector cpuTimeMeasuring(argsVector(*cpuComputedMethod)(argsVector argsIn), unsigned int iters, argsVector argsIn);

void matrixFillIndices(float* matrix, int nRows, int nCols);

void matrixPrint(float* matrix, int nRows, int nCols);


#endif // !__HELPER_H__
