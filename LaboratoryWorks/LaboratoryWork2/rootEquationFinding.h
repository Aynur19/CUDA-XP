#ifndef __ROOT_EQUATION_FINDING__
#define __ROOT_EQUATION_FINDING__

#include "../../_common/helper.h"
#include "../../_common/helper.cuh"
#include <tuple>

#define EPS 0.00001

#define BLOCK_DIM dim3(16, 16)
#define THREADS 16 * 16

/// <summary>
/// Вычисление корня уравнения (sin(x)=1/x) на CPU.
/// </summary>
argsVector rootEquationFindingCPU(argsVector argsIn);

/// <summary>
/// Вычисление корня уравнения (sin(x)=1/x) на GPU с использованием Shared Memory.
/// </summary>
argsVector rootEquationFindingGPU_Shared(argsVector argsIn);

/// <summary>
/// Вычисление корня уравнения (sin(x)=1/x) на GPU с использованием Global Memory.
/// </summary>
argsVector rootEquationFindingGPU_Global(argsVector argsIn);

/// <summary>
/// Проверка результата нахождения корня.
/// </summary>
/// <param name="x">Найденный корень уравнения: sin(x)=1/x</param>
bool checkRootEquationFinding(float x);

/// <summary>
/// Получение шага изменения X с правильным знаком.
/// </summary>
float getSignedStep(float startX, float endX, float stepX);

/// <summary>
/// Получение оптимальных параметров (размерность сетки, шаг X). 
/// </summary>
std::tuple<dim3, float> getOptimalParameters(float startX, float endX, float stepX, dim3 blockDimension);

#endif // !__ROOT_EQUATION_FINDING__
