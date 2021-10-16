#ifndef __ROOT_EQUATION_FINDING__
#define __ROOT_EQUATION_FINDING__

#include "../../_common/helper.h"
#include "../../_common/helper.cuh"
#include <tuple>

#define EPS 0.00001

#define BLOCK_DIM dim3(16, 16)
#define THREADS 16 * 16

/// <summary>
/// ���������� ����� ��������� (sin(x)=1/x) �� CPU.
/// </summary>
argsVector rootEquationFindingCPU(argsVector argsIn);

/// <summary>
/// ���������� ����� ��������� (sin(x)=1/x) �� GPU � �������������� Shared Memory.
/// </summary>
argsVector rootEquationFindingGPU_Shared(argsVector argsIn);

/// <summary>
/// ���������� ����� ��������� (sin(x)=1/x) �� GPU � �������������� Global Memory.
/// </summary>
argsVector rootEquationFindingGPU_Global(argsVector argsIn);

/// <summary>
/// �������� ���������� ���������� �����.
/// </summary>
/// <param name="x">��������� ������ ���������: sin(x)=1/x</param>
bool checkRootEquationFinding(float x);

/// <summary>
/// ��������� ���� ��������� X � ���������� ������.
/// </summary>
float getSignedStep(float startX, float endX, float stepX);

/// <summary>
/// ��������� ����������� ���������� (����������� �����, ��� X). 
/// </summary>
std::tuple<dim3, float> getOptimalParameters(float startX, float endX, float stepX, dim3 blockDimension);

#endif // !__ROOT_EQUATION_FINDING__
