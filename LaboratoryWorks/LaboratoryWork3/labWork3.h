#ifndef __LAB_WORK3_H__
#define __LAB_WORK3_H__

#include "../../_common/helper.h"
#include "../../_common/helper.cuh"
#include <tuple>

#define EPS 0.001

float getEquationValue_Option19(float x);

float getFunctionDerivative_Option19(float x);

void checkRootEquationFinding(float x);

argsVector option19_rootEquationFindingCPU(argsVector argsIn);

argsVector option19_rootEquationFindingGPU(argsVector argsIn);

argsVector option19_rootEquationFindingGPU_Thrust(argsVector argsIn);


#endif // !__LAB_WORK3_H__
