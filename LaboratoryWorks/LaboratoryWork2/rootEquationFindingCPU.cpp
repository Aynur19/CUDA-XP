#include "rootEquationFinding.h"

argsVector rootEquationFindingCPU(argsVector argsIn) {
    argsVector argsOut;

    float startX = getValueFromArgs<float>("--startX", 0.0, argsIn);
    float endX = getValueFromArgs<float>("--endX", 1.0, argsIn);
    float stepX = getValueFromArgs<float>("--stepX", 0.00001, argsIn);

    stepX = getSignedStep(startX, endX, stepX);

    float currentX = startX;
    float resultX, nextY;

    do {
        resultX = 1 / (sin(M_PI * currentX / 180));
        float fabsX = fabs(resultX - currentX);
        if (fabsX <= EPS) {
            break;
        }

        currentX += stepX;
    } while (stepX < fabs(currentX - endX));

    checkRootEquationFinding(currentX);
    
    argsOut.push_back("--root " + std::to_string(currentX));
    return argsOut;
}

