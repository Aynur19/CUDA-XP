#include "labWork3.h"


float getFunctionDerivative_Option19(float x) {
    return 0.51 * powf(x, 2.0) - 1.14 * x - 1.6;
}

void checkRootEquationFinding(float x) {
    printf("x: 0.17 * %.5f^3 - 0.57 * %.5f^2 - 1.6 * %.5f + 3.7 = 0.17 * %.5f - 0.57 * %.5f - 1.6 * %.5f + 3.7 = %.5f - %.5f - %.5f + 3.7 = %.5f\n",
            x, x, x,
            pow(x, 3), pow(x, 2), x,
            0.17 * pow(x, 3), 0.57 * pow(x, 2), 1.6 * x,
            0.17 * pow(x, 3) - 0.57 * pow(x, 2) - 1.6 * x + 3.7);
}

argsVector option19_rootEquationFindingCPU(argsVector argsIn) {
    argsVector argsOut;

    float startX = getValueFromArgs<float>("--startX", -10.0, argsIn);
    float endX = getValueFromArgs<float>("--endX", 10.0, argsIn);

    float lambda = 1 / getFunctionDerivative_Option19(startX);

    float startY = getEquationValue_Option19(startX);
    float endY = getEquationValue_Option19(endX);

    float currentX = startX;
    float currentY = startY;
    
    while (fabs(currentY) > EPS) {
        lambda = 1 / getFunctionDerivative_Option19(currentX);
        currentX = currentX - lambda * getEquationValue_Option19(currentX);
        currentY = getEquationValue_Option19(currentX);

        printf("%.7f\n", currentY);
    }

    printf("x = %.7f\n", currentX);
    printf("y = %.7f\n", currentY);
    checkRootEquationFinding(currentX);

    argsOut.push_back("--root " + std::to_string(currentX));
    return argsOut;
}
