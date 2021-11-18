#include "labWork3.h"


float getFunctionDerivative_Option19(float x) {
    return 0.51 * powf(x, 2.0) - 1.14 * x - 1.6;
}


void printAllXY(argsVector argsIn) {
    int count = getValueFromArgs<int>("--rootsCount", 0, argsIn);
    float x = 0.0;
    float y = 0.0;
    for (int i = 0; i < count; i++) {
        x = getValueFromArgs<float>("--x[" + std::to_string(i) + "]", 0.0, argsIn);
        y = getValueFromArgs<float>("--y[" + std::to_string(i) + "]", 0.0, argsIn);

        printf("x: %.7f \t y: %.7f \n", x, y);
    }
}

bool getPassPrediction(float currentX, float endX, int isStraightPass = 1) {
    if (isStraightPass > 0) {
        return currentX <= endX;
    }
    else {
        return currentX >= endX;
    }
}

argsVector option19_rootEquationFindingByPass(argsVector argsIn, int isStraightPass = 1) {
    argsVector argsOut;

    float startX = 0.0;
    float endX = 0.0;

    if (isStraightPass > 0) {
        startX = getValueFromArgs<float>("--startX", -10.0, argsIn);
        endX = getValueFromArgs<float>("--endX", 10.0, argsIn);
    }
    else {
        startX = getValueFromArgs<float>("--endX", -10.0, argsIn);
        endX = getValueFromArgs<float>("--startX", 10.0, argsIn);
    }

    float lambda = isStraightPass / getFunctionDerivative_Option19(startX);

    float startY = getEquationValue_Option19(startX);
    float endY = getEquationValue_Option19(endX);

    float currentX = startX;
    float currentY = startY;

    int count = 0;
    while (getPassPrediction(currentX, endX, isStraightPass)) {
        while (fabs(currentY) > EPS && getPassPrediction(currentX, endX, isStraightPass)) {
            lambda = isStraightPass / getFunctionDerivative_Option19(currentX);
            currentX = currentX + isStraightPass*fabs(lambda * getEquationValue_Option19(currentX));
            currentY = getEquationValue_Option19(currentX);
        }

        while (fabs(currentY) < EPS && getPassPrediction(currentX, endX, isStraightPass)) {
            argsOut.push_back("--x[" + std::to_string(count) + "] " + std::to_string(currentX));
            argsOut.push_back("--y[" + std::to_string(count) + "] " + std::to_string(currentY));
            currentX = currentX + isStraightPass * EPS / 2;
            currentY = getEquationValue_Option19(currentX);
            count++;
        }
    }

    argsOut.push_back("--rootsCount " + std::to_string(count));
    printAllXY(argsOut);
    return argsOut;
}

argsVector option19_rootEquationFindingCPU(argsVector argsIn) {
    argsVector argsOut;

    int count = 0;
    
    argsVector argsCurr = option19_rootEquationFindingByPass(argsIn, 1);
    int curCount = getValueFromArgs<int>("----rootsCount", 0, argsCurr);
    count += curCount;

    for (int i = 0; i < curCount; i++) {
        argsOut.push_back(argsOut[i]);
    }

    argsCurr = option19_rootEquationFindingByPass(argsIn, -1);
    curCount = getValueFromArgs<int>("----rootsCount", 0, argsCurr);
    count += curCount;

    for (int i = 0; i < curCount; i++) {
        argsOut.push_back(argsOut[i]);
    }

    argsOut.push_back("--rootsCount " + std::to_string(count));
    printAllXY(argsOut);
    return argsOut;
}


