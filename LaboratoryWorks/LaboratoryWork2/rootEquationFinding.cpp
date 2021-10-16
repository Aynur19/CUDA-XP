#include "rootEquationFinding.h"

bool checkRootEquationFinding(float x) {
    float sinX = sinf(M_PI * x / 180);
    float xDel = 1 / x;

    if (fabs(sinX - xDel) <= EPS) {
        printf("x: %.7f =>\tsin(%.7f) = 1/%.7f =>\t%.7f = %.7f => \tabc()=> %.7f\n", x, x, x, sinX, xDel, fabs(sinX - xDel));
        return true;
    }
    else {
        printf("x: %.7f =>\tsin(%.7f) != 1/%.7f =>\t%.7f != %.7f =>\tabc()=> %.7f\n", x, x, x, sinX, xDel, fabs(sinX - xDel));
        return false;
    }
}

float getSignedStep(float startX, float endX, float stepX) {
    float sign = 1.0f;
    if (startX > endX) {
        sign = -1.0;
    }

    if (stepX < 0) {
        stepX *= -1 * sign;
    }
    else {
        stepX *= sign;
    }

    return stepX;
}

std::tuple<dim3, float> getOptimalParameters(float startX, float endX, float stepX, dim3 blockDimension) {
    int blockThreads = blockDimension.x * blockDimension.y * blockDimension.z;
    int needThreads = fabs((startX - endX) / stepX);

    int needBlocks = needThreads / blockThreads;
    int gridSize = sqrtf(needBlocks) + 1;

    if (gridSize > 1024) {
        gridSize = 1024;
    }

    stepX = fabs(startX - endX) / (gridSize * gridSize * blockThreads);
    return std::make_tuple(dim3(gridSize, gridSize), stepX);
}