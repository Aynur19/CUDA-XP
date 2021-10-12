#define _USE_MATH_DEFINES

#include <cstdio>
#include <iostream>
#include <math.h>

#define EPS 0.00001

using namespace std;


void findEquationRoot(float x, float stepX, int maxIters) {
    int iter = 0;
    float temp = x;
    float step = stepX;
    float currentX = x;

    do {
        temp = 1 / (sin(M_PI * currentX / 180));
        currentX += step;
        iter++;

        printf("Iteration %d: \t result: %.3f \t x: %.3f \t abs(result-x): %.7f\n",
            iter, temp, currentX, fabs(currentX - temp));
    } while (fabs(currentX - temp) > EPS && iter < maxIters);
}

int main(int argc, char* argv) {
    float x;
    float stepX;
    int maxIters;

    printf("Finding the roots of the equation 'f(x) => sin(x) = 1/x'\n");
    printf("Enter initial approximation: ");
    scanf("%f", &x);

    printf("Enter the step X: ");
    scanf("%f", &stepX);

    printf("Enter the maximum iterations: ");
    scanf("%d", &maxIters);

    findEquationRoot(x, stepX, maxIters);

    return 0;
}
