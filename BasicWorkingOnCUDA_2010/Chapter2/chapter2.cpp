#include "chapter2.h"

bool vectorAddValidate(float* vec1, float* vec2, float* vecSum, int numItems)
{
    bool result = true;
    for (int i = 0; i < numItems; i++)
    {
        if (fabs(vec1[i] + vec2[i] - vecSum[i]) > EPS)
        {
            printf("Error at index %d\n", i);

            result = false;
        }
        else
        {
            printf("Sum at index %d: %f = %f + %f\n", i, vecSum[i], vec1[i], vec2[i]);
        }
    }

    return result;
}