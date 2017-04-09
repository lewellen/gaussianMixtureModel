#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif

#include "cudaWrappers.h"

void testSingleStdNormLogL() {
	const size_t numPoints = 1024;
	const size_t numComponents = 1;

	double logPi[numComponents];
	double logUniform = -log((double)numComponents);
	for(size_t k = 0; k < numComponents; ++k) {
		logPi[k] = logUniform;
	}

	double logProb[numComponents * numPoints];
	for(size_t k = 0; k < numComponents; ++k) {
		for(size_t i = 0; i < numPoints; ++i) {
			double x = 3.0 * ((double)i - (double)(numPoints/2.0))/((double)numPoints/2.0);
			logProb[numPoints * k + i] = -0.5 * sqrt(2.0 * M_PI) - 0.5 * x * x;
		}
	}
		
	parallelGmmLogLikelihood(
		numPoints, numComponents,
		logPi, logProb
	);

	double expected = 0;
	for(size_t k = 0; k < numComponents; ++k) {
		for(size_t i = 0; i < numPoints; ++i) {
			double x = 3.0 * ((double)i - (double)(numPoints/2.0))/((double)numPoints/2.0);
			expected += logPi[k] - 0.5 * sqrt(2.0 * M_PI) - 0.5 * x * x;
		}
	}

	double actual = 0;
	for(size_t i = 0; i < numPoints; ++i) {
		actual += logProb[i];
	}

	double absDiff = abs(expected - actual);
	if(absDiff >= DBL_EPSILON) {
		printf("actual = %.15f, expected = %.15f, absDiff = %.15f\n",
			actual, expected, absDiff);
	}

	assert(absDiff < DBL_EPSILON);
}

int main(int argc, char** argv) {
	testSingleStdNormLogL();

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}
