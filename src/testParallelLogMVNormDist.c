#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif

#include "cudaWrappers.h"

void test1DStandardNormal() {
	const size_t pointDim = 1;
	const size_t numPoints = 1024;

	double sigmaL[pointDim * pointDim];
	memset(sigmaL, 0, pointDim * pointDim * sizeof(double));
	for(size_t i = 0; i < pointDim; ++i) {
		sigmaL[i * pointDim + i] = 1;
	}

	double det = 1;
	for(size_t i = 0; i < pointDim; ++i) {
		det *= sigmaL[i * pointDim + i] * sigmaL[i * pointDim + i];
	}

	double logNormalizer = -0.5 * pointDim * log(2.0 * M_PI) - 0.5 * log(det);

	double mu[pointDim];
	memset(mu, 0, pointDim * sizeof(double));

	double X[pointDim * numPoints];
	memset(X, 0, pointDim * numPoints * sizeof(double));
	for(size_t i = 0; i < numPoints; ++i) {
		X[i * pointDim + 0] = 3.0 * ( ( (double)i - (double)numPoints/2 ) / (double)(numPoints/2.0) );
	}

	double logP[numPoints];
	memset(logP, 0, numPoints * sizeof(double));
	
	parallelLogMVNormDist(
		numPoints, pointDim,
		X, mu, sigmaL, logNormalizer,
		logP
	);

	double normalizer = sqrt(2.0 * M_PI);
	for(size_t i = 0; i < numPoints; ++i) {
		double x = X[i];
		double actual = exp(logP[i]);
		double expected = exp(-0.5 * x * x) / normalizer;

		double absDiff = abs(expected - actual);
		if(absDiff >= DBL_EPSILON) {
			printf("f(%.7f) = %.7f, but should equal = %.7f; absDiff = %.15f\n", 
				x, actual, expected, absDiff);
		}

		assert(absDiff < DBL_EPSILON);
	}
}

int main(int argc, char** argv) {
	test1DStandardNormal();

	printf("PASS %s\n", argv[0]);
	return EXIT_SUCCESS;
}
