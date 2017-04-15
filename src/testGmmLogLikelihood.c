#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif

#include "component.h"
#include "gmm.h"
#include "cudaWrappers.h"

typedef double (*GmmLogLikelihoodWrapper)(const size_t, const size_t, const double*, double*);

void test1DStandardNormalLogLikelihood(GmmLogLikelihoodWrapper target) {
	const size_t numPoints = 1024 + 512;
	const size_t numComponents = 2;
	const size_t pointDim = 1;

	double sigma = 1;
	double det = 1;


	double logNormalizer = -0.5 * pointDim * log(2.0 * M_PI) - 0.5 * log(det);

	double mu0 = -1.5;
	double mu1 = +1.5;


	double X[numPoints];
	memset(X, 0, numPoints * sizeof(double));
	for(size_t i = 0; i < numPoints; ++i) {
		X[i] = 3.0 * ( ( (double)i - (double)numPoints/2 ) / (double)(numPoints/2.0) );
	}

	double logP[numComponents * numPoints];
	memset(logP, 0, numComponents * numPoints * sizeof(double));

	struct Component phi;
	phi.sigmaL = &sigma;
	phi.normalizer = logNormalizer;

	phi.mu = &mu0;
	logMvNormDist(&phi, 1, X, numPoints, logP);

	phi.mu = &mu1;
	logMvNormDist(&phi, 1, X, numPoints, &logP[numPoints]);

	double logPi[numComponents];
	logPi[0] = log(0.25);
	logPi[1] = log(0.75);

	double actual = target(numPoints, numComponents, logPi, logP);
	assert(actual != -INFINITY);
	assert(actual != INFINITY);
	assert(actual == actual);

	// gpu impl overwrites logP with loggamma (logP - log p(x)), just run again
	phi.mu = &mu0;
	logMvNormDist(&phi, 1, X, numPoints, logP);

	phi.mu = &mu1;
	logMvNormDist(&phi, 1, X, numPoints, &logP[numPoints]);

	double expected = 0;
	for(size_t i = 0; i < numPoints; ++i) {
		double maxValue = -INFINITY;
		for(size_t k = 0; k < numComponents; ++k) {
			const double value = logPi[k] + logP[k * numPoints + i];
			if(maxValue < value) {
				maxValue = value;
			}
		}

		double sum = 0;
		for(size_t k = 0; k < numComponents; ++k) {
			const double value = logPi[k] + logP[k * numPoints + i];
			sum += exp(value - maxValue);
		}

		expected += maxValue + log(sum);
	}

	double absDiff = fabs(expected - actual);
	if(absDiff >= DBL_EPSILON) {
		printf("log L = %.16f, but should equal = %.16f; absDiff = %.16f\n", 
			actual, expected, absDiff);
	}

	assert(absDiff < DBL_EPSILON);
}

double cpuGmmLogLikelihoodWrapper(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, double* logP
) {
	double logL = 0;
	logLikelihood(logPi, numComponents, logP, numPoints, 0, numPoints, &logL);
	return logL;
}

double gpuGmmLogLikelihoodWrapper(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, double* logP
) {
	return gpuGmmLogLikelihood(numPoints, numComponents, logPi, logP);
}

int main(int argc, char** argv) {
	test1DStandardNormalLogLikelihood(cpuGmmLogLikelihoodWrapper);
	test1DStandardNormalLogLikelihood(gpuGmmLogLikelihoodWrapper);

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}
