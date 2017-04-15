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

	const double sigma = 1;
	const double det = 1;
	const double logNormalizer = -0.5 * pointDim * log(2.0 * M_PI) - 0.5 * log(det);

	const double logPi[] = { log(0.25), log(0.75) };
	const double mu[] = { -1.5, 1.5 };

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
	for(size_t k = 0; k < numComponents; ++k) {
		phi.mu = &mu[k];
		logMvNormDist(&phi, 1, X, numPoints, & logP[k * numPoints]);
	}

	double actualLogL = target(numPoints, numComponents, logPi, logP);

	// Verify the logL portion
	{
		assert(actualLogL != -INFINITY);
		assert(actualLogL != INFINITY);
		assert(actualLogL == actualLogL);

		double expectedLogL = 0;
		for(size_t i = 0; i < numPoints; ++i) {
			double sum = 0;
			for(size_t k = 0; k < numComponents; ++k) {
				sum += exp(logPi[k] + logNormalizer - 0.5 * pow( X[i] - mu[k], 2.0 )); 
			}

			expectedLogL += log(sum);
		}

		double absDiff = fabs(expectedLogL - actualLogL);
		if(absDiff >= DBL_EPSILON) {
			printf("log L = %.16f, but should equal = %.16f; absDiff = %.16f\n", 
				actualLogL, expectedLogL, absDiff);
		}

		assert(absDiff < DBL_EPSILON);
	}

	// Verify the gammaNK portion
	{
		for(size_t i = 0; i < numPoints; ++i) { 
			double sum = 0;
			for(size_t k = 0; k < numComponents; ++k) {
				sum += exp(logPi[k] + logNormalizer - 0.5 * pow( X[i] - mu[k], 2.0 )); 
			}
			double logPx = log(sum);

			for(size_t k = 0; k < numComponents; ++k) {
				double expectedGammaNK = logNormalizer - 0.5 * pow(X[i] - mu[k], 2.0) - logPx;
				double actualGammaNK = logP[k * numPoints + i];

				double absDiff = fabs(expectedGammaNK - actualGammaNK);
				if(absDiff >= 10.0 * DBL_EPSILON) {
					printf("gamma_{n = %zu, k = %zu} = %.16f, but should equal = %.16f; absDiff = %.16f, epsilon = %.16f\n", 
						i, k, actualGammaNK, expectedGammaNK, absDiff, 10.0 * DBL_EPSILON);
				}

				assert(absDiff < 10.0 * DBL_EPSILON);
			}
		}
	}
}

double cpuGmmLogLikelihoodWrapper(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, double* logP
) {
	double logL = 0;
	logLikelihoodAndGammaNK(logPi, numComponents, logP, numPoints, 0, numPoints, &logL);
	return logL;
}

double gpuGmmLogLikelihoodWrapper(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, double* logP
) {
	// does both loglikelihood and gamma nk
	return gpuGmmLogLikelihood(numPoints, numComponents, logPi, logP);
}

int main(int argc, char** argv) {
	test1DStandardNormalLogLikelihood(cpuGmmLogLikelihoodWrapper);
	test1DStandardNormalLogLikelihood(gpuGmmLogLikelihoodWrapper);

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}
