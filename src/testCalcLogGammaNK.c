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

typedef void (*CalcLogGammaNKWrapper)(const size_t, const size_t, const double*, double*);

void test1DStandardNormalLogGammaNK(int gnkIncPik, CalcLogGammaNKWrapper target) {
	const size_t pointDim = 1;
	const size_t numPoints = 1024;
	const size_t numComponents = 1;

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

	struct Component phi;
	phi.mu = mu;
	phi.sigmaL = sigmaL;
	phi.normalizer = logNormalizer;
	logMvNormDist(&phi, pointDim, X, numPoints, logP);
	
	double logPi[numComponents];
	double uniformPi = 1.0 / (double)numComponents;
	for(size_t k = 0; k < numComponents; ++k) {
		logPi[k] = log(uniformPi);
	}

	double loggamma[numPoints];
	memcpy(loggamma, logP, numPoints * sizeof(double));

	target(numPoints, numComponents, logPi, loggamma);

	for(size_t i = 0; i < numPoints; ++i) {
		double sum = 0;
		for(size_t k = 0; k < numComponents; ++k) {
			sum += logPi[k] + logP[k * numPoints + i];
		}

		for(size_t k = 0; k < numComponents; ++k) {
			double actual = loggamma[k * numPoints + i];
			assert(actual != -INFINITY);
			assert(actual != INFINITY);
			assert(actual == actual);

			double expected = logP[k * numPoints + i] - sum;
			if(gnkIncPik) {
				expected += logPi[k];
			}

			double absDiff = fabs(expected - actual);
			if(absDiff >= DBL_EPSILON) {
				printf("gamma_{n = %zu, k = %zu} = %.16f, but should equal = %.16f; absDiff = %.16f\n", 
					i, k, actual, expected, absDiff);
			}

			assert(absDiff < DBL_EPSILON);
		}
	}
}

void cpuCalcLogGammaNKWrapper(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, double* logP
) {
	calcLogGammaNK(logPi, numComponents, 0, numPoints, logP, numPoints);
}

void gpuCalcLogGammaNKWrapper(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, double* logP
) {
	gpuCalcLogGammaNK(numPoints, numComponents, logPi, logP);
}

int main(int argc, char** argv) {
	test1DStandardNormalLogGammaNK(1, cpuCalcLogGammaNKWrapper);
	test1DStandardNormalLogGammaNK(1, gpuCalcLogGammaNKWrapper);

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}
