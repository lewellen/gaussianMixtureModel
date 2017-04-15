#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>

#include "gmm.h"
#include "cudaGmm.h"
#include "util.h"

#include "cudaWrappers.h"

struct GMM* cudaFit(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents
) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(numComponents > 0);
	
	struct GMM* gmm = initGMM(X, numPoints, pointDim, numComponents);

	const double tolerance = 1e-8;
	size_t maxIterations = 100;
	double prevLogL = -INFINITY;
	double currentLogL = -INFINITY;

	double* logpi = (double*)checkedCalloc(numComponents, sizeof(double));
	double* loggamma = (double*)checkedCalloc(numPoints * numComponents, sizeof(double));
	double* logGamma = (double*)checkedCalloc(numComponents, sizeof(double));

	double* xm = (double*)checkedCalloc(pointDim, sizeof(double));
	double* outerProduct = (double*)checkedCalloc(pointDim * pointDim, sizeof(double));

	for(size_t k = 0; k < numComponents; ++k) {
		const double pik = gmm->components[k].pi;
		assert(pik >= 0);
		logpi[k] = log(pik);
	}

	double* Mu = (double*) checkedCalloc(pointDim * numComponents, sizeof(double));
	double* SigmaL = (double*) checkedCalloc(pointDim * pointDim * numComponents, sizeof(double));

	do {
		// --- E-Step ---

		for(size_t k = 0; k < numComponents; ++k) {
			struct Component* C = & gmm->components[k];
			memcpy(&Mu[k * pointDim], C->mu, pointDim * sizeof(double));
			memcpy(&SigmaL[k * pointDim * pointDim], C->sigmaL, pointDim * pointDim * sizeof(double));
		}

		prevLogL = currentLogL;
		currentLogL = gpuPerformEStep(
			numPoints, pointDim, numComponents,
			X,
			logpi, Mu, SigmaL,
			loggamma
		);

		printf("%.16f %.16f\n", prevLogL, currentLogL);
		
		assert(maxIterations > 0);
		--maxIterations;
		if(!shouldContinue(maxIterations, prevLogL, currentLogL, tolerance)) {
			break;
		}

		// Let Gamma[component] = \Sum_point gamma[component, point]
		gpuCalcLogGammaK(
			numPoints, numComponents,
			loggamma, logGamma
		);

		// Not worth running on gpu since k should be smallish	
		double logGammaSum = calcLogGammaSum(logpi, numComponents, logGamma);

		// --- M-Step ---
		for(size_t k = 0; k < numComponents; ++k) {
			gpuPerformMStep(
				numPoints, pointDim,
				X, 
				& loggamma[k * numPoints], logGamma[k], logGammaSum,
				& logpi[k], gmm->components[k].mu, gmm->components[k].sigma
			);

			gmm->components[k].pi = exp(logpi[k]);

			prepareCovariance(& gmm->components[k], pointDim);
		}
	} while (1 == 1);

	free(Mu);
	free(SigmaL);

	free(logpi);
	free(loggamma);
	free(logGamma);

	free(xm);
	free(outerProduct);

	return gmm;
}
