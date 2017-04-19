#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "gmm.h"
#include "seqGmm.h"
#include "util.h"

struct GMM* fit(
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
	size_t iteration = 0;
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

	do {
		// --- E-Step ---

		// Compute gamma
		calcLogMvNorm(
			gmm->components, numComponents, 
			0, numComponents, 
			X, numPoints, pointDim,
			loggamma
		);
	
		// 2015-09-20 GEL Eliminated redundant mvNorm clac in logLikelihood by 
		// passing in precomputed gamma values. Also moved loop termination here
		// since likelihood determines termination. Result: 1.3x improvement in 
		// execution time.  (~8 ms to ~6 ms on oldFaithful.dat)
		// 2017-04-14 GEL Decided to fuse logLikelihood and Gamma NK calculation
		// since they both rely on the log p(x) calculation, and it would be 
		// wasteful to compute and store p(x), since log L and gamma NK are only
		// consumers of that data.
		prevLogL = currentLogL;
		logLikelihoodAndGammaNK(
			logpi, numComponents, 
			loggamma, numPoints,
			0, numPoints,
			& currentLogL
		);

		assert(maxIterations > 0);
		--maxIterations;
		if(!shouldContinue(prevLogL, currentLogL, tolerance)) {
			break;
		}

		// Let Gamma[component] = \Sum_point gamma[component, point]
		calcLogGammaK(
			loggamma, numPoints, 
			0, numComponents, 
			logGamma, numComponents
		);
	
		double logGammaSum = calcLogGammaSum(logpi, numComponents, logGamma);

		// --- M-Step ---
		performMStep(
			gmm->components, numComponents,
			0, numComponents,
			logpi, loggamma, logGamma, logGammaSum,
			X, numPoints, pointDim,
			outerProduct, xm
		);

	} while (++iteration < maxIterations);

	free(logpi);
	free(loggamma);
	free(logGamma);

	free(xm);
	free(outerProduct);

	return gmm;
}
