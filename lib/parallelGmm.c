#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "barrier.h"
#include "gmm.h"
#include "seqGmm.h"
#include "parallelGmm.h"
#include "util.h"

struct SharedThreadStartArgs {
	const double* X;
	size_t numPoints;
	size_t pointDim;

	struct GMM* gmm;

	double tolerance;
	size_t maxIterations;
	double prevLogL;
	double currentLogL;

	int shouldContinue;

	double* gamma;
	double* Gamma;
	double GammaSum;

	struct Barrier* barrier;
};

struct ThreadStartArgs {
	size_t id;

	struct SharedThreadStartArgs* shared;		
	struct Component* component;
	double* gammaComponent;

	size_t pointStart;
	size_t pointEnd; // i = pointStart; i < pointEnd; ++i
};

void checkStopCriteria(void* untypedArgs) {
	struct SharedThreadStartArgs* sargs = (struct SharedThreadStartArgs*) untypedArgs;
	assert(sargs != NULL);

	sargs->prevLogL = sargs->currentLogL;
	sargs->currentLogL = logLikelihood(sargs->gmm, sargs->gamma, sargs->numPoints);

	assert(sargs->maxIterations > 0);
	if(--sargs->maxIterations == 0) {
		sargs->shouldContinue = 0;
	} else if(sargs->maxIterations >= 80) {
		// assumes maxIterations = 100, so we always do atleast 20 iterations
		sargs->shouldContinue = 1;
	} else if(fabs(sargs->currentLogL - sargs->prevLogL) < sargs->tolerance ) {
		sargs->shouldContinue = 0;
	} else {
		sargs->shouldContinue = 1;
	}
}

void computeGammaSum(void* untypedArgs) {
	struct SharedThreadStartArgs* sargs = (struct SharedThreadStartArgs*) untypedArgs;
	assert(sargs != NULL);

	sargs->GammaSum = 0;
	for (size_t k = 0; k < sargs->gmm->numComponents; ++k) {
		sargs->GammaSum += sargs->gmm->components[k].pi * sargs->Gamma[k];
	}
}

void* parallelFitStart(void* untypedArgs) {
	struct ThreadStartArgs* args = (struct ThreadStartArgs*) untypedArgs;
	assert(args != NULL);

	struct SharedThreadStartArgs* sargs = args->shared;
	struct Component* component = args->component;
	const size_t numPoints = sargs->numPoints;
	const size_t pointDim = sargs->pointDim;

	double* xm = (double*)checkedCalloc(pointDim, sizeof(double));
	double* outerProduct = (double*)checkedCalloc(pointDim * pointDim, sizeof(double));

	do {
		// --- E-Step ---

		// Compute gamma
		mvNormDist(
			component, sargs->pointDim, 
			sargs->X, numPoints, 
			args->gammaComponent
		);

		arriveAt(sargs->barrier, sargs, checkStopCriteria);
		if(!sargs->shouldContinue) {
			break;
		}

		// Parallelism here is ugly since it's spread across points instead of 
		// components like everything else. Ugly cache behavior on 
		// gamm_{n, k} /= sum.
		for (size_t point = args->pointStart; point < args->pointEnd; ++point) {
			double sum = 0.0;
			for (size_t k = 0; k < sargs->gmm->numComponents; ++k) {
				sum += sargs->gmm->components[k].pi * sargs->gamma[k * numPoints + point];
			}

			if (sum > sargs->tolerance) {
				for (size_t k = 0; k < sargs->gmm->numComponents; ++k) {
					sargs->gamma[k * numPoints + point] /= sum;
				}
			}
		}

		arriveAt(sargs->barrier, NULL, NULL);

		// Afterwards, everybody does
		double GammaK = 0;
		for (size_t point = 0; point < numPoints; ++point) {
			GammaK += args->gammaComponent[point];
		}

		// Using local sum to avoid multicore cache coherence overhead until we have to
		sargs->Gamma[ args->id ] = GammaK;

		arriveAt(sargs->barrier, sargs, computeGammaSum);	

		// --- M-Step ---
		// Update pi
		component->pi *= GammaK / sargs->GammaSum;
		
		// Update mu
		memset(component->mu, 0, pointDim * sizeof(double));
		for (size_t point = 0; point < numPoints; ++point) {
			for (size_t dim = 0; dim < pointDim; ++dim) {
				component->mu[dim] += args->gammaComponent[point] * sargs->X[point * pointDim + dim];
			}
		}

		for (size_t i = 0; i < pointDim; ++i) {
			component->mu[i] /= GammaK;
		}

		// Update sigma
		memset(component->sigma, 0, pointDim * pointDim * sizeof(double));
		for (size_t point = 0; point < numPoints; ++point) {
			// (x - m)
			for (size_t dim = 0; dim < pointDim; ++dim) {
				xm[dim] = sargs->X[point * pointDim + dim] - component->mu[dim];
			}

			// (x - m) (x - m)^T
			for (size_t row = 0; row < pointDim; ++row) {
				for (size_t column = 0; column < pointDim; ++column) {
					outerProduct[row * pointDim + column] = xm[row] * xm[column];
				}
			}

			for (size_t i = 0; i < pointDim * pointDim; ++i) {
				component->sigma[i] += args->gammaComponent[point] * outerProduct[i];
			}
		}

		for (size_t i = 0; i < pointDim * pointDim; ++i) {
			component->sigma[i] /= GammaK;
		}
	
		prepareCovariance(component, pointDim);

	} while (1 == 1);

	free(xm);
	free(outerProduct);

	return NULL;
}

struct GMM* parallelFit(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents
) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(numComponents > 0);

	if(numComponents == 1) {
		return fit(X, numPoints, pointDim, numComponents);
	}

	size_t numProcessors = numComponents;

	struct GMM* gmm = initGMM(X, numPoints, pointDim, numComponents);

	struct Barrier barrier;
	initBarrier(&barrier, numProcessors);

	struct SharedThreadStartArgs stsa;
	stsa.X = X;
	stsa.numPoints = numPoints;
	stsa.pointDim = pointDim;

	stsa.gmm = gmm;

	stsa.tolerance = 1e-8;
	stsa.maxIterations = 100;
	stsa.prevLogL = -INFINITY;
	stsa.currentLogL = -INFINITY;
	stsa.gamma = (double*)checkedCalloc(numComponents * numPoints, sizeof(double));
	stsa.Gamma = (double*)checkedCalloc(numComponents, sizeof(double));
	stsa.GammaSum = 0.0;
	stsa.barrier = &barrier;

	size_t residual = numPoints % numProcessors;
	size_t pointsPerProcessor = (numPoints - residual) / numProcessors;

	struct ThreadStartArgs args[numProcessors];
	for(size_t i = 0; i < numProcessors; ++i) {
		args[i].id = i;	
		args[i].shared = &stsa;
		args[i].component = & gmm->components[i];
		args[i].gammaComponent = & stsa.gamma[i * numPoints];

		if(i == 0) {
			args[i].pointStart = 0;
		} else {
			args[i].pointStart = args[i-1].pointEnd;
		}

		args[i].pointEnd = args[i].pointStart + pointsPerProcessor;
		if(residual > 0) {
			--residual;
			++args[i].pointEnd;
		}

		assert(args[i].pointEnd <= numPoints);
	}

	size_t numThreads = numProcessors - 1;
	pthread_t threads[numThreads];

	for(size_t i = 0; i < numThreads; ++i) {
		int result = pthread_create(&threads[i], NULL, parallelFitStart, &args[i]);
		if(result != 0) {
			break;
		}
	}

	parallelFitStart(&args[numThreads]);

	for(size_t i = 0; i < numThreads; ++i) {
		int result = pthread_join(threads[i], NULL);
		if(result != 0) {
			break;
		}
	}

	destroyBarrier(&barrier);

	free(stsa.gamma);
	free(stsa.Gamma);

	return gmm;
}
