#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "barrier.h"
#include "gmm.h"
#include "seqGmm.h"
#include "parallelGmm.h"
#include "util.h"

void checkStopCriteria(void* untypedArgs) {
	struct SharedThreadStartArgs* sargs = (struct SharedThreadStartArgs*) untypedArgs;
	assert(sargs != NULL);

	sargs->prevLogL = sargs->currentLogL;
	sargs->currentLogL = logLikelihood(
		sargs->logpi, sargs->gmm->numComponents,
		sargs->loggamma, sargs->numPoints
	);

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

	const size_t numComponents = sargs->gmm->numComponents;
	const double* logpi = sargs->logpi;
	const double* logGamma = sargs->logGamma;

	double maxArg = -INFINITY;
	for(size_t k = 0; k < numComponents; ++k) {
		const double arg = logpi[k] + logGamma[k];
		if(arg > maxArg) {
			maxArg = arg;
		}
	}

	double sum = 0;
	for (size_t k = 0; k < numComponents; ++k) {
		const double arg = logpi[k] + logGamma[k];
		sum += exp(arg - maxArg);
	}
	assert(sum >= 0);

	sargs->logGammaSum = maxArg + log(sum);
}

void* parallelFitStart(void* untypedArgs) {
	struct ThreadStartArgs* args = (struct ThreadStartArgs*) untypedArgs;
	assert(args != NULL);

	struct SharedThreadStartArgs* sargs = args->shared;

	const size_t numComponents = sargs->gmm->numComponents;

	const double* X = sargs->X;
	const size_t numPoints = sargs->numPoints;
	const size_t pointDim = sargs->pointDim;

	double* logpi = sargs->logpi;
	double* loggamma = sargs->loggamma;
	double* logGamma = sargs->logGamma;

	double* xm = (double*)checkedCalloc(pointDim, sizeof(double));
	double* outerProduct = (double*)checkedCalloc(pointDim * pointDim, sizeof(double));

	do {
		// --- E-Step ---

		// Compute gamma
		for(size_t k = args->componentStart; k < args->componentEnd; ++k) {
			logMvNormDist(
				& sargs->gmm->components[k], pointDim,
				X, numPoints, 
				& loggamma[k * numPoints]
			);
		}

		arriveAt(sargs->barrier, sargs, checkStopCriteria);
		if(!sargs->shouldContinue) {
			break;
		}

		// Parallelism here is ugly since it's spread across points instead of 
		// components like everything else. Ugly cache behavior on 
		// gamm_{n, k} /= sum.
		for (size_t point = args->pointStart; point < args->pointEnd; ++point) {
			double maxArg = -INFINITY;
			for (size_t k = 0; k < numComponents; ++k) {
				const double arg = logpi[k] + loggamma[k * numPoints + point];
				if(arg > maxArg) {
					maxArg = arg;
				}
			}

			// compute log p(x)
			double sum = 0;
			for(size_t k = 0; k < numComponents; ++k) {
				const double arg = logpi[k] + loggamma[k * numPoints + point];
				sum += exp(arg - maxArg);
			}
			assert(sum >= 0);

			const double logpx = maxArg + log(sum);
			for(size_t k = 0; k < numComponents; ++k) {
				loggamma[k * numPoints + point] += -logpx;
			}
		}

		arriveAt(sargs->barrier, NULL, NULL);

		// Afterwards, everybody does
		for(size_t k = args->componentStart; k < args->componentEnd; ++k) {
			double maxArg = -INFINITY;
			for(size_t point = 0; point < numPoints; ++point) {
				const double loggammank = loggamma[k * numPoints + point];
				if(loggammank > maxArg) {
					maxArg = loggammank;
				}
			}

			double sum = 0;
			for(size_t point = 0; point < numPoints; ++point) {
				const double loggammank = loggamma[k * numPoints + point];
				sum += exp(loggammank - maxArg);
			}
			assert(sum >= 0);

			logGamma[k] = maxArg + log(sum);
		}

		// Using local sum to avoid multicore cache coherence overhead until we have to

		arriveAt(sargs->barrier, sargs, computeGammaSum);

		// --- M-Step ---
		for(size_t k = args->componentStart; k < args->componentEnd; ++k) {
			struct Component* component = & sargs->gmm->components[k];

			// Update pi
			logpi[k] += logGamma[k] - sargs->logGammaSum;
			component->pi = exp(logpi[k]);
			assert(0 <= component->pi && component->pi <= 1);
	
			// Update mu
			memset(component->mu, 0, pointDim * sizeof(double));
			for (size_t point = 0; point < numPoints; ++point) {
				for (size_t dim = 0; dim < pointDim; ++dim) {
					component->mu[dim] += exp(loggamma[k * numPoints + point]) * X[point * pointDim + dim];
				}
			}

			for (size_t i = 0; i < pointDim; ++i) {
				component->mu[i] /= exp(logGamma[k]);
			}

			// Update sigma
			memset(component->sigma, 0, pointDim * pointDim * sizeof(double));
			for (size_t point = 0; point < numPoints; ++point) {
				// (x - m)
				for (size_t dim = 0; dim < pointDim; ++dim) {
					xm[dim] = X[point * pointDim + dim] - component->mu[dim];
				}

				// (x - m) (x - m)^T
				for (size_t row = 0; row < pointDim; ++row) {
					for (size_t column = 0; column < pointDim; ++column) {
						outerProduct[row * pointDim + column] = xm[row] * xm[column];
					}
				}

				for (size_t i = 0; i < pointDim * pointDim; ++i) {
					component->sigma[i] += exp(loggamma[k * numPoints + point]) * outerProduct[i];
				}
			}

			for (size_t i = 0; i < pointDim * pointDim; ++i) {
				component->sigma[i] /= exp(logGamma[k]);
			}
		
			prepareCovariance(component, pointDim);
		}

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

	struct GMM* gmm = initGMM(X, numPoints, pointDim, numComponents);

	size_t numProcessors = 8;
	if(numComponents < numProcessors) {
		numProcessors = numComponents;
	}

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
	stsa.logpi = (double*)checkedCalloc(numComponents, sizeof(double));
	stsa.loggamma = (double*)checkedCalloc(numComponents * numPoints, sizeof(double));
	stsa.logGamma = (double*)checkedCalloc(numComponents, sizeof(double));
	stsa.logGammaSum = 0.0;
	stsa.barrier = &barrier;

	for(size_t k = 0; k < numComponents; ++k) {
		const double pik = gmm->components[k].pi;
		assert(pik >= 0);
		stsa.logpi[k] = log(pik);
	}

	size_t pointResidual = numPoints % numProcessors;
	size_t pointsPerProcessor = (numPoints - pointResidual) / numProcessors;

	size_t componentResidual = numComponents % numProcessors;
	size_t componentsPerProcessor = (numComponents - componentResidual) / numProcessors;

	struct ThreadStartArgs args[numProcessors];
	for(size_t i = 0; i < numProcessors; ++i) {
		args[i].id = i;	
		args[i].shared = &stsa;

		if(i == 0) {
			args[i].pointStart = 0;
			args[i].componentStart = 0;
		} else {
			args[i].pointStart = args[i - 1].pointEnd;
			args[i].componentStart = args[i - 1].componentEnd;
		}

		args[i].pointEnd = args[i].pointStart + pointsPerProcessor;
		if(pointResidual > 0) {
			--pointResidual;
			++args[i].pointEnd;
		}

		assert(args[i].pointEnd <= numPoints);

		args[i].componentEnd = args[i].componentStart + componentsPerProcessor;
		if(componentResidual > 0) {
			--componentResidual;
			++args[i].componentEnd;
		}
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

	free(stsa.logpi);
	free(stsa.loggamma);
	free(stsa.logGamma);

	return gmm;
}
