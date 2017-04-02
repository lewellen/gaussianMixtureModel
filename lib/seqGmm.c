#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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
		for (size_t k = 0; k < numComponents; ++k) {
			logMvNormDist(
				& gmm->components[k], gmm->pointDim,
				X, numPoints, 
				& loggamma[k * numPoints]
			);
		}
	
		// 2015-09-20 GEL Eliminated redundant mvNorm clac in logLikelihood by 
		// passing in precomputed gamma values. Also moved loop termination here
		// since likelihood determines termination. Result: 1.3x improvement in 
		// execution time.  (~8 ms to ~6 ms on oldFaithful.dat)
		prevLogL = currentLogL;
		currentLogL = logLikelihood(logpi, numComponents, loggamma, numPoints);
		if (--maxIterations == 0 || !(maxIterations < 80 ? currentLogL > prevLogL : 1 == 1)) {
			break;
		}

		// convert loggamma (really p(x_n|mu_k, Sigma_k)) into actual loggamma
		for (size_t point = 0; point < numPoints; ++point) {
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

		// Let Gamma[component] = \Sum_point gamma[component, point]
		memset(logGamma, 0, numComponents * sizeof(double));
		for(size_t k = 0; k < numComponents; ++k) {
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
		
		double logGammaSum = 0;
		{
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

			logGammaSum = maxArg + log(sum);
		}

		// --- M-Step ---
		for(size_t k = 0; k < numComponents; ++k) {
			struct Component* component = & gmm->components[k];
	
			// Update pi
			logpi[k] += logGamma[k] - logGammaSum;
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

	free(logpi);
	free(loggamma);
	free(logGamma);

	free(xm);
	free(outerProduct);

	return gmm;
}
