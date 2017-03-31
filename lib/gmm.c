#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gmm.h"
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

	double* gamma = (double*)checkedCalloc(numPoints * numComponents, sizeof(double));
	double* Gamma = (double*)checkedCalloc(numComponents, sizeof(double));

	double* xm = (double*)checkedCalloc(pointDim, sizeof(double));
	double* outerProduct = (double*)checkedCalloc(pointDim * pointDim, sizeof(double));

	do {
		// --- E-Step ---

		// Compute gamma
		for (size_t k = 0; k < numComponents; ++k) {
			mvNormDist(
				& gmm->components[k], gmm->pointDim, 
				X, numPoints, 
				& gamma[k * numPoints]
			);
		}
	
		// 2015-09-20 GEL Eliminated redundant mvNorm clac in logLikelihood by 
		// passing in precomputed gamma values. Also moved loop termination here
		// since likelihood determines termination. Result: 1.3x improvement in 
		// execution time.  (~8 ms to ~6 ms on oldFaithful.dat)
		prevLogL = currentLogL;
		currentLogL = logLikelihood(gmm, gamma, numPoints);
		if (--maxIterations == 0 || !(maxIterations < 80 ? currentLogL > prevLogL : 1 == 1)) {
			break;
		}

		for (size_t point = 0; point < numPoints; ++point) {
			double sum = 0.0;
			for (size_t k = 0; k < numComponents; ++k) {
				sum += gmm->components[k].pi * gamma[k * numPoints + point];
			}

			if (sum > tolerance) {
				for (size_t k = 0; k < numComponents; ++k) {
					gamma[k * numPoints + point] /= sum;
				}
			}
		}

		// Let Gamma[component] = \Sum_point gamma[component, point]
		memset(Gamma, 0, numComponents * sizeof(double));
		for (size_t k = 0; k < numComponents; ++k) {
			for (size_t point = 0; point < numPoints; ++point) {
				Gamma[k] += gamma[k * numPoints + point];
			}
		}

		double GammaSum = 0;
		for (size_t k = 0; k < numComponents; ++k) {
			GammaSum += gmm->components[k].pi * Gamma[k];
		}

		// --- M-Step ---
		for(size_t k = 0; k < numComponents; ++k) {
			struct Component* component = & gmm->components[k];
	
			// Update pi
			component->pi *= Gamma[k] / GammaSum;
			
			// Update mu
			memset(component->mu, 0, pointDim * sizeof(double));
			for (size_t point = 0; point < numPoints; ++point) {
				for (size_t dim = 0; dim < pointDim; ++dim) {
					component->mu[dim] += gamma[k * numPoints + point] * X[point * pointDim + dim];
				}
			}

			for (size_t i = 0; i < pointDim; ++i) {
				component->mu[i] /= Gamma[k];
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
					component->sigma[i] += gamma[k * numPoints + point] * outerProduct[i];
				}
			}

			for (size_t i = 0; i < pointDim * pointDim; ++i) {
				component->sigma[i] /= Gamma[k];
			}
		
			prepareCovariance(component, pointDim);
		}

	} while (1 == 1);

	free(gamma);
	free(Gamma);

	free(xm);
	free(outerProduct);

	return gmm;
}

struct GMM* initGMM(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents
) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(numComponents > 0);

	// X is an numPoints x pointDim set of training data

	struct GMM* gmm = (struct GMM*)checkedCalloc(1, sizeof(struct GMM));
	gmm->pointDim = pointDim;
	gmm->numComponents = numComponents;
	gmm->components = (struct Component*) checkedCalloc(numComponents, sizeof(struct Component));

	double uniformTau = 1.0 / numComponents;
	for(size_t k = 0; k < gmm->numComponents; ++k) {
		struct Component* component = & gmm->components[k];

		// Assume every component has uniform weight
		component->pi = uniformTau;

		// Use a random point for mean of each component
		component->mu = (double*)checkedCalloc(pointDim, sizeof(double));
		size_t j = rand() % numPoints;
		for(size_t dim = 0; dim < gmm->pointDim; dim++) {
			component->mu[dim] = X[j * gmm->pointDim + dim];
		}

		// Use identity covariance- assume dimensions are independent
		component->sigma = (double*)checkedCalloc(pointDim * pointDim, sizeof(double));
		for (size_t dim = 0; dim < pointDim; ++dim)
			component->sigma[dim * pointDim + dim] = 1;
		
		// Initialize zero artifacts
		component->sigmaL = (double*)checkedCalloc(pointDim * pointDim, sizeof(double));
		component->normalizer = 0;
	
		prepareCovariance(component, pointDim);
	}


	return gmm;
}

void freeGMM(struct GMM* gmm) {
	assert(gmm != NULL);

	for(size_t k = 0; k < gmm->numComponents; ++k) {
		struct Component* component = & gmm->components[k];
		free(component->mu);
		free(component->sigma);
		free(component->sigmaL);
	}

	free(gmm->components);
	free(gmm);
}

double logLikelihood(
	const struct GMM* gmm,
	const double* prob, const size_t numPoints
) { 
	assert(gmm != NULL);
	assert(prob != NULL);
	assert(numPoints > 0);

	double logL = 0.0;
	for (size_t point = 0; point < numPoints; ++point) {
		double inner = 0.0;
		for (size_t component = 0; component < gmm->numComponents; ++component)
			inner += gmm->components[component].pi * prob[component * numPoints + point];

		logL += log(inner);
	}

	return logL;
}

