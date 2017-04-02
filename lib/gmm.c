#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gmm.h"
#include "util.h"

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

void calcLogMvNorm(
	const struct Component* components, const size_t numComponents,
	const size_t componentStart, const size_t componentEnd,
	const double* X, const size_t numPoints, const size_t pointDim,
	double* logProb
) {
	assert(components != NULL);
	assert(numComponents > 0);
	assert(componentStart < componentEnd);
	assert(componentEnd > 0);
	assert(X != NULL);
	assert(numComponents > 0);
	assert(pointDim > 0);
	assert(logProb != NULL);

	for (size_t k = componentStart; k < componentEnd; ++k) {
		logMvNormDist(
			& components[k], pointDim,
			X, numPoints, 
			& logProb[k * numPoints]
		);
	}
}

double logLikelihood(
	const double* logpi, const size_t numComponents,
	const double* logProb, const size_t numPoints
) { 
	assert(logpi != NULL);
	assert(numComponents > 0);
	assert(logProb != NULL);
	assert(numPoints > 0);

	double logL = 0.0;
	for (size_t point = 0; point < numPoints; ++point) {
		double maxArg = -INFINITY;
		for(size_t k = 0; k < numComponents; ++k) {
			const double logProbK = logpi[k] + logProb[k * numPoints + point];
			if(logProbK > maxArg) {
				maxArg = logProbK;
			}
		}

		double sum = 0.0;
		for (size_t k = 0; k < numComponents; ++k) {
			const double logProbK = logpi[k] + logProb[k * numPoints + point];
			sum = exp(logProbK - maxArg);
		}

		assert(sum >= 0);
		logL += maxArg + log(sum);
	}

	return logL;
}

int shouldContinue(
	const size_t maxIterations, 
	const double prevLogL, const double currentLogL,
	const double tolerance
) {
	if(maxIterations == 0) {
		return 0;
	}

	if(maxIterations >= 80) {
		// assumes maxIterations = 100, so we always do atleast 20 iterations
		return 1;
	}

	if(fabs(currentLogL - prevLogL) < tolerance ) {
		return 0;
	}

	return 1;
}

void calcLogGammaNK(
	const double* logpi, const size_t numComponents,
	const size_t pointStart, const size_t pointEnd,
	double* loggamma, const size_t numPoints
) {
	assert(logpi != NULL);
	assert(numComponents > 0);

	assert(numPoints > 0);
	assert(pointStart < pointEnd);
	assert(pointEnd <= numPoints);

	assert(loggamma != NULL);

	for (size_t point = pointStart; point < pointEnd; ++point) {
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
}

void calcLogGammaK(
	const double* loggamma, const size_t numPoints,
	const size_t componentStart, const size_t componentEnd,
	double* logGamma, const size_t numComponents
) {
	assert(loggamma != NULL);
	assert(numPoints > 0);

	assert(componentStart < componentEnd);
	assert(componentEnd <= numComponents);
	assert(numComponents > 0);

	assert(logGamma != NULL);

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
}


double calcLogGammaSum(
	const double* logpi, const size_t numComponents,
	const double* logGamma
) {
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

	return maxArg + log(sum);
}

void performMStep(
	struct Component* components, const size_t numComponents,
	const size_t componentStart, const size_t componentEnd,
	double* logpi, const double* loggamma, const double* logGamma, const double logGammaSum,
	const double* X, const size_t numPoints, const size_t pointDim,
	double* outerProduct, double* xm
) {
	assert(components != NULL);
	assert(numComponents > 0);
	assert(componentStart < componentEnd);
	assert(componentEnd <= numComponents);
	assert(logpi != NULL);
	assert(loggamma != NULL);
	assert(logGamma != NULL);
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(outerProduct != NULL);
	assert(xm != NULL);

	for(size_t k = componentStart; k < componentEnd; ++k) {
		struct Component* component = & components[k];

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
}
