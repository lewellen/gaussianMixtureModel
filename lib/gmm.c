#include <assert.h>
#include <math.h>
#include <stdlib.h>

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

