#ifndef GMM_H
#define GMM_H

#include <stdlib.h>

#include "component.h"

struct GMM {
	// Dimension of the data (each data point is a vector \in R^{pointDim})
	size_t pointDim;

	// The individual components that constitute the model
	size_t numComponents;
	struct Component* components;
};

struct GMM* initGMM(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents
); 

void freeGMM(struct GMM* gmm);
 
double logLikelihood(
	const double* logpi, const size_t numComponents,
	const double* logProb, const size_t numPoints
);

#endif
