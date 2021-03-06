#ifndef COMPONENT_H
#define COMPONENT_H

#include <stdlib.h>

struct Component {
	// Parameters, weight, mean, covariance
	double pi;
	double* mu;
	double* sigma;

	// Lower triangular covariance matrix
	double* sigmaL;

	// Probability density normalizer
	double normalizer;
};

void printToConsole(
	const struct Component* component,
	const size_t pointDim
);

void prepareCovariance(
	struct Component* component, 
	const size_t pointDim
); 

void logMvNormDist(
	const struct Component* component, const size_t pointDim, 
	const double* X, const size_t numPoints, 
	double* logProb
); 

double sampleStandardNormal();

double* sampleWishart(
	const size_t dimension, const size_t degreeFreedom
);

double* sampleWishartCholesky(
	const size_t dimension, const size_t degreeFreedom
);

#endif
