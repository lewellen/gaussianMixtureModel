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

void prepareCovariance(
	struct Component* component, 
	const size_t pointDim
); 

void mvNormDist(
	const struct Component* component, const size_t pointDim, 
	const double* X, const size_t numPoints, 
	double* P
); 

#endif
