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

void calcLogMvNorm(
	const struct Component* components, const size_t numComponents,
	const size_t componentStart, const size_t componentEnd,
	const double* X, const size_t numPoints, const size_t pointDim,
	double* logProb
);

double logLikelihood(
	const double* logpi, const size_t numComponents,
	const double* logProb, const size_t numPoints
);

int shouldContinue(
	const size_t maxIterations, 
	const double prevLogL, const double currentLogL,
	const double tolerance
);

void calcLogGammaNK(
	const double* logpi, const size_t numComponents,
	const size_t pointStart, const size_t pointEnd,
	double* loggamma, const size_t numPoints
);

void calcLogGammaK(
	const double* loggamma, const size_t numPoints,
	const size_t componentStart, const size_t componentEnd,
	double* logGamma, const size_t numComponents
);

double calcLogGammaSum(
	const double* logpi, const size_t numComponents,
	const double* logGamma
);

void performMStep(
	struct Component* components, const size_t numComponents,
	const size_t componentStart, const size_t componentEnd,
	double* logpi, const double* loggamma, const double* logGamma, const double logGammaSum,
	const double* X, const size_t numPoints, const size_t pointDim,
	double* outerProduct, double* xm
);

#endif
