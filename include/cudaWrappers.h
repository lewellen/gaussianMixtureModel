#ifndef CUDAWRAPPERS_H
#define CUDAWRAPPERS_H

#include <stdlib.h>

extern void parallelGmmLogLikelihood(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, double* logProb
);

extern void parallelLogMVNormDist(
	const size_t numPoints, const size_t pointDim,
	const double* X, const double* mu, const double* sigmaL, const double logNormalizer,
	double* logP
);

extern double parallelSum(
	double* host_a, const size_t N
);

#endif
