#ifndef CUDAWRAPPERS_H
#define CUDAWRAPPERS_H

#include <stdlib.h>

extern void gpuSum(
	size_t numPoints, size_t pointDim, 
	double* host_a, double* host_sum
);

extern double gpuMax(
	const size_t N, double* a
);

extern void gpuLogMVNormDist(
	const size_t numPoints, const size_t pointDim,
	const double* X, const double* mu, const double* sigmaL, const double logNormalizer,
	double* logP
);

extern double gpuGmmLogLikelihood(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, const double* logP
);

extern void gpuCalcLogGammaNK(
	const size_t numPoints, const size_t pointDim, const size_t numComponents,
	const double* logpi, double* loggamma
);

extern void gpuCalcLogGammaK(
	const size_t numPoints, const size_t numComponents,
	const double* loggamma, double* logGamma
);

extern void gpuPerformMStep(
	const size_t numPoints, const size_t pointDim,
	const double* X, 
	double* loggamma, double logGammaK, double logGammaSum,
	double* logpik, double* mu, double* sigma
);

#endif
