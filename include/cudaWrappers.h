#ifndef CUDAWRAPPERS_H
#define CUDAWRAPPERS_H

#include <stdlib.h>

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

extern double gpuSum(
	const size_t N, double* a
);

#endif
