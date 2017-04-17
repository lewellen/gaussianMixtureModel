#ifndef CUDAWRAPPERS_H
#define CUDAWRAPPERS_H

#include <stdlib.h>

struct GmmEmGpuCtx;

extern void gpuGmmFit(
	const double* X,
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents,
	double* pi,
	double* Mu,
	double* Sigma,
	double* SigmaL,
	double* normalizers
);

// Wrappers for unit testing
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
	const double* logPi, double* logP
);

extern void gpuCalcLogGammaNK(
	const size_t numPoints, const size_t numComponents,
	const double* logpi, double* loggamma
);

extern void gpuCalcLogGammaK(
	const size_t numPoints, const size_t numComponents,
	const double* loggamma, double* logGamma
);

#endif
