#include <assert.h>
#include <stdlib.h>

#include "cudaGmm.hu"

__global__ void kernGmmLogLikelihood(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, const double* logP,
	double* logL
) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	double maxArg = -INFINITY;
	for(size_t k = 0; k < numComponents; ++k) {
		const double logProbK = logPi[k] + logP[k * numPoints + i];
		if(logProbK > maxArg) {
			maxArg = logProbK;
		}
	}

	double sum = 0.0;
	for (size_t k = 0; k < numComponents; ++k) {
		const double logProbK = logPi[k] + logP[k * numPoints + i];
		sum = exp(logProbK - maxArg);
	}

	logL[i] = maxArg + log(sum);
}

__global__ void kernCalcLogGammaNK(
	const size_t numPoints, const size_t pointDim, const size_t numComponents,
	const double* logpi, double* loggamma
) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	double maxArg = -INFINITY;
	for (size_t k = 0; k < numComponents; ++k) {
		const double arg = logpi[k] + loggamma[k * numPoints + i];
		if(arg > maxArg) {
			maxArg = arg;
		}
	}

	// compute log p(x)
	double sum = 0;
	for(size_t k = 0; k < numComponents; ++k) {
		const double arg = logpi[k] + loggamma[k * numPoints + i];
		sum += exp(arg - maxArg);
	}

	const double logpx = maxArg + log(sum);
	for(size_t k = 0; k < numComponents; ++k) {
		loggamma[k * numPoints + i] += -logpx;
	}
}
