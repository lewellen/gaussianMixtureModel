#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// Intentionally not including header since it is meant for gcc consumption.
// #include "cudaWrappers.h"

#include "cudaCommon.hu"
#include "cudaMVNormal.hu"
#include "cudaGmm.hu"

extern "C" double gpuSum(size_t N, double* host_a) {
	assert(host_a != NULL);
	assert(N > 0);

	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double cpuSum = 0;
	if(N > 1024) {
		// cudaArraySum is meant for powers of two when N > 1024; 
		size_t M = 2;
		while(M < N) {
			M *= 2;
		}

		if(M > N) {
			M /= 2;
			for(size_t i = M; i < N; ++i) {
				cpuSum += host_a[i];
			}
			N = M;
		}
	}

	double *device_a = sendToGpu(N, host_a);

	// cudaArraySum is synchronous
	double sum = cpuSum + cudaArraySum(
		&deviceProp, N, device_a
		);

	cudaFree(device_a);

	return sum;
}

extern "C" double gpuMax(size_t N, double* host_a) {
	assert(host_a != NULL);
	assert(N > 0);

	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double cpuMax = -INFINITY;
	if(N > 1024) {
		// cudaArraySum is meant for powers of two when N > 1024; 
		size_t M = 2;
		while(M < N) {
			M *= 2;
		}

		if(M > N) {
			M /= 2;
			for(size_t i = M; i < N; ++i) {
				if(host_a[i] > cpuMax) {
					cpuMax = host_a[i];
				}
			}
			N = M;
		}
	}

	double *device_a = sendToGpu(N, host_a);

	// cudaArraySum is synchronous
	double gpuMax = cudaArrayMax(
		&deviceProp, N, device_a
		);

	cudaFree(device_a);

	if(cpuMax > gpuMax) {
		return cpuMax;
	}

	return gpuMax;
}

extern "C" void gpuLogMVNormDist(
	const size_t numPoints, const size_t pointDim,
	const double* X, const double* mu, const double* sigmaL,
	double* logP
) {
	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double* device_X = sendToGpu(numPoints * pointDim, X);
	double* device_mu = sendToGpu(pointDim, mu);
	double* device_sigmaL = sendToGpu(pointDim * pointDim, sigmaL);
	double* device_logP = mallocOnGpu(numPoints);

	// TODO: calcDim...
	kernLogMVNormDist<<<1, numPoints>>>(
		numPoints, pointDim,
		device_X, device_mu, device_sigmaL,
		device_logP
		);

	check(cudaMemcpy(logP, device_logP, numPoints * sizeof(double), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	cudaFree(device_X);
	cudaFree(device_mu);
	cudaFree(device_sigmaL);
	cudaFree(device_logP);
}

extern "C" double gpuGmmLogLikelihood(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, const double* logP
) {
	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double* device_logPi = sendToGpu(numComponents, logPi);
	double* device_logP = sendToGpu(numComponents * numPoints, logP);
	double* device_logL = mallocOnGpu(numPoints);

	// TODO: calcDim...
	kernGmmLogLikelihood<<<1, numPoints>>>(
		numPoints, numComponents,
		device_logPi, device_logP, device_logL
	);

	// cudaArraySum is synchronous
	double logL = cudaArraySum(&deviceProp, numPoints, device_logL);

	cudaFree(device_logPi);
	cudaFree(device_logP);
	cudaFree(device_logL);

	return logL;
}

extern "C" void gpuCalcLogGammaNK(
	const size_t numPoints, const size_t pointDim, const size_t numComponents,
	const double* logpi, double* loggamma
) { 
	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double* device_logpi = sendToGpu(numComponents, logpi);
	double* device_loggamma = sendToGpu(numComponents * numPoints, loggamma);

	// TODO: calcDim...
	kernCalcLogGammaNK<<<1, numPoints>>>(
		numPoints, pointDim, numComponents,
		device_logpi, device_loggamma
	);

	cudaMemcpy(loggamma, device_loggamma, numComponents * numPoints * sizeof(double), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaFree(device_logpi);
	cudaFree(device_loggamma);
}

extern "C" void gpuCalcLogGammaK(
	const size_t numPoints, const size_t numComponents,
	const double* loggamma, double* logGamma
) {
	// Gamma[k] = max + log sum exp(loggamma - max)

	double* working = (double*)malloc(numPoints * sizeof(double));
	for(size_t k = 0; k < numComponents; ++k) {
		// TODO: refactor to have a generic z = a + log sum exp(x - a)
		memcpy(working, & loggamma[k * numPoints], numPoints * sizeof(double));
		double maxValue = gpuMax(numPoints, working);

		memcpy(working, & loggamma[k * numPoints], numPoints * sizeof(double));
		for(size_t i = 0; i < numPoints; ++i) {
			working[i] = exp(working[i] - maxValue);
		}

		logGamma[k] = maxValue + log( gpuSum(numPoints, working) );
	}
	free(working);
}
