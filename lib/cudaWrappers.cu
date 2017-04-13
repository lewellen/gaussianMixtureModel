#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// Intentionally not including header since it is meant for gcc consumption.
// #include "cudaWrappers.h"

#include "cudaCommon.hu"
#include "cudaFolds.hu"
#include "cudaGmm.hu"
#include "cudaMVNormal.hu"

extern "C" void gpuSum(size_t numPoints, const size_t pointDim, double* host_a, double* host_sum) {
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(host_a != NULL);
	assert(host_sum != NULL);

	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	// cudaArraySum is meant for powers of two 
	size_t M = largestPowTwoLessThanEq(numPoints);

	double cpuSum[pointDim];
	memset(cpuSum, 0, pointDim * sizeof(double));
	for(size_t i = M; i < numPoints; ++i) {
		for(size_t j = 0; j < pointDim; ++j) {
			cpuSum[j] += host_a[i * pointDim + j];
		}
	}

	numPoints = M;

	double *device_a = sendToGpu(numPoints * pointDim, host_a);

	// cudaArraySum is synchronous
	cudaArraySum(
		&deviceProp, numPoints, pointDim, device_a, host_sum
		);

	cudaFree(device_a);

	for(size_t i = 0; i < pointDim; ++i) {
		host_sum[i] += cpuSum[i];
	}
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
		// cudaArrayMax is meant for powers of two when N > 1024; 
		size_t M = largestPowTwoLessThanEq(N);
		for(size_t i = M; i < N; ++i) {
			if(host_a[i] > cpuMax) {
				cpuMax = host_a[i];
			}
		}
		N = M;
	}

	double *device_a = sendToGpu(N, host_a);

	// cudaArrayMax is synchronous
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

	dim3 grid, block;
	calcDim(numPoints, &deviceProp, &block, &grid);
	kernLogMVNormDist<<<grid, block>>>(
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
	const double* logpi, const double* logP
) {
	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	const size_t M = largestPowTwoLessThanEq(numPoints); 

	double* device_logpi = sendToGpu(numComponents, logpi);
	double* device_logP = sendToGpu(numComponents * M, logP);

	double logL = cudaGmmLogLikelihood(
		& deviceProp,
		numPoints, numComponents,
		M,
		logpi, logP,
		device_logpi, device_logP
	);

	cudaFree(device_logpi);
	cudaFree(device_logP);

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

	dim3 grid, block;
	calcDim(numPoints, &deviceProp, &block, &grid);
	kernCalcLogGammaNK<<<grid, block>>>(
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

		double sum = 0;
		gpuSum(numPoints, 1, working, & sum);
 		logGamma[k] = maxValue + log(sum );
	}
	free(working);
}

extern "C" double gpuPerformEStep(
	const size_t numPoints, const size_t pointDim, const size_t numComponents,
	const double* X, 
	const double* logpi, const double* Mu, const double* SigmaL,
	double* logP
) {
	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double* device_X = sendToGpu(numPoints * pointDim, X);

	double* device_logpi = sendToGpu(numComponents, logpi);
	double* device_Mu = sendToGpu(pointDim * numComponents, Mu);
	double* device_SigmaL = sendToGpu(pointDim * pointDim * numComponents, SigmaL);

	double* device_logP = mallocOnGpu(numPoints * numComponents);

	dim3 grid, block;
	calcDim(numPoints, &deviceProp, &block, &grid);
	for(size_t k = 0; k < numComponents; ++k) {
		kernLogMVNormDist<<<grid, block>>>(
			numPoints, pointDim,
			device_X, 
			& device_Mu[k * pointDim], 
			& device_SigmaL[k * pointDim * pointDim],
			& device_logP[k * numPoints]
		);
	}

	const size_t M = largestPowTwoLessThanEq(numPoints);

	double logL = cudaGmmLogLikelihood(
		& deviceProp,
		numPoints, numComponents,
		M,
		logpi, logP,
		device_logpi, device_logP
	);

	cudaFree(device_X);

	cudaFree(device_logpi);
	cudaFree(device_Mu);
	cudaFree(device_SigmaL);

	check(cudaMemcpy(logP, device_logP, 
		numPoints * numComponents * sizeof(double), cudaMemcpyDeviceToHost));

	cudaFree(device_logP);

	return logL;
}

extern "C" void gpuPerformMStep(
	const size_t numPoints, const size_t pointDim,
	const double* X, 
	double* loggamma, double logGammaK, double logGammaSum,
	double* logpik, double* mu, double* sigma
) {
	// X: pointDim x numPoints
	// loggamma: 1 x numPoints
	// logGamma: 1 x 1
	// logGammaSum: 1 x 1
	// logPi: 1 x 1
	// mu: 1 x pointDim
	// sigma: pointDim x pointDim

	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	const size_t M = largestPowTwoLessThanEq(numPoints);

	double* device_X = sendToGpu(M * pointDim, X);
	double* device_loggamma = sendToGpu(M, loggamma);

	// update pi_+1
	*logpik += logGammaK - logGammaSum;

	// calculate mu_+1
	cudaUpdateMu(
		& deviceProp,
		numPoints, pointDim,
		M,
		X, loggamma, logGammaK,
		device_X, device_loggamma,
		mu
	);

	cudaDeviceSynchronize();

	// Calculate sigma_+1
	cudaUpdateSigma(
		& deviceProp,
		numPoints, pointDim,
		M,
		X, loggamma, logGammaK,
		device_X, device_loggamma,
		mu, 
		sigma
	);

	cudaFree(device_X);
	cudaFree(device_loggamma);

	// doing the cholesky decomposition is caller (cpu) side for now
}
