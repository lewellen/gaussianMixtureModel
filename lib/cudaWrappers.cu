#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// Intentionally not including header since it is meant for gcc consumption.
// #include "cudaWrappers.h"

#include "cudaCommon.hu"
#include "cudaMVNormal.hu"
#include "cudaGmm.hu"

double* mallocOnGpu(const size_t N) {
	double* device_A;
	double ABytes = N * sizeof(double);
	check(cudaMalloc(&device_A, ABytes));
	return device_A;
}

double* sendToGpu(const size_t N, const double* A) {
	double* device_A;
	const size_t ABytes = N * sizeof(double);
	check(cudaMalloc(&device_A, ABytes));
	check(cudaMemcpy(device_A, A, ABytes, cudaMemcpyHostToDevice));
	return device_A;
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

	kernLogMVNormDist<<<numPoints, 1>>>(
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

	// TODO: Power of two padding?
	double* device_logPi = sendToGpu(numComponents, logPi);
	double* device_logP = sendToGpu(numComponents, logP);
	double* device_logL = mallocOnGpu(numPoints);

	// TODO: calcDim...
	kernGmmLogLikelihood<<<numPoints, 1>>>(
		numPoints, numComponents,
		device_logPi, device_logP, device_logL
	);

	double logL = cudaReduceSum(&deviceProp, numPoints, device_logL);

	cudaFree(device_logPi);
	cudaFree(device_logP);
	cudaFree(device_logL);

	return logL;
}

extern "C" double gpuSum(const size_t N, double* host_a) {
	assert(host_a != NULL);
	assert(N > 0);
	assertPowerOfTwo(N);

	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double *device_a = sendToGpu(N, host_a);

	double sum = cudaReduceSum(
		&deviceProp, N, device_a
		);

	cudaFree(device_a);

	return sum;
}

