#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// Intentionally not including header since it is meant for gcc consumption.
// #include "cudaWrappers.h"
#include "cudaKernels.hu"

#define check(call) { cudaError_t __ABC123 = call; assert(__ABC123 == cudaSuccess); }

extern "C" void parallelGmmLogLikelihood(
	const size_t numPoints, const size_t numComponents,
	const double* logPi, double* logProb
) {
	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double* device_logPi;
	const size_t logPiBytes = numComponents * sizeof(double);
	check(cudaMalloc(&device_logPi, logPiBytes));
	check(cudaMemcpy(device_logPi, logPi, logPiBytes, cudaMemcpyHostToDevice));

	double* device_logProb;
	const size_t logProbBytes = numComponents * numPoints * sizeof(double);
	check(cudaMalloc(&device_logProb, logProbBytes));
	check(cudaMemcpy(device_logProb, logProb, logProbBytes, cudaMemcpyHostToDevice));

	cudaGmmLogLikelihood<<<numPoints, 1>>>(
		numPoints, numComponents, 
		device_logPi, device_logProb
	);

	// Not final action; need to do parallelSum
	check(cudaMemcpy(logProb, device_logProb, numPoints * sizeof(double), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
	
	cudaFree(device_logPi);
	cudaFree(device_logProb);
}

extern "C" void parallelLogMVNormDist(
	const size_t numPoints, const size_t pointDim,
	const double* X, const double* mu, const double* sigmaL, const double logNormalizer,
	double* logP
) {
	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double* device_sigmaL;
	const size_t sigmaLBytes = pointDim * pointDim * sizeof(double);
	check(cudaMalloc(&device_sigmaL, sigmaLBytes));
	check(cudaMemcpy(device_sigmaL, sigmaL, sigmaLBytes, cudaMemcpyHostToDevice));

	double* device_mu;
	const size_t muBytes = pointDim * sizeof(double);
	check(cudaMalloc(&device_mu, muBytes));
	check(cudaMemcpy(device_mu, mu, muBytes, cudaMemcpyHostToDevice));

	double* device_X;
	const size_t XBytes = numPoints * pointDim * sizeof(double);
	check(cudaMalloc(&device_X, XBytes));
	check(cudaMemcpy(device_X, X, XBytes, cudaMemcpyHostToDevice));

	double* device_logP;
	double logPBytes = numPoints * sizeof(double);
	check(cudaMalloc(&device_logP, logPBytes));

	cudaLogMVNormDist<<<numPoints, 1>>>(
		numPoints, pointDim,
		device_X, device_mu, device_sigmaL, logNormalizer,
		device_logP
		);

	check(cudaMemcpy(logP, device_logP, logPBytes, cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	cudaFree(device_sigmaL);
	cudaFree(device_mu);
	cudaFree(device_X);
	cudaFree(device_logP);
}

extern "C" double parallelSum(double* host_a, const size_t N) {
	assert(host_a != NULL);
	assert(N > 0);
	assertPowerOfTwo(N);

	const size_t NBytes = N * sizeof(double);

	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double *device_a;
	check(cudaMalloc(&device_a, NBytes));
	check(cudaMemcpy(device_a, host_a, NBytes, cudaMemcpyHostToDevice));

	// Parallel sum by continually folding the array in half and adding the right 
	// half to the left half until the fold size is 1024 (single block), then let
	// GPU reduce the remaining block to a single value and copy it over. O(log n).
	if(N >= 1024) {
		dim3 block, grid;
		for(size_t n = N/2; n >= 1024; n /= 2) {
			calcDim(n, &deviceProp, &block, &grid);
			cudaSum<<<grid, block>>>(n, device_a, device_a + n);
		}
		cudaReduceBlocks<<<1, 1024>>>(device_a);
	} else {
		cudaReduceBlocks<<<1, N>>>(device_a);
	}

	double host_sum = 0;
	check(cudaMemcpy(&host_sum, device_a, sizeof(double), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	cudaFree(device_a);

	return host_sum;
}

