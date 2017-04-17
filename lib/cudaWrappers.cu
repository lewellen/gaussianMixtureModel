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
	const double* logpi, double* logP
) {
	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	const size_t M = largestPowTwoLessThanEq(numPoints); 

	double* device_logpi = sendToGpu(numComponents, logpi);
	
	// Sending all data because logP is an array organized by:
	// [ <- numPoints -> ]_0 [ <- numPoints -> ]_... [ <- numPoints -> ]_{k-1}
	// So even though we are only using M of those points on the GPU,
	// we need all numPoints to ensure indexing by numPoints * k + i works
	// correctly to access prob(x_i|mu_k,Sigma_k).
	double* device_logP = sendToGpu(numComponents * numPoints, logP);

	double logL = cudaGmmLogLikelihoodAndGammaNK(
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
	const size_t numPoints, const size_t numComponents,
	const double* logpi, double* loggamma
) { 
	gpuGmmLogLikelihood(
		numPoints, numComponents,
		logpi, loggamma
	); 
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

__global__ void kernExp(double* A) {
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	A[i] = exp(A[i]);
}

extern "C" void gpuGmmFit(
	const double* X,
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents,
	double* pi,
	double* Mu,
	double* Sigma,
	double* SigmaL,
	double* normalizers
) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0 && pointDim <= 1024);
	assert(numComponents > 0 && numComponents <= 1024);

	assert(pi != NULL);
	assert(Mu != NULL);
	assert(Sigma != NULL);
	assert(SigmaL != NULL);
	assert(normalizers != NULL);

	int deviceId;

	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double* device_X = sendToGpu(numPoints * pointDim, X);

	for(size_t i = 0; i < numComponents; ++i) {
		assert(pi[i] > 0);
		pi[i] = log(pi[i]);
	}

	double* device_logpi = sendToGpu(numComponents, pi);
	double* device_Mu = sendToGpu(numComponents * pointDim, Mu);
	double* device_Sigma = sendToGpu(numComponents * pointDim * pointDim, Sigma);

	double* device_SigmaL = sendToGpu(numComponents * pointDim * pointDim, SigmaL);
	double* device_normalizers = sendToGpu(numComponents, normalizers);


	double* device_loggamma = mallocOnGpu(numPoints * numComponents);
	double* device_logGamma = mallocOnGpu(numPoints * numComponents);	

	double currentLogL = -INFINITY, previousLogL = -INFINITY;

	// logPx, mu, sigma reductions
	// This means for mu and sigma can only do one component at a time otherwise 
	// the memory foot print will limit how much data we can actually work with.
	double* device_working = mallocOnGpu(numPoints * pointDim * pointDim * numComponents);

	dim3 grid, block;
	calcDim(numPoints, &deviceProp, &block, &grid);

	size_t iteration = 0;
	const size_t maxIterations = 100;
	const double tolerance = 1e-8;

	do {
		// --------------------------------------------------------------------------
		// E-Step
		// --------------------------------------------------------------------------

		// loggamma[k * numPoints + i] = p(x_i | mu_k, Sigma_k )
		for(size_t k = 0; k < numComponents; ++k) {
			// Fill in numPoint many probabilities
			kernLogMVNormDist<<<grid, block>>>(
				numPoints, pointDim,
				device_X, 
				& device_Mu[k * pointDim], 
				& device_SigmaL[k * pointDim * pointDim],
				& device_loggamma[k * numPoints]
			);
		}

		// Need all of the above log prob (x_i | ...) value before we can continue
		cudaDeviceSynchronize();

		// loggamma[k * numPoints + i] = p(x_i | mu_k, Sigma_k) / p(x_i)
		// working[i] = p(x_i)
		kernCalcLogLikelihoodAndGammaNK<<<grid, block>>>(
			numPoints, numPoints, numComponents,
			device_logpi, device_working, device_loggamma
		);

		// working[0] = sum_{i} p(x_i)
		cudaArraySum(&deviceProp, numPoints, 1, device_working);

		previousLogL = currentLogL;		
		check(cudaMemcpy(
			&currentLogL, device_working, 
			sizeof(double), 
			cudaMemcpyDeviceToHost
		));

		if(fabs(currentLogL - previousLogL) < tolerance || currentLogL < previousLogL) {
			break;
		}

		// logGamma = loggamma
		check(cudaMemcpy(
			device_logGamma, device_loggamma, 
			numPoints * numComponents * sizeof(double), 
			cudaMemcpyDeviceToDevice
		));

		for(size_t k = 0; k < numComponents; ++k) {
			kernExp<<<grid, block>>>(&device_logGamma[k * numPoints]);
		}

		cudaDeviceSynchronize();

		// logGamma[k * numPoints + 0] = sum_{i} loggamma[k * numPoints + i]
		for(size_t k = 0; k < numComponents; ++k) {
			cudaArraySum(&deviceProp, numPoints, 1, & device_logGamma[k * numPoints]);
		}

		// --------------------------------------------------------------------------
		// M-Step
		// --------------------------------------------------------------------------

		for(size_t k = 0; k < numComponents; ++k) {
			// working[i * pointDim + j] = gamma_ik / Gamma K * x_j
			kernCalcMu<<<grid, block>>>(
				numPoints, pointDim,
				device_X, 
				& device_loggamma[k * numPoints], 
				& device_logGamma[k * numPoints],
				device_working
			);

			// working[0 + j] = sum gamma_ik / Gamma K * x_j
			cudaArraySum(&deviceProp, numPoints, pointDim, device_working);

			check(cudaMemcpy(
				& device_Mu[k * pointDim],
				device_working,
				pointDim * sizeof(double),
				cudaMemcpyDeviceToDevice
			));

			// working[i * pointDim * pointDim + j] = 
			// 	gamma_ik / Gamma_k [ (x_i - mu) (x_i - mu)^T ]_j
			kernCalcSigma<<<grid, block>>>(
				numPoints, pointDim,
				device_X, 
				& device_Mu[k * pointDim],
				& device_loggamma[k * numPoints],
				& device_logGamma[k * numPoints],
				device_working
			);

			// working[0 + j] = sum gamma_ik / Gamma K * [...]_j
			cudaArraySum(&deviceProp, numPoints, pointDim * pointDim, device_working);

			check(cudaMemcpy(
				& device_Sigma[k * pointDim * pointDim],
				device_working,
				pointDim * pointDim * sizeof(double),
				cudaMemcpyDeviceToDevice
			));
		
			cudaDeviceSynchronize();		
		}
		
		// pi_k^(t+1) = pi_k Gamma_k / sum_{i}^{K} pi_i * Gamma_i
		// Use thread sync to compute denom to avoid data race
		kernUpdatePi<<<1, numComponents>>>(
			numPoints, numComponents,
			device_logpi, device_logGamma
		);

		// recompute sigmaL and normalizer
		kernPrepareCovariances<<<1, numComponents>>>(
			numComponents, pointDim,
			device_Sigma, device_SigmaL,
			device_normalizers
		);	

		cudaDeviceSynchronize();		
	
	} while(++iteration < maxIterations);

	cudaFree(device_working);

	cudaFree(device_logGamma);
	cudaFree(device_loggamma);

	check(cudaMemcpy(normalizers, device_normalizers, numComponents * sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(device_normalizers);

	check(cudaMemcpy(SigmaL, device_SigmaL, numComponents * pointDim * pointDim * sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(device_SigmaL);

	check(cudaMemcpy(Sigma, device_Sigma, numComponents * pointDim * pointDim * sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(device_Sigma);

	check(cudaMemcpy(Mu, device_Mu, numComponents * pointDim * sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(device_Mu);

	check(cudaMemcpy(pi, device_logpi, numComponents * sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(device_logpi);

	for(size_t i = 0; i < numComponents; ++i) {
		pi[i] = exp(pi[i]);
	}

	cudaFree(device_X);
}
