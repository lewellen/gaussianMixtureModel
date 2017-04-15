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

struct GmmEmGpuCtx {
	size_t numPoints;
	size_t pointDim;
	size_t numComponents;

	double* X;
	double* logpi;
	double* mu;
	double* sigmaL;
	double* loggamma;

	size_t M;
	cudaDeviceProp* deviceProp;
	double* device_X;
	double* device_logpi;
	double* device_mu;
	double* device_sigmaL;
	double* device_loggamma;
};

extern "C" struct GmmEmGpuCtx* gpuInitCtx(
	size_t numPoints,
	size_t pointDim,
	size_t numComponents,
	double* X,
	double* logpi,
	double* mu,
	double* sigmaL,
	double* loggamma
) {
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(numComponents > 0);

	assert(X != NULL);
	assert(logpi != NULL);
	assert(mu != NULL);
	assert(sigmaL != NULL);
	assert(loggamma != NULL);

	struct GmmEmGpuCtx* ctx = (struct GmmEmGpuCtx*) malloc(sizeof(struct GmmEmGpuCtx));

	ctx->numPoints = numPoints;
	ctx->pointDim = pointDim;
	ctx->numComponents = numComponents;
	
	ctx->X = X;
	ctx->logpi = logpi;
	ctx->mu = mu;
	ctx->sigmaL = sigmaL;
	ctx->loggamma = loggamma;

	int deviceId;
	check(cudaGetDevice(&deviceId));

	ctx->deviceProp = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp));
	check(cudaGetDeviceProperties(ctx->deviceProp, deviceId));

	ctx->M = largestPowTwoLessThanEq(numPoints);

	check(ctx->device_X = sendToGpu(numPoints * pointDim, X));
	check(ctx->device_logpi = sendToGpu(numComponents, logpi));
	check(ctx->device_mu = sendToGpu(numComponents * pointDim, mu));
	check(ctx->device_sigmaL = sendToGpu(numComponents * pointDim * pointDim, sigmaL));
	check(ctx->device_loggamma = sendToGpu(numPoints * pointDim, loggamma));

	return ctx;
}

extern "C" void gpuDestroyCtx(
	struct GmmEmGpuCtx* ctx
) {
	assert(ctx != NULL);
	cudaFree(ctx->device_X);
	cudaFree(ctx->device_logpi);
	cudaFree(ctx->device_mu);
	cudaFree(ctx->device_sigmaL);
	cudaFree(ctx->device_loggamma);

	free(ctx->deviceProp);
	free(ctx);
}

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

extern "C" double gpuPerformEStep(
	struct GmmEmGpuCtx* ctx
) {
	assert(ctx != NULL);

	check(cudaMemcpy(ctx->device_mu, ctx->mu, ctx->numComponents * ctx->pointDim * sizeof(double), cudaMemcpyHostToDevice));
	check(cudaMemcpy(ctx->device_mu, ctx->sigma, ctx->numComponents * ctx->pointDim * ctx->pointDim * sizeof(double), cudaMemcpyHostToDevice));

	dim3 grid, block;
	calcDim(ctx->numPoints, ctx->deviceProp, &block, &grid);
	for(size_t k = 0; k < ctx->numComponents; ++k) {
		kernLogMVNormDist<<<grid, block>>>(
			ctx->numPoints, ctx->pointDim,
			ctx->device_X, 
			& ctx->device_mu[k * ctx->pointDim], 
			& ctx->device_sigmaL[k * ctx->pointDim * ctx->pointDim],
			& ctx->device_loggamma[k * ctx->numPoints]
		);
	}

	double logL = cudaGmmLogLikelihoodAndGammaNK(
		ctx->deviceProp,
		ctx->numPoints, ctx->numComponents,
		ctx->M,
		ctx->logpi, ctx->loggamma,
		ctx->device_logpi, ctx->device_loggamma
	);

	// decide if necessary to keep
	check(cudaMemcpy(logP, device_logP, 
		numPoints * numComponents * sizeof(double), cudaMemcpyDeviceToHost));

	return logL;
}

extern "C" void gpuPerformMStep(
	struct GmmEmGpuCtx* ctx
) {
	for(size_t k = 0; k < numComponents; ++k) {
		// update pi_+1
		*logpi[k] += logGamma[k] - logGammaSum;

		// calculate mu_+1
		cudaUpdateMu(
			ctx->deviceProp,
			ctx->numPoints, ctx->pointDim,
			ctx->M,
			ctx->X, & ctx->loggamma[k * numPoints], ctx->logGamma[k],
			ctx->device_X, & ctx->device_loggamma[k * numPoints],
			& ctx->mu[k * pointDim]
		);

		cudaDeviceSynchronize();

		// Calculate sigma_+1
		cudaUpdateSigma(
			ctx->deviceProp,
			ctx->numPoints, ctx->pointDim,
			ctx->M,
			ctx->X, & ctx->loggamma[k * ctx->numPoints], logGamma[k],
			ctx->device_X, & ctx->device_loggamma[k * ctx->numPoints],
			ctx->mu[k * ctx->pointDim], 
			ctx->sigma[k * ctx->pointDim * ctx->pointDim];
		);

		// doing the cholesky decomposition is caller (cpu) side for now
	}

}
