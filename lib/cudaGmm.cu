#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaCommon.hu"
#include "cudaFolds.hu"
#include "cudaGmm.hu"

__global__ void kernCalcLogLikelihoodAndGammaNK(
	const size_t M, const size_t numPoints, const size_t numComponents,
	const double* logpi, double* logPx, double* loggamma
) {
	// loggamma[k * numPoints + i] = 
	// On Entry: log p(x_i | mu_k, Sigma_k)
	// On exit: [log pi_k] + [log p(x_i | mu_k, sigma_k)] - [log p(x_i)]

	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= M) {
		return;
	}

	double maxArg = -INFINITY;
	for(size_t k = 0; k < numComponents; ++k) {
		const double logProbK = logpi[k] + loggamma[k * numPoints + i];
		if(logProbK > maxArg) {
			maxArg = logProbK;
		}
	}

	double sum = 0.0;
	for (size_t k = 0; k < numComponents; ++k) {
		const double logProbK = logpi[k] + loggamma[k * numPoints + i];
		sum += exp(logProbK - maxArg);
	}

	assert(sum >= 0);
	const double logpx = maxArg + log(sum);

	for(size_t k = 0; k < numComponents; ++k) {
		loggamma[k * numPoints + i] += -logpx;
	}

	logPx[i] = logpx;
}

__host__ double cudaGmmLogLikelihoodAndGammaNK(
	cudaDeviceProp* deviceProp,
	const size_t numPoints, const size_t numComponents,
	const size_t M,
	const double* logpi, double* logP,
	const double* device_logpi, double* device_logP
) {
	// logpi: 1 x numComponents
	// logP: numComponents x numPoints

	// Do the first M (2^n) points on the gpu; remainder on cpu

	dim3 grid, block;
	calcDim(M, deviceProp, &block, &grid);

	double logL = 0;
	double* device_logPx = mallocOnGpu(M);

	kernCalcLogLikelihoodAndGammaNK<<<grid, block>>>(
		M, numPoints, numComponents,
		device_logpi, device_logPx, device_logP
	);

	cudaArraySum(
		deviceProp, 
		M, 1, 
		device_logPx 
	);

	check(cudaMemcpy(&logL, device_logPx, sizeof(double), cudaMemcpyDeviceToHost));

	cudaFree(device_logPx);

	// Copy back the full numPoints * numComponents values
	check(cudaMemcpy(logP, device_logP, 
		numPoints * numComponents * sizeof(double), cudaMemcpyDeviceToHost));

	if(M != numPoints) {
		for (size_t point = M; point < numPoints; ++point) {
			double maxArg = -INFINITY;
			for(size_t k = 0; k < numComponents; ++k) {
				const double logProbK = logpi[k] + logP[k * numPoints + point];
				if(logProbK > maxArg) {
					maxArg = logProbK;
				}
			}

			double sum = 0.0;
			for (size_t k = 0; k < numComponents; ++k) {
				const double logProbK = logpi[k] + logP[k * numPoints + point];
				sum += exp(logProbK - maxArg);
			}

			assert(sum >= 0);
			const double logpx = maxArg + log(sum);

			for(size_t k = 0; k < numComponents; ++k) {
				logP[k * numPoints + point] += -logpx;
			}

			logL += logpx;
		}
	}

	return logL;
}

__global__ void kernExp(double* A, double* bias) {
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	A[i] = exp(A[i] - *bias);
}

__global__ void kernBiasAndLog(double* sumexp, double* bias) {
	*sumexp = *bias + log(*sumexp);
}

__host__ void cudaLogSumExp(
	cudaDeviceProp* deviceProp, dim3 grid, dim3 block, 
	const size_t numPoints,
	double* device_src, double* device_dest, 
	double* device_working, 
	cudaStream_t stream
) {
	cudaEvent_t a, b;
	check(cudaEventCreateWithFlags(&a, cudaEventDisableTiming));
	check(cudaEventCreateWithFlags(&b, cudaEventDisableTiming));

	// dest <- src
	check(cudaMemcpyAsync(
		device_dest, device_src, 
		numPoints * sizeof(double), 
		cudaMemcpyDeviceToDevice,
		stream
	));

	check(cudaEventRecord(b, stream));

	// working <- src
	check(cudaMemcpyAsync(
		device_working, device_src, 
		numPoints * sizeof(double), 
		cudaMemcpyDeviceToDevice,
		stream
	));

	check(cudaEventRecord(a, stream));

	check(cudaStreamWaitEvent(stream, b, 0));

	// working <- max { src }
	cudaArrayMax(deviceProp, numPoints, device_working, stream);

	check(cudaStreamWaitEvent(stream, a, 0));

	// dest <- exp(src - max)
	kernExp<<<grid, block, 0, stream>>>(
		device_dest,
		device_working
	);

	// dest <- sum exp(src - max)
	cudaArraySum(deviceProp, numPoints, 1, device_dest, stream);

	// dest <- max + log sum exp(src - max)
	kernBiasAndLog<<<1, 1, 0, stream>>>(
		device_dest, device_working
	);

	cudaEventDestroy(a);
	cudaEventDestroy(b);
}

__global__ void kernCalcMu(
	const size_t numPoints, const size_t pointDim,
	const double* X, const double* loggamma, const double* GammaK,
	double* dest
) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	const double a = exp(loggamma[i]) / exp(*GammaK);
	const double* x = & X[i * pointDim];
	double* y = & dest[i * pointDim]; 

	for(size_t i = 0; i < pointDim; ++i) {
		y[i] = a * x[i];
	}
}

__global__ void kernCalcSigma(
	const size_t numPoints, const size_t pointDim,
	const double* X, const double* mu, const double* loggamma, const double* GammaK,
	double* dest
) {
	assert(pointDim < 1024);
	
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	// gamma_{n,k} / Gamma_{k} (x - mu) (x - mu)^T

	const double a = exp(loggamma[i]) / exp(*GammaK);
	const double* x = & X[i * pointDim];
	double* y = & dest[i * pointDim * pointDim]; 

	double u[1024];
	for(size_t i = 0; i < pointDim; ++i) {
		u[i] = x[i] - mu[i];
	}

	for(size_t i = 0; i < pointDim; ++i) {
		double* yp = &y[i * pointDim];
		for(size_t j = 0; j < pointDim; ++j) {
			yp[j] = a * u[i] * u[j];
		}
	}
}

__global__ void kernUpdatePi(
	const size_t numPoints, const size_t numComponents,
	double* logpi, double* Gamma
) {
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int comp = b * blockDim.x + threadIdx.x;
	if(comp > numComponents) {
		return;
	}

	__shared__ double A[1024];
	A[comp] = logpi[comp] + log(Gamma[comp * numPoints]);
	__syncthreads();

	double sum = 0;
	for(size_t k = 0; k < numComponents; ++k) {
		sum += exp(A[k]);
	}

	logpi[comp] = A[comp] - log(sum);
}

__global__ void kernPrepareCovariances(
	const size_t numComponents, const size_t pointDim,
	double* Sigma, double* SigmaL,
	double* normalizers
) {
	// Parallel in the number of components

	// Sigma: numComponents x pointDim * pointDim
	// SigmaL: numComponents x pointDim * pointDim
	// normalizers: 1 x numComponents

	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int comp = b * blockDim.x + threadIdx.x;
	if(comp > numComponents) {
		return;
	}

	// L is the resulting lower diagonal portion of A = LL^T
	const size_t ALen = pointDim * pointDim;
	double* A = & Sigma[comp * ALen];
	double* L = & SigmaL[comp * ALen];
	for(size_t i = 0; i < ALen; ++i) { 
		L[i] = 0;
	}

	for (size_t k = 0; k < pointDim; ++k) {
		double sum = 0;
		for (int s = 0; s < k; ++s) {
			const double l = L[k * pointDim + s];
			const double ll = l * l;
			sum += ll;
		}

		assert(sum >= 0);

		sum = A[k * pointDim + k] - sum;
		if (sum <= DBL_EPSILON) {
			printf("A must be positive definite. (sum = %E)\n", sum);
			assert(sum > 0);
			break;
		}

		L[k * pointDim + k] = sqrt(sum);
		for (int i = k + 1; i < pointDim; ++i) {
			double subsum = 0;
			for (int s = 0; s < k; ++s)
				subsum += L[i * pointDim + s] * L[k * pointDim + s];

			L[i * pointDim + k] = (A[i * pointDim + k] - subsum) / L[k * pointDim + k];
		}
	}

	double logDet = 1.0;
	for (size_t i = 0; i < pointDim; ++i) {
		double diag = L[i * pointDim + i];
		assert(diag > 0);
		logDet += log(diag);
	}

	logDet *= 2.0;

	normalizers[comp] = - 0.5 * (pointDim * log(2.0 * M_PI) + logDet);
}
