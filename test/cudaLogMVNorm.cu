#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define check(call) { cudaError_t __ABC123 = call; assert(__ABC123 == cudaSuccess); }

__device__ void devVecMinus(const size_t N, double* x, double* y) {
	for(size_t i = 0; i < N; ++i) {
		x[i] -= y[i];
	}
}

__device__ void devSolveLowerTri(const size_t N, double* L, double* x, double* b) {
	for(size_t i = 0; i < N; ++i) {
		double sum = 0.0;
		if(i > 0) {
			for(size_t j = 0; j <= i - 1; ++j) {
				sum += L[i * N + j] * x[j];
			}
		}

		x[i] = (b[i] - sum) / L[i * N + i];
	}
}

__device__ void devSolveLowerTriT(const size_t N, double* L, double* x, double* b) {
	// treat L as an upper triangular matrix U
	for(size_t i = 0; i < N; i++) {
		size_t ip = N - 1 - i;
		double sum = 0;
		for(size_t j = ip + 1; j < N; ++j) {
			sum += L[j * N + ip] * x[j];
		}

		x[ip] = (b[ip] - sum) / L[ip * N + ip];
	}
}

__device__ double devVecDot(const size_t N, double* x, double* y) {
	double sum = 0;
	for(size_t i = 0; i < N; ++i) {
		sum += x[i] * y[i];
	}
	return sum;
}

__global__ void cudaLogMVNorm(
	const size_t numPoints, const size_t pointDim, 
	double* X, double* mu, double* sigmaL, double logNormalizer,
	double* dest
) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	double u[2048];

	devVecMinus(pointDim, &X[i * pointDim], mu); // x[i] -= mu[i]
	devSolveLowerTri(pointDim, sigmaL, u, &X[i * pointDim]); // L u = (x - mu)
	devSolveLowerTriT(pointDim, sigmaL, &u[pointDim], u); // L^T v = u
	dest[i] = logNormalizer - 0.5 * devVecDot(pointDim, &u[pointDim], & X[i]);
}

int main(int argc, char** argv) {
	const size_t pointDim = 1;
	const size_t numPoints = 1024;

	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	double sigmaL[pointDim * pointDim];
	memset(sigmaL, 0, pointDim * pointDim * sizeof(double));
	for(size_t i = 0; i < pointDim; ++i) {
		sigmaL[i * pointDim + i] = 1;
	}

	double* device_sigmaL;
	check(cudaMalloc(&device_sigmaL, pointDim * pointDim * sizeof(double)));
	check(cudaMemcpy(device_sigmaL, sigmaL, pointDim * pointDim * sizeof(double), cudaMemcpyHostToDevice));

	double det = 1;
	for(size_t i = 0; i < pointDim; ++i) {
		det *= sigmaL[i * pointDim + i] * sigmaL[i * pointDim + i];
	}

	double logNormalizer = -0.5 * pointDim * log(2.0 * M_PI) - 0.5 * log(det);

	double mu[pointDim];
	memset(mu, 0, pointDim * sizeof(double));

	double* device_mu;
	check(cudaMalloc(&device_mu, pointDim * sizeof(double)));
	check(cudaMemcpy(device_mu, mu, pointDim * sizeof(double), cudaMemcpyHostToDevice));

	double X[pointDim * numPoints];
	memset(X, 0, pointDim * numPoints * sizeof(double));
	for(size_t i = 0; i < numPoints; ++i) {
		X[i * pointDim + 0] = 3.0 * ( ( (double)i - (double)numPoints/2 ) / (double)(numPoints/2.0) );
	}

	double* device_X;
	check(cudaMalloc(&device_X, pointDim * numPoints * sizeof(double)));
	check(cudaMemcpy(device_X, X, pointDim * numPoints * sizeof(double), cudaMemcpyHostToDevice));

	double logP[numPoints];
	memset(logP, 0, numPoints * sizeof(double));

	double* device_logP;
	check(cudaMalloc(&device_logP, numPoints * sizeof(double)));

	cudaLogMVNorm<<<numPoints, 1>>>(
		numPoints, pointDim,
		device_X, device_mu, device_sigmaL, logNormalizer,
		device_logP
		);

	check(cudaMemcpy(logP, device_logP, numPoints * sizeof(double), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	cudaFree(device_sigmaL);
	cudaFree(device_mu);
	cudaFree(device_X);
	cudaFree(device_logP);

	cudaDeviceReset();

	for(size_t i = 0; i < numPoints; ++i) {
		double x = 3.0 * ( ( (double)i - (double)numPoints/2 ) / (double)(numPoints/2.0) );
		printf("%.3f %.6f\n", x, exp(logP[i]));
	}

	return EXIT_SUCCESS;
}
