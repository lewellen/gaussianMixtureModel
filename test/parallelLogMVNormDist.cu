#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

#define check(call) { cudaError_t __ABC123 = call; assert(__ABC123 == cudaSuccess); }

__device__ void devVecMinus(const size_t N, double* x, double* y) {
	assert(N > 0);
	assert(x != NULL);
	assert(y != NULL);
	// x == y allowed

	for(size_t i = 0; i < N; ++i) {
		x[i] -= y[i];
	}
}

__device__ void devSolveLowerTri(const size_t N, double* L, double* x, double* b) {
	assert(N > 0);
	assert(L != NULL);
	assert(x != NULL);
	assert(b != NULL);
	// x == b allowed

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
	assert(N > 0);
	assert(L != NULL);
	assert(x != NULL);
	assert(b != NULL);
	// x == b allowed

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
	assert(N > 0);
	assert(x != NULL);
	assert(y != NULL);

	double sum = 0;
	for(size_t i = 0; i < N; ++i) {
		sum += x[i] * y[i];
	}
	return sum;
}

__global__ void cudaLogMVNormDist(
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

	double u[1024];

	double* x = & X[i * pointDim];
	devVecMinus(pointDim, x, mu); // x <- x - mu
	devSolveLowerTri(pointDim, sigmaL, u, x); // u <- u s.t. L u = (x - mu)
	devSolveLowerTriT(pointDim, sigmaL, u, u); // u <- v s.t. L^T v = u
	dest[i] = logNormalizer - 0.5 * devVecDot(pointDim, u, x);
}

void parallelLogMVNormDist(
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

void test1DStandardNormal() {
	const size_t pointDim = 1;
	const size_t numPoints = 1024;

	double sigmaL[pointDim * pointDim];
	memset(sigmaL, 0, pointDim * pointDim * sizeof(double));
	for(size_t i = 0; i < pointDim; ++i) {
		sigmaL[i * pointDim + i] = 1;
	}

	double det = 1;
	for(size_t i = 0; i < pointDim; ++i) {
		det *= sigmaL[i * pointDim + i] * sigmaL[i * pointDim + i];
	}

	double logNormalizer = -0.5 * pointDim * log(2.0 * M_PI) - 0.5 * log(det);

	double mu[pointDim];
	memset(mu, 0, pointDim * sizeof(double));

	double X[pointDim * numPoints];
	memset(X, 0, pointDim * numPoints * sizeof(double));
	for(size_t i = 0; i < numPoints; ++i) {
		X[i * pointDim + 0] = 3.0 * ( ( (double)i - (double)numPoints/2 ) / (double)(numPoints/2.0) );
	}

	double logP[numPoints];
	memset(logP, 0, numPoints * sizeof(double));

	parallelLogMVNormDist(
		numPoints, pointDim,
		X, mu, sigmaL, logNormalizer,
		logP
	);

	double normalizer = sqrt(2.0 * M_PI);
	for(size_t i = 0; i < numPoints; ++i) {
		double x = X[i];
		double actual = exp(logP[i]);
		double expected = exp(-0.5 * x * x) / normalizer;

		double absDiff = abs(expected - actual);
		if(absDiff >= DBL_EPSILON) {
			printf("f(%.7f) = %.7f, but should equal = %.7f; absDiff = %.15f\n", 
				x, actual, expected, absDiff);
		}

		assert(absDiff < DBL_EPSILON);
	}
}

int main(int argc, char** argv) {
	test1DStandardNormal();

	cudaDeviceReset();

	printf("PASS %s\n", argv[0]);
	return EXIT_SUCCESS;
}
