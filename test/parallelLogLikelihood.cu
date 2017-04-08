#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define check(call) { cudaError_t __ABC123 = call; assert(__ABC123 == cudaSuccess); }

__global__ void cudaGmmLogLikelihood(
	const size_t numPoints, const size_t numComponents,
	const double* logpi, double* logProb
) { 
	// Input:
	// logpi = 1 x numComponets
	// logProb = numComponents x numPoints

	// Output:
	// logProb[i] = log likelihood of single point i

	assert(numPoints > 0);
	assert(numComponents > 0);
	assert(logpi != NULL);
	assert(logProb != NULL);

	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	
	double maxArg = -INFINITY;
	for(size_t k = 0; k < numComponents; ++k) {
		const double logProbK = logpi[k] + logProb[k * numPoints + i];
		if(logProbK > maxArg) {
			maxArg = logProbK;
		}
	}

	double sum = 0.0;
	for (size_t k = 0; k < numComponents; ++k) {
		const double logProbK = logpi[k] + logProb[k * numPoints + i];
		sum = exp(logProbK - maxArg);
	}

	assert(sum >= 0);

	logProb[i] = maxArg + log(sum);
}

void parallelGmmLogLikelihood(
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

void testSingleStdNormLogL() {
	const size_t numPoints = 1024;
	const size_t numComponents = 1;

	double logPi[numComponents];
	double logUniform = -log((double)numComponents);
	for(size_t k = 0; k < numComponents; ++k) {
		logPi[k] = logUniform;
	}

	double logProb[numComponents * numPoints];
	for(size_t k = 0; k < numComponents; ++k) {
		for(size_t i = 0; i < numPoints; ++i) {
			double x = 3.0 * ((double)i - (double)(numPoints/2.0))/((double)numPoints/2.0);
			logProb[numPoints * k + i] = -0.5 * sqrt(2.0 * M_PI) - 0.5 * x * x;
		}
	}
	
	parallelGmmLogLikelihood(
		numPoints, numComponents,
		logPi, logProb
	);

	double expected = 0;
	for(size_t k = 0; k < numComponents; ++k) {
		for(size_t i = 0; i < numPoints; ++i) {
			double x = 3.0 * ((double)i - (double)(numPoints/2.0))/((double)numPoints/2.0);
			expected += logPi[k] - 0.5 * sqrt(2.0 * M_PI) - 0.5 * x * x;
		}
	}

	double actual = 0;
	for(size_t i = 0; i < numPoints; ++i) {
		actual += logProb[i];
	}

	double absDiff = abs(expected - actual);
	if(absDiff >= DBL_EPSILON) {
		printf("actual = %.15f, expected = %.15f, absDiff = %.15f\n",
			actual, expected, absDiff);
	}

	assert(absDiff < DBL_EPSILON);
}

int main(int argc, char** argv) {
	testSingleStdNormLogL();

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}
