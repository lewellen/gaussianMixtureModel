#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cudaWrappers.h"

void initialize(double* a, size_t N) {
	for(size_t i = 0; i < N; ++i) {
		a[i] = i + 1; 
	}
}

double sequentialSum(double* a, const size_t N) {
	assert(a != NULL);
	assert(N > 0);

	double sum = 0;
	for(size_t i = 0; i < N; ++i) {
		sum += a[i];
	}
	return sum;
}

void test1DEvens() {
	const size_t minN = 2;
	const size_t maxN = 16 * 1048576;
	for(size_t N = minN; N <= maxN; N *= 2) {
		double* a = (double*) malloc(N * sizeof(double));
		initialize(a, N);

		double host_sum = sequentialSum(a, N);
		double device_sum;
		gpuSum(N, 1, a, &device_sum);

		assert(device_sum != -INFINITY);
		assert(device_sum != INFINITY);
		assert(device_sum == device_sum);

		double absDiff = fabs(host_sum - device_sum);
		if(absDiff >= DBL_EPSILON) {
			printf("N: %zu, host_sum: %.16f, device_sum: %.16f, absDiff: %.16f\n", 
				N, host_sum, device_sum, absDiff
				);
		}

		assert(absDiff < DBL_EPSILON);

		free(a);
	}
}

void test2DOdds() {
	const size_t minN = 1;
	const size_t maxN = 10000;
	const size_t pointDim = 2;

	for(size_t N = minN; N <= maxN; N += 9) {
		double* a = (double*) malloc(pointDim * N * sizeof(double));
		for(size_t i = 0; i < N; ++i) {
			for(size_t j = 0; j < pointDim; ++j) {
				a[pointDim * i + j ] = i + j;
			}
		}

		double host_sum[pointDim];
		memset(host_sum, 0, pointDim * sizeof(double));

		for(size_t i = 0; i < N; ++i) {
			for(size_t j = 0; j < pointDim; ++j) {
				host_sum[j] += a[pointDim * i + j];
			}
		}	
	
		double device_sum[pointDim];
		memset(device_sum, 0, pointDim * sizeof(double));

		gpuSum(N, pointDim, a, device_sum);

		for(size_t i = 0; i < pointDim; ++i) {
			assert(device_sum[i] != -INFINITY);
			assert(device_sum[i] != INFINITY);
			assert(device_sum[i] == device_sum[i]);

			double absDiff = fabs(host_sum[i] - device_sum[i]);
			if(absDiff >= DBL_EPSILON) {
				printf("N: %zu, i: %zu, host_sum: %.16f, device_sum: %.16f, absDiff: %.16f\n", 
					N, i, host_sum[i], device_sum[i], absDiff
					);
			}

			assert(absDiff < DBL_EPSILON);
		}

		free(a);
	}
}

int main(int argc, char** argv) {
	test1DEvens();
	test2DOdds();

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}

