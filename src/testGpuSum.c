#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

void testParallelEvens() {
	const size_t minN = 2;
	const size_t maxN = 16 * 1048576;
	for(size_t N = minN; N <= maxN; N *= 2) {
		double* a = (double*) malloc(N * sizeof(double));
		initialize(a, N);

		double host_sum = sequentialSum(a, N);
		double device_sum = gpuSum(N, a);

		double absDiff = fabs(host_sum - device_sum);
		if(absDiff > DBL_EPSILON) {
			printf("N: %zu, host_sum: %.15f, device_sum: %.15f, absDiff: %.15f\n", 
				N, host_sum, device_sum, absDiff
				);
			break;
		}

		free(a);
	}
}

void testParallelOdds() {
	const size_t minN = 1;
	const size_t maxN = 10000;
	for(size_t N = minN; N <= maxN; N += 9) {
		double* a = (double*) malloc(N * sizeof(double));
		initialize(a, N);

		double host_sum = sequentialSum(a, N);
		double device_sum = gpuSum(N, a);
		assert(device_sum != -INFINITY);
		assert(device_sum != INFINITY);
		assert(device_sum == device_sum);

		double absDiff = fabs(host_sum - device_sum);
		if(absDiff > DBL_EPSILON) {
			printf("N: %zu, host_sum: %.15f, device_sum: %.15f, absDiff: %.15f\n", 
				N, host_sum, device_sum, absDiff
				);
			break;
		}

		free(a);
	}
}
int main(int argc, char** argv) {
	testParallelEvens();
	testParallelOdds();

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}

