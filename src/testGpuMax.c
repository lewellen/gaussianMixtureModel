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

double sequentialMax(double* a, const size_t N) {
	assert(a != NULL);
	assert(N > 0);

	double max = -INFINITY;
	for(size_t i = 0; i < N; ++i) {
		if(max < a[i]) {
			max = a[i];
		}
	}
	return max;
}

void testEvens() {
	const size_t minN = 2;
	const size_t maxN = 16 * 1048576;
	for(size_t N = minN; N <= maxN; N *= 2) {
		double* a = (double*) malloc(N * sizeof(double));
		initialize(a, N);

		double host_max = sequentialMax(a, N);
		double device_max = gpuMax(N, a);

		double absDiff = fabs(host_max - device_max);
		if(absDiff >= DBL_EPSILON) {
			printf("N: %zu, host_max: %.15f, device_max: %.15f, absDiff: %.15f\n", 
				N, host_max, device_max, absDiff
				);
		}

		assert(absDiff < DBL_EPSILON);

		free(a);
	}
}

void testOdds() {
	const size_t minN = 1;
	const size_t maxN = 10000;
	for(size_t N = minN; N <= maxN; N += 9) {
		double* a = (double*) malloc(N * sizeof(double));
		initialize(a, N);

		double host_max = sequentialMax(a, N);
		double device_max = gpuMax(N, a);
		assert(device_max != -INFINITY);
		assert(device_max != INFINITY);
		assert(device_max == device_max);

		double absDiff = fabs(host_max - device_max);
		if(absDiff >= DBL_EPSILON) {
			printf("N: %zu, host_max: %.15f, device_max: %.15f, absDiff: %.15f\n", 
				N, host_max, device_max, absDiff
				);
		}

		assert(absDiff < DBL_EPSILON);
		free(a);
	}
}
int main(int argc, char** argv) {
	testEvens();
	testOdds();

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}

