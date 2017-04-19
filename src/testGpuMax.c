#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cudaWrappers.h"

void test(const size_t N, double* a) {
	for(size_t i = 0; i < N; ++i) {
		a[i] = i;
	}

	double host_max = -INFINITY;

	for(size_t i = 0; i < N; ++i) {
		if(host_max < a[i]) {
			host_max = a[i];
		}
	}	

	double device_max = -INFINITY;

	device_max = gpuMax(N, a);

	assert(device_max != -INFINITY);
	assert(device_max != INFINITY);
	assert(device_max == device_max);

	double absDiff = fabs(host_max - device_max);
	if(absDiff >= DBL_EPSILON) {
		printf("N: %zu, host_max: %.16f, device_max: %.16f, absDiff: %.16f\n", 
			N, host_max, device_max, absDiff
			);
	}

	assert(absDiff < DBL_EPSILON);
}

void testPowTwos() {
	const size_t minN = 2;
	const size_t maxN = 16 * 1048576;

	for(size_t N = minN; N <= maxN; N *= 2) {
		double* a = (double*) malloc(N * sizeof(double));
		test(N, a);
		free(a);
	}
}

void testEvens() {
	const size_t minN = 1;
	const size_t maxN = 10000;

	for(size_t N = minN; N <= maxN; N += 8) {
		double* a = (double*) malloc(N * sizeof(double));
		test(N, a);
		free(a);
	}
}

void testOdds() {
	const size_t minN = 1;
	const size_t maxN = 10000;

	for(size_t N = minN; N <= maxN; N += 9) {
		double* a = (double*) malloc(N * sizeof(double));
		test(N, a);
		free(a);
	}
}

int main(int argc, char** argv) {
	testPowTwos();
	testEvens();
	testOdds();

	printf("PASS: %s\n", argv[0]);
	return EXIT_SUCCESS;
}

