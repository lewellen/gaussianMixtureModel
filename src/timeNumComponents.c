#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "gmm.h"
#include "seqGmm.h"
#include "parallelGmm.h"
#include "util.h"

int main(int argc, char** argv) {
	const size_t minNumComponents = 1;
	const size_t maxNumComponents = 256;

	const size_t numPoints = 1 << 14;
	const size_t pointDim = 1;

	const size_t numSamples = 10;

	double* X = (double*) malloc(numPoints * pointDim * sizeof(double));
	for(size_t i = 0; i < numPoints * pointDim; ++i) {
		X[i] = rand() / (double)RAND_MAX;
	}

	struct timeval start, end;

	fprintf(stdout, "#numPoints numComponents pointDim seqElapsedSec parallelElapsedSec\n");
	for(size_t sample = 0; sample < numSamples; ++sample) {	
		for(size_t numComponents = minNumComponents; numComponents < maxNumComponents; numComponents *= 2) {
			gettimeofday(&start, NULL);
			freeGMM(fit(X, numPoints, pointDim, numComponents));
			gettimeofday(&end, NULL);
			double seqElapsedSec = calcElapsedSec(&start, &end);

			gettimeofday(&start, NULL);
			freeGMM(parallelFit(X, numPoints, pointDim, numComponents));
			gettimeofday(&end, NULL);
			double parallelElapsedSec = calcElapsedSec(&start, &end);

			fprintf(stdout, "%zu %zu %zu %f %f\n", 
				numPoints, numComponents, pointDim, seqElapsedSec, parallelElapsedSec);
		}
	}

	free(X);

	return EXIT_SUCCESS;
}
