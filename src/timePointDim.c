#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include "gmm.h"
#include "seqGmm.h"
#include "parallelGmm.h"
#include "util.h"

int main(int argc, char** argv) {
	const size_t minPointDim = 1;
	const size_t maxPointDim = 64;

	const size_t numPoints = 1024;
	const size_t numComponents = 1;

	const size_t numSamples = 10;

	double* X = (double*) malloc(numPoints * maxPointDim * sizeof(double));
	for(size_t i = 0; i < numPoints * maxPointDim; ++i) {
		X[i] = rand() / (double)RAND_MAX;
	}

	struct timeval start, end;

	fprintf(stdout, "#numPoints numComponents pointDim seqElapsedSec parallelElapsedSec\n");
	for(size_t sample = 0; sample < numSamples; ++sample) {	
		for(size_t pointDim = minPointDim; pointDim < maxPointDim; pointDim += 1) {
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
