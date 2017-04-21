#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "datFile.h"
#include "gmm.h"
#include "seqGmm.h"
#include "util.h"

void usage(const char* programName) {
	assert(programName != NULL);
	fprintf(stdout, "%s <train.dat> <numComponents>\n", programName);
}

int main(int argc, char** argv) {
	if(argc != 3) {
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	size_t numComponents;
	if(!strToPositiveInteger(argv[2], &numComponents)) {
		fprintf(stdout, "Expected numComponents to be a positive integer.\n");
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	size_t numPoints = 0, pointDim = 0;
	double* data = parseDatFile(argv[1], &numPoints, &pointDim);
	if(data == NULL) {
		return EXIT_FAILURE;
	}

	if(numPoints < numComponents) {
		fprintf(stdout, "Number of components should be less than or equal to number of points.\n");
		free(data);
		return EXIT_FAILURE;
	}

	struct timeval start, stop;
	gettimeofday(&start, NULL);

	struct GMM* gmm = fit(data, numPoints, pointDim, numComponents, 100);

	gettimeofday(&stop, NULL);
	double elapsedSec = calcElapsedSec(&start, &stop);

	fprintf(stdout, "{\n");
	fprintf(stdout, "\"file\": \"%s\",\n", argv[1]);
	fprintf(stdout, "\"elapsedSec\": %.6f,\n", elapsedSec);
	fprintf(stdout, "\"model\": ");
	printGmmToConsole(gmm);
	fprintf(stdout, "}\n");

	freeGMM(gmm);
	free(data);

	return EXIT_SUCCESS;
}
