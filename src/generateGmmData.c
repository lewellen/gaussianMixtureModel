#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gmm.h"
#include "util.h"

void usage(const char* progName) {
	fprintf(stdout, "%s <numPoints> <pointDim> <numComponents>\n", progName);
}

int main(int argc, char** argv) {
	if(argc != 4) {
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	const size_t numParams = 3;
	const char* paramNames[] = { "numPoints", "pointDim", "numComponents" };
	size_t paramValues[numParams];

	memset(paramValues, 0, numParams * sizeof(size_t));
	for(size_t i = 1; i < 1 + numParams; ++i) {
		if(!strToPositiveInteger(argv[i], & paramValues[i-1] ) || paramValues[i-1] == 0) {
			fprintf(stdout, "%s '%s' needs to be a positive integer.\n", 
				paramNames[i - 1], argv[i]);
			usage(argv[0]);
			return EXIT_FAILURE;
		}
	}

	const size_t numPoints = paramValues[0];
	const size_t pointDim = paramValues[1];
	const size_t numComponents = paramValues[2];

	srand(time(0));

	double* X = generateGmmData(numPoints, pointDim, numComponents);

	for(size_t i = 0; i < numPoints; ++i) {
		for(size_t j = 0; j < pointDim; ++j) {
			fprintf(stdout, "%f ", X[i * pointDim + j]);
		}
		fprintf(stdout, "\n");
	}

	free(X);

	return EXIT_SUCCESS;
}
