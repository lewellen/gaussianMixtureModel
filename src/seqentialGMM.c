#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "datFile.h"
#include "gmm.h"

void usage(const char* programName) {
	assert(programName != NULL);
	fprintf(stdout, "%s <train.dat> <numComponents>\n", programName);
}

int strToPositiveInteger(const char* str, size_t* value) {
	assert(value != NULL);
	*value = 0;

	if(str == NULL || str[0] == '\0' || str[0] == '-') {
		return 0;
	}

	errno = 0;
	*value = strtoul(str, NULL, 10);
	if(errno != 0 || *value == 0) {
		return 0;
	}
	
	return 1;
}

int main(int argc, char** argv) {
	if(argc != 3) {
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	size_t numMixtures;
	if(!strToPositiveInteger(argv[2], &numMixtures)) {
		fprintf(stdout, "Expected numComponents to be a positive integer.\n");
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	size_t numPoints = 0, pointDim = 0;
	double* data = parseDatFile(argv[1], &numPoints, &pointDim);
	if(data == NULL) {
		return EXIT_FAILURE;
	}

	if(numPoints < numMixtures) {
		fprintf(stdout, "Number of mixtures should be less than or equal to number of points.\n");
		free(data);
		return EXIT_FAILURE;
	}

	struct GMM* gmm = fit(data, numPoints, pointDim, numMixtures);

	fprintf(stdout, "numPoints: %zu, pointDim: %zu\n", numPoints, pointDim);
	fprintf(stdout, "numMixtures: %zu\n", numMixtures);
	for (size_t mixture = 0; mixture < gmm->numMixtures; ++mixture) {
		fprintf(stdout, "Mixture %zu:\n", mixture);

		fprintf(stdout, "\ttau: %.3f\n", gmm->tau[mixture]);

		fprintf(stdout, "\tmu: ");
		for (size_t dim = 0; dim < gmm->pointDim; ++dim)
			fprintf(stdout, "%.3f ", gmm->mu[mixture * pointDim + dim]);

		fprintf(stdout, "\n\tsigma: ");
		for (size_t dim = 0; dim < gmm->pointDim * gmm->pointDim; ++dim)
			fprintf(stdout, "%.3f ", gmm->sigma[mixture * gmm->pointDim * gmm->pointDim + dim]);

		fprintf(stdout, "\n\n");
	}

	freeGMM(gmm);

	free(data);

	return EXIT_SUCCESS;
}
