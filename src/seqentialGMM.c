#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "datFile.h"
#include "gmm.h"

void usage(const char* programName) {
	fprintf(stdout, "%s <train.dat> <numComponents>\n", programName);
}

int main(int argc, char** argv) {
	if(argc != 3) {
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	errno = 0;
	size_t numMixtures = strtoul(argv[2], NULL, 0);
	if(errno != 0) {
		perror("Expected numComponents to be a positive integer.");
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	size_t numPoints = 0, pointDim = 0;
	double* data = parseDatFile(argv[1], &numPoints, &pointDim);
	if(data == NULL) {
		return EXIT_FAILURE;
	}

	struct GMM* gmm = fit(data, numPoints, pointDim, numMixtures);

	for (size_t mixture = 0; mixture < gmm->numMixtures; ++mixture) {
		fprintf(stdout, "Mixture %zu:\n", mixture);

		fprintf(stdout, "\ttau: %.3f\n", gmm->tau[mixture]);

		fprintf(stdout, "\tmu: ");
		for (size_t dim = 0; dim < gmm->pointDim; ++dim)
			fprintf(stdout, "%.2f ", gmm->mu[mixture * pointDim + dim]);

		fprintf(stdout, "\n\tsigma: ");
		for (size_t dim = 0; dim < gmm->pointDim * gmm->pointDim; ++dim)
			fprintf(stdout, "%.2f ", gmm->sigma[mixture * gmm->pointDim * gmm->pointDim + dim]);

		fprintf(stdout, "\n\n");
	}

	freeGMM(gmm);

	free(data);

	return EXIT_SUCCESS;
}
