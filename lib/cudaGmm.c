#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gmm.h"
#include "cudaGmm.h"
#include "util.h"

#include "cudaWrappers.h"

struct GMM* cudaFit(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numComponents
) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(numComponents > 0);
	
	struct GMM* gmm = initGMM(X, numPoints, pointDim, numComponents);

	double* pi = (double*) malloc(numComponents * sizeof(double));
	double* Mu = (double*) malloc(numComponents * pointDim * sizeof(double));
	double* Sigma = (double*) malloc(numComponents * pointDim * pointDim * sizeof(double));	
	double* SigmaL = (double*) malloc(numComponents * pointDim * pointDim * sizeof(double));
	double* normalizers = (double*) malloc(numComponents * sizeof(double));

	for(size_t k = 0; k < numComponents; ++k) {
		struct Component* c = & gmm->components[k];

		pi[k] = c->pi;
		memcpy(&Mu[k * pointDim], c->mu, pointDim * sizeof(double));
		memcpy(&Sigma[k * pointDim * pointDim], c->sigma, pointDim * pointDim * sizeof(double));
		memcpy(&SigmaL[k * pointDim * pointDim], c->sigmaL, pointDim * pointDim * sizeof(double));
		normalizers[k] = c->normalizer;
	}

	gpuGmmFit(
		X,
		numPoints, pointDim, numComponents,
		pi, Mu, Sigma,
		SigmaL, normalizers
	);

	for(size_t k = 0; k < numComponents; ++k) {
		struct Component* c = & gmm->components[k];

		c->pi = pi[k];
		memcpy(c->mu, &Mu[k * pointDim], pointDim * sizeof(double));
		memcpy(c->sigma, &Sigma[k * pointDim * pointDim], pointDim * pointDim * sizeof(double));
		memcpy(c->sigmaL, &SigmaL[k * pointDim * pointDim], pointDim * pointDim * sizeof(double));
		c->normalizer = normalizers[k];
	}

	free(normalizers);
	free(SigmaL);
	free(Sigma);
	free(Mu);
	free(pi);

	return gmm;
}
