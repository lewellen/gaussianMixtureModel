#ifndef GMM_H
#define GMM_H

#include <stdlib.h>

struct GMM {
	// Dimension of the data (each data point is a vector \in R^{pointDim})
	size_t pointDim;

	// Number of components
	size_t numMixtures;

	// Component weights: numMixtures x 1
	double* tau;

	// Component means: numMixtures x pointDim
	double* mu;

	// Component covariances: numMixtures x pointDim^2
	double* sigma;

	// Component (lower triangular) covariance: numMixtures x pointDim^2
	double* sigmaL;

	// Leading normalizer on each component so prob integrates to 1: numMixtures x 1
	double* normalizer;
};

struct GMM* initGMM(double* X, size_t numPoints, size_t numMixtures, size_t pointDim); 

void prepareCholesky(struct GMM* gmm); 

void freeGMM(struct GMM* gmm);

void mvNormDist(double* X, size_t numPoints, struct GMM* gmm, size_t mixture, double* P); 

double logLikelihood(double* prob, size_t numPoints, struct GMM* gmm); 

struct GMM* fit(double* X, size_t numPoints, size_t pointDim, size_t numMixtures);
 
#endif
