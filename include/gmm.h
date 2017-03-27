#ifndef GMM_H
#define GMM_H

#include <stdlib.h>

// There is an implict struct Component to all the GMM entries. Decided to leave
// everything coalesced since those values are accessed sequentially rather than
// by component throughout most of the code.

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

struct GMM* fit(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numMixtures
	);

struct GMM* initGMM(
	const double* X, 
	const size_t numPoints, 
	const size_t pointDim, 
	const size_t numMixtures
	); 

void prepareCovariances(struct GMM* gmm); 

void freeGMM(struct GMM* gmm);

void mvNormDist(
	const struct GMM* gmm, const size_t mixture, 
	const double* X, const size_t numPoints, 
	double* P
); 

double logLikelihood(
	const struct GMM* gmm,
	const double* prob, const size_t numPoints
); 
 
#endif
