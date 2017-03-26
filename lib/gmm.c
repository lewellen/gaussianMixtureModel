#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "linearAlgebra.h"
#include "gmm.h"

#ifndef PI
#define PI 3.141592653589793238
#endif

struct GMM* initGMM(double* X, size_t numPoints, size_t numMixtures, size_t pointDim) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(numMixtures > 0);
	assert(pointDim > 0);

	// X is an numPoints x pointDim set of training data

	struct GMM* gmm = (struct GMM*)checkedCalloc(1, sizeof(struct GMM));
	gmm->pointDim = pointDim;
	gmm->numMixtures = numMixtures;

	// Initial guesses

	// Assume every mixture has uniform weight
	double uniformTau = 1.0 / numMixtures;
	gmm->tau = (double*)checkedCalloc(numMixtures, sizeof(double));
	for (size_t mixture = 0; mixture < numMixtures; ++mixture)
		gmm->tau[mixture] = uniformTau;

	// Use a random point for mean of each mixture
	gmm->mu = (double*)checkedCalloc(numMixtures * pointDim, sizeof(double));
	for (size_t mixture = 0; mixture < numMixtures; ++mixture) {
		size_t j = rand() % numPoints;
		for (size_t dim = 0; dim < pointDim; ++dim)
			gmm->mu[mixture * pointDim + dim] = X[j * pointDim + dim];
	}

	// Use identity covariance- assume dimensions are independent
	gmm->sigma = (double*)checkedCalloc(numMixtures * pointDim * pointDim, sizeof(double));
	for (size_t mixture = 0; mixture < numMixtures; ++mixture)
		for (size_t j = 0; j < pointDim; ++j)
			gmm->sigma[mixture * pointDim * pointDim + j * pointDim + j] = 1;

	gmm->sigmaL = (double*)checkedCalloc(numMixtures * pointDim * pointDim, sizeof(double));
	gmm->normalizer = (double*)checkedCalloc(numMixtures, sizeof(double));

	prepareCovariances(gmm);

	return gmm;
}

void prepareCovariances(struct GMM* gmm) {
	size_t blockSize = gmm->pointDim * gmm->pointDim;

	// Perform cholesky factorization once each iteration instead of 
	// repeadily for each normDist execution.
	for (size_t mixture = 0; mixture < gmm->numMixtures; ++mixture) {
		choleskyDecomposition(
			&(gmm->sigma[mixture * blockSize]), 
			gmm->pointDim, 
			&(gmm->sigmaL[mixture * blockSize])
		);
	}

	// det(Sigma) = det(L L^T) = det(L)^2
	for (size_t mixture = 0; mixture < gmm->numMixtures; ++mixture) {
		double det = 1.0;
		for (size_t i = 0; i < gmm->pointDim; ++i) {
			det *= gmm->sigmaL[mixture * blockSize + i * gmm->pointDim + i];
		}

		det *= det;

		gmm->normalizer[mixture] = sqrt(pow(2.0 * PI, gmm->pointDim) * det);
	}
}

void freeGMM(struct GMM* gmm) {
	free(gmm->tau);
	free(gmm->mu);
	free(gmm->sigma);
	free(gmm->sigmaL);
	free(gmm->normalizer);
	free(gmm);
}


void mvNormDist(double* X, size_t numPoints, struct GMM* gmm, size_t mixture, double* P) {
	// 2015-09-23 GEL Through profiling (Sleepy CS), found that program was 
	// spending most of its time in this method. Decided to change from 
	// processing single point at a time to processing set of points at a time. 
	// Found this gave 1.36x improvement (~6ms -> ~4ms) on the Old Faithful 
	// dataset. Total 1.77x improvement.

	// Here we are computing the probability density function of the  multivariate
	// normal distribution conditioned on a single mixture for the set of points 
	// given by X.
	//
	// P(x|mixture) = exp{ -0.5 * (x - mu)^T Sigma^{-} (x - mu) } / sqrt{ (2pi)^k det(Sigma) }
	//
	// Where Sigma and Mu are really Sigma_{mixture} Mu_{mixture}

	assert(X != NULL);
	assert(numPoints > 0);
	assert(gmm != NULL);
	assert(P != NULL);

	double* XM = (double*)malloc(numPoints * gmm->pointDim * sizeof(double));
	double* SXM = (double*)malloc(numPoints * gmm->pointDim * sizeof(double));
	double* innerProduct = (double*)malloc(numPoints * sizeof(double));

	double* mu = &(gmm->mu[mixture * gmm->pointDim]);
	double* sigmaL = &(gmm->sigmaL[mixture * gmm->pointDim * gmm->pointDim]);

	// Let XM = (x - m)
	for (size_t point = 0; point < numPoints; ++point)
		for (size_t dim = 0; dim < gmm->pointDim; ++dim)
			XM[point * gmm->pointDim + dim] = X[point * gmm->pointDim + dim] - mu[dim];

	// Sigma SXM = XM => Sigma^{-} XM = SXM
	solvePositiveSemidefinite(
		sigmaL, 
		XM, 
		SXM,
		gmm->pointDim, 
		numPoints 
		);

	// XM^T SXM
	for (size_t point = 0; point < numPoints; ++point) {
		innerProduct[point] = 0.0;
		for (size_t dim = 0; dim < gmm->pointDim; ++dim) {
			innerProduct[point] += XM[point * gmm->pointDim + dim] * SXM[point * gmm->pointDim + dim];
		}
	}

	// Compute P exp( -0.5 innerProduct ) / normalizer
	for (size_t point = 0; point < numPoints; ++point) {
		P[point] = exp(-0.5 * innerProduct[point]) / gmm->normalizer[mixture];
		if (P[point] < 1e-8) {
			P[point] = 0.0;
		}

		if (1.0 - P[point] < 1e-8) {
			P[point] = 1.0;
		}
	}

	free(XM);
	free(SXM);
	free(innerProduct);
}

double logLikelihood(double* prob, size_t numPoints, struct GMM* gmm) {
	double logL = 0.0;
	for (size_t point = 0; point < numPoints; ++point) {
		double inner = 0.0;
		for (size_t mixture = 0; mixture < gmm->numMixtures; ++mixture)
			inner += gmm->tau[mixture] * prob[mixture * numPoints + point];

		logL += log(inner);
	}

	return logL;
}

struct GMM* fit(double* X, size_t numPoints, size_t pointDim, size_t numMixtures) {
	struct GMM* gmm = initGMM(X, numPoints, numMixtures, pointDim);

	size_t maxIterations = 100;
	double tolerance = 1e-8;
	double prevLogL = -INFINITY;
	double currentLogL = -INFINITY;

	double* prob = (double*)checkedCalloc(numPoints * numMixtures, sizeof(double));
	double* margin = (double*)checkedCalloc(numMixtures, sizeof(double));

	double* xm = (double*)checkedCalloc(pointDim, sizeof(double));
	double* outerProduct = (double*)checkedCalloc(pointDim * pointDim, sizeof(double));

	do {
		// --- E-Step ---

		// Compute T
		for (size_t mixture = 0; mixture < numMixtures; ++mixture)
			mvNormDist(X, numPoints, gmm, mixture, &(prob[mixture * numPoints]));

		for (size_t point = 0; point < numPoints; ++point) {
			double sum = 0.0;
			for (size_t mixture = 0; mixture < numMixtures; ++mixture)
				sum += prob[mixture * numPoints + point];

			if (sum > tolerance)
				for (size_t mixture = 0; mixture < numMixtures; ++mixture)
					prob[numPoints * mixture + point] /= sum;
		}

		// 2015-09-20 GEL Eliminated redundant mvNorm clac in logLikelihood by 
		// passing in precomputed prob values. Also moved loop termination here
		// since likelihood determines termination. Result: 1.3x improvement in 
		// execution time.  (~8 ms to ~6 ms on oldFaithful.dat)
		prevLogL = currentLogL;
		currentLogL = logLikelihood(prob, numPoints, gmm);
		if (--maxIterations == 0 || !(maxIterations < 80 ? currentLogL > prevLogL : 1 == 1))
			break;

		// Let U[mixture] = \Sum_i T[mixture, i]
		memset(margin, 0, numMixtures * sizeof(double));
		for (size_t mixture = 0; mixture < numMixtures; ++mixture)
			for (size_t point = 0; point < numPoints; ++point)
				margin[mixture] += prob[mixture * numPoints + point];

		double normTerm = 0;
		for (size_t mixture = 0; mixture < numMixtures; ++mixture)
			normTerm += margin[mixture];

		// --- M-Step ---

		// Update tau
		for (size_t mixture = 0; mixture < numMixtures; ++mixture)
			gmm->tau[mixture] = margin[mixture] / normTerm;

		// Update mu
		memset(gmm->mu, 0, numMixtures * pointDim * sizeof(double));
		for (size_t mixture = 0; mixture < numMixtures; ++mixture)
			for (size_t point = 0; point < numPoints; ++point)
				for (size_t dim = 0; dim < pointDim; ++dim)
					gmm->mu[mixture * pointDim + dim] += prob[mixture * numPoints + point] * X[point * pointDim + dim];

		for (size_t mixture = 0; mixture < numMixtures; ++mixture)
			for (size_t dim = 0; dim < pointDim; ++dim)
				gmm->mu[mixture * pointDim + dim] /= margin[mixture];

		// Update sigma
		memset(gmm->sigma, 0, numMixtures * pointDim * pointDim * sizeof(double));
		for (size_t mixture = 0; mixture < numMixtures; ++mixture) {
			for (size_t point = 0; point < numPoints; ++point) {
				// (x - m)
				for (size_t dim = 0; dim < pointDim; ++dim)
					xm[dim] = X[point * pointDim + dim] - gmm->mu[mixture * pointDim + dim];

				// (x - m) (x - m)^T
				for (size_t row = 0; row < pointDim; ++row)
					for (size_t column = 0; column < pointDim; ++column)
						outerProduct[row * pointDim + column] = xm[row] * xm[column];

				for (size_t i = 0; i < pointDim * pointDim; ++i)
					gmm->sigma[mixture * pointDim * pointDim + i] += prob[mixture * numPoints + point] * outerProduct[i];
			}

			for (size_t i = 0; i < pointDim * pointDim; ++i)
				gmm->sigma[mixture * pointDim * pointDim + i] /= margin[mixture];
		}

		prepareCovariances(gmm);

	} while (1 == 1);

	free(prob);
	free(margin);

	free(xm);
	free(outerProduct);

	return gmm;
}
