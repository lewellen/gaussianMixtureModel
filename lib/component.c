#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "component.h"
#include "linearAlgebra.h"

#ifndef PI
#define PI 3.141592653589793238
#endif

void prepareCovariance(struct Component* component, const size_t pointDim) {
	assert(component != NULL);

	// Perform cholesky factorization once each iteration instead of 
	// repeadily for each normDist execution.
	choleskyDecomposition(
		component->sigma, 
		pointDim, 
		component->sigmaL
	);

	// det(Sigma) = det(L L^T) = det(L)^2
	double det = 1.0;
	for (size_t i = 0; i < pointDim; ++i) {
		det *= component->sigmaL[i * pointDim + i];
	}

	det *= det;

	component->normalizer = sqrt(pow(2.0 * PI, pointDim) * det);
}

void mvNormDist(
	const struct Component* component, const size_t pointDim,
	const double* X, const size_t numPoints, 
	double* P
) {
	// 2015-09-23 GEL Through profiling (Sleepy CS), found that program was 
	// spending most of its time in this method. Decided to change from 
	// processing single point at a time to processing set of points at a time. 
	// Found this gave 1.36x improvement (~6ms -> ~4ms) on the Old Faithful 
	// dataset. Total 1.77x improvement.

	// Here we are computing the probability density function of the  multivariate
	// normal distribution conditioned on a single component for the set of points 
	// given by X.
	//
	// P(x|component) = exp{ -0.5 * (x - mu)^T Sigma^{-} (x - mu) } / sqrt{ (2pi)^k det(Sigma) }
	//
	// Where Sigma and Mu are really Sigma_{component} Mu_{component}

	assert(component != NULL);
	assert(pointDim > 0);
	assert(X != NULL);
	assert(numPoints > 0);
	assert(P != NULL);

	double* XM = (double*)malloc(numPoints * pointDim * sizeof(double));
	double* SXM = (double*)malloc(numPoints * pointDim * sizeof(double));
	double* innerProduct = (double*)malloc(numPoints * sizeof(double));

	// Let XM = (x - m)
	for (size_t point = 0; point < numPoints; ++point)
		for (size_t dim = 0; dim < pointDim; ++dim)
			XM[point * pointDim + dim] = X[point * pointDim + dim] - component->mu[dim];

	// Sigma SXM = XM => Sigma^{-} XM = SXM
	solvePositiveSemidefinite(
		component->sigmaL, 
		XM, 
		SXM,
		pointDim, 
		numPoints 
		);

	// XM^T SXM
	for (size_t point = 0; point < numPoints; ++point) {
		innerProduct[point] = 0.0;
		for (size_t dim = 0; dim < pointDim; ++dim) {
			innerProduct[point] += XM[point * pointDim + dim] * SXM[point * pointDim + dim];
		}
	}

	// Compute P exp( -0.5 innerProduct ) / normalizer
	for (size_t point = 0; point < numPoints; ++point) {
		P[point] = exp(-0.5 * innerProduct[point]) / component->normalizer;
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
