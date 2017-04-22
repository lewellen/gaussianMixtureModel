#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kmeans.h"
#include "linearAlgebra.h"

void kmeans(const double* X, const size_t numPoints, const size_t pointDim, double* M, const size_t numComponents) {
	assert(X != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(M != NULL);
	assert(numComponents > 0);

	const double tolerance = 1e-3;
	double diff = 0;

	const size_t maxIterations = 20;

	double MP[numComponents * pointDim];
	size_t counts[numComponents];

	for(size_t iteration = 0; iteration < maxIterations && diff > tolerance; ++iteration) {
		memset(MP, 0, numComponents * pointDim * sizeof(double));	
		memset(counts, 0, numComponents * sizeof(size_t));	

		for(size_t i = 0; i < numPoints; ++i) {
			const double* Xi = & X[i * pointDim];

			// arg min
			double minD = INFINITY;
			size_t minDk = 0;
			for(size_t k = 0; k < numComponents; ++k) {
				const double* Mk = & M[k * pointDim];
				double dist = vecDiffNorm(Xi, Mk, pointDim);
				if(minD > dist) {
					minD = dist;
					minDk = k;
				}	
			}

			vecAddInPlace(&M[minDk * pointDim], Xi, pointDim);
			++counts[minDk];
		}

		for(size_t k = 0; k < numComponents; ++k) {
			vecDivByScalar(&MP[k * pointDim], counts[k], pointDim);
		}

		diff = 0;
		for(size_t k = 0; k < numComponents; ++k) {
			diff += vecDiffNorm(&MP[k * pointDim], &M[k * pointDim], pointDim);
		}
		diff /= (double) numComponents;

		memcpy(M, MP, numComponents * pointDim * sizeof(double));
	}
}

