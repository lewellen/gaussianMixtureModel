#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#include <stdlib.h>

void choleskyDecomposition(
	const double* A, const size_t pointDim, 
	double* L
	);

void solvePositiveSemidefinite(
	const double* L, const double* B, 
	double* X, 
	const size_t pointDim, const size_t numPoints
	);

#endif
