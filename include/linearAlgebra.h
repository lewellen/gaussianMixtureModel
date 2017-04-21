#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#include <stdlib.h>

void choleskyDecomposition(
	const double* A, const size_t pointDim, 
	double* L
);

void solvePositiveDefinite(
	const double* L, const double* B, 
	double* X, 
	const size_t pointDim, const size_t numPoints
);

void lowerDiagByVector(
	const double* L,
	const double* x,
	double* b,
	const size_t n
);

void vectorAdd(
	const double* a,
	const double* b,
	double* c,
	const size_t n
); 

void vecAddInPlace(
	double* a, 
	const double* b, 
	const size_t D
);

void vecDivByScalar(
	double* a, 
	const double b, 
	const size_t D
);

double vecDiffNorm(
	const double* a, 
	const double* b, 
	const size_t D
);

#endif
