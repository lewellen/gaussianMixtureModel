#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#include <stdlib.h>

void choleskyDecomposition(
	double* A, double* L, size_t N); 

void solvePositiveSemidefinite(
	double* B, size_t numPoints, double* L, size_t N, double* X); 

#endif
