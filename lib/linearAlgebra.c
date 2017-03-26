#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "linearAlgebra.h"

void choleskyDecomposition(double* A, size_t pointDim, double* L) {
	// p. 157-158., Cholesky Factorization, 4.2 LU and Cholesky Factorizations, 
	// Numerical Analysis by Kincaid, Cheney.

	// A is a real, symmetric, and positive definite pointDim x pointDim matrix
	assert(pointDim > 0);
	assert(A != NULL);
	for(size_t i = 0; i < pointDim; ++i) {
		for(size_t j = 0; j < pointDim; ++j) {
			double a = A[i * pointDim + j];
			if(a != a || fabs(a) == INFINITY) {
				fprintf(stdout, "A[%zu, %zu] = %f should be real value\n", i,j,a);
				assert(0);
			}

			double b = A[j * pointDim + i];
			if(a != b) {
				fprintf(stdout, "A[%zu, %zu] should be symmetric (%f != %f)\n", i, j, a, b);
				assert(0);
			}
		}
	}
	
	// L is the resulting lower diagonal portion of A = LL^T
	assert(L != NULL);
	memset(L, 0, sizeof(double) * pointDim * pointDim);

	for (size_t k = 0; k < pointDim; ++k) {
		double sum = 0;
		for (int s = 0; s < k; ++s) {
			sum += L[k*pointDim+s] * L[k*pointDim+s];
		}

		sum = A[k * pointDim + k] - sum;
		if (sum <= 0) {
			fprintf(stdout, "A must be positive definite.\n");
			exit(1);
			break;
		}

		L[k * pointDim + k] = sqrt(sum);
		for (int i = k + 1; i < pointDim; ++i) {
			double subsum = 0;
			for (int s = 0; s < k; ++s)
				subsum += L[i * pointDim + s] * L[k * pointDim + s];

			L[i * pointDim + k] = (A[i * pointDim + k] - subsum) / L[k * pointDim + k];
		}
	}
}

void solvePositiveSemidefinite(const double* L, const double* B, double* X, const size_t pointDim, const size_t numPoints) {
	// Want to solve the system given by: L(L^T)X = B where:
	// 	L: pointDim x pointDim lower diagonal matrix
	//	X: pointDim x numPoints unknown
	//	B: pointDim x numPoints known
	//
	// Solve by first finding L Z = B, then L^T X = Z

	double* Z = (double*)calloc(numPoints * pointDim, sizeof(double));

	// 2015-09-23 GEL play the access of L into L(F)orward and L(B)ackward. 
	// Found that sequential access improved runtime. 2017-03-24 GEL basically
	// pretend to carry out the forward and backward solvers, but to improve
	// runtime, load in L in sequential order ahead of time, so second time
	// around, we will have cached that data so the CPU will prefetch as needed.
	double* LF = (double*)malloc(pointDim * pointDim * sizeof(double));
	for (size_t i = 0, lf = 0; i < pointDim; i++) {
		if(i > 0) {
			for (size_t j = 0; j <= i - 1; j++) {
				LF[lf++] = L[i * pointDim + j];
			}
		}

		LF[lf++] = L[i * pointDim + i];
	}

	double* LB = (double*)malloc(pointDim * pointDim * sizeof(double));
	for(size_t i = 0, lb = 0; i < pointDim; ++i) {
		size_t ip = pointDim - 1 - i;
		for (size_t j = ip + 1; j < pointDim; j++) {
			LB[lb++] = L[j * pointDim + ip];
		}

		LB[lb++] = L[ip * pointDim + ip];
	}

	// Use forward subsitution to solve lower triangular matrix system Lz = b.
	// p. 150., Easy-to-Solve Systems, 4.2 LU and Cholesky Factorizations, Numerical Analysis by Kincaid, Cheney.
	for (size_t point = 0; point < numPoints; ++point) {
		double* b = &(B[point * pointDim]);
		double* z = &(Z[point * pointDim]);

		for (size_t i = 0, lf = 0; i < pointDim; i++) {
			double sum = 0.0;
			if(i > 0) {
				for (size_t j = 0; j <= i - 1; j++) {
					sum += LF[lf++] * z[j];
				}
			}

			z[i] = (b[i] - sum) / LF[lf++];
		}
	}

	// use backward subsitution to solve L^T x = z
	// p. 150., Easy-to-Solve Systems, 4.2 LU and Cholesky Factorizations, Numerical Analysis by Kincaid, Cheney.
	for (size_t point = 0; point < numPoints; ++point) {
		double* z = &(Z[point * pointDim]);
		double* x = &(X[point * pointDim]);

		for (size_t i = 0, lb = 0; i < pointDim; i++) {
			size_t ip = pointDim - 1 - i;

			double sum = 0;
			for (size_t j = ip + 1; j < pointDim; j++)
				// Want A^T so switch switch i,j
				sum += LB[lb++] * x[j];

			x[ip] = (z[ip] - sum) / LB[lb++];
		}
	}

	free(LF);
	free(LB);
	free(Z);
}
