#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "linearAlgebra.h"

void choleskyDecomposition(double* A, double* L, size_t N) {
	// Matrix A is positive semidefinite, perform Cholesky decomposition A = LL^T
	// p. 157-158., Cholesky Factorization, 4.2 LU and Cholesky Factorizations, Numerical Analysis by Kincaid, Cheney.

	for (size_t k = 0; k < N; ++k) {
		double sum = 0;
		for (int s = 0; s < k; ++s)
			sum += L[k * N + s] * L[k * N + s];

		sum = A[k * N + k] - sum;
		if (sum <= 0) {
			fprintf(stdout, "A must be positive definite.\n");
			exit(1);
			break;
		}

		L[k * N + k] = sqrt(sum);
		for (int i = k + 1; i < N; ++i) {
			double subsum = 0;
			for (int s = 0; s < k; ++s)
				subsum += L[i * N + s] * L[k * N + s];

			L[i * N + k] = (A[i * N + k] - subsum) / L[k * N + k];
		}
	}
}

void solvePositiveSemidefinite(double* B, size_t numPoints, double* L, size_t N, double* X) {
	// Want to solve the system given by: L(L^T)X = B where:
	// 	L: N x N lower diagonal matrix
	//	X: numPoints x N unknown
	//	B: numPoints x N known
	//
	// Solve by first finding L Z = B, then L^T X = Z

	double* Z = (double*)calloc(numPoints * N, sizeof(double));

	// 2015-09-23 GEL play the access of L into L(F)orward and L(B)ackward. 
	// Found that sequential access improved runtime. 2017-03-24 GEL basically
	// pretend to carry out the forward and backward solvers, but to improve
	// runtime, load in L in sequential order ahead of time, so second time
	// around, we will have cached that data so the CPU will prefetch as needed.
	double* LF = (double*)malloc(N * N * sizeof(double));
	for (size_t i = 0, lf = 0; i < N; i++) {
		if(i > 0)
			for (size_t j = 0; j <= i - 1; j++)
				LF[lf++] = L[i * N + j];

		LF[lf++] = L[i * N + i];
	}

	double* LB = (double*)malloc(N*N*sizeof(double));
	for(size_t i = 0, lb = 0; i < N; ++i) {
		size_t ip = N - 1 - i;
		for (size_t j = ip + 1; j < N; j++)
			LB[lb++] = L[j * N + ip];

		LB[lb++] = L[ip * N + ip];
	}

	/// Use forward subsitution to solve lower triangular matrix system Lz = b.
	/// p. 150., Easy-to-Solve Systems, 4.2 LU and Cholesky Factorizations, Numerical Analysis by Kincaid, Cheney.</remarks>
	for (size_t point = 0; point < numPoints; ++point) {
		double* b = &(B[point * N]);
		double* z = &(Z[point * N]);

		for (size_t i = 0, lf = 0; i < N; i++) {
			double sum = 0.0;
			if(i > 0)
				for (size_t j = 0; j <= i - 1; j++)
					sum += LF[lf++] * z[j];

			z[i] = (b[i] - sum) / LF[lf++];
		}
	}

	// use backward subsitution to solve L^T x = z
	// p. 150., Easy-to-Solve Systems, 4.2 LU and Cholesky Factorizations, Numerical Analysis by Kincaid, Cheney.
	for (size_t point = 0; point < numPoints; ++point) {
		double* z = &(Z[point * N]);
		double* x = &(X[point * N]);

		for (size_t i = 0, lb = 0; i < N; i++) {
			size_t ip = N - 1 - i;

			double sum = 0;
			for (size_t j = ip + 1; j < N; j++)
				// Want A^T so switch switch i,j
				sum += LB[lb++] * x[j];

			x[ip] = (z[ip] - sum) / LB[lb++];
		}
	}

	free(LF);
	free(LB);
	free(Z);
}
