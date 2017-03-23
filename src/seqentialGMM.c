#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PI 3.141592653589793238

struct GMM {
	int pointDim;
	int numMixtures;

	double* tau;
	double* mu;
	double* sigma;

	double* sigmaL;
	double* normalizer;
	double* normBuff;
};

void prepareCholesky(struct GMM* gmm);

void* checkedCalloc(size_t count, size_t size) {
	void* result = calloc(count, size);
	if (result == NULL) {
		perror("Failed to allocate memory.");
		exit(1);
	}

	return result;
}

struct GMM* initGMM(double* X, int numPoints, int numMixtures, int pointDim) {
	struct GMM* gmm = (struct GMM*)checkedCalloc(1, sizeof(struct GMM));
	gmm->pointDim = pointDim;
	gmm->numMixtures = numMixtures;

	// Initial guesses
	double uniformTau = 1.0 / numMixtures;
	gmm->tau = (double*)checkedCalloc(numMixtures, sizeof(double));
	for (int mixture = 0; mixture < numMixtures; ++mixture)
		gmm->tau[mixture] = uniformTau;

	gmm->mu = (double*)checkedCalloc(numMixtures * pointDim, sizeof(double));
	for (int mixture = 0; mixture < numMixtures; ++mixture) {
		int j = rand() % numPoints;
		for (int dim = 0; dim < pointDim; ++dim)
			gmm->mu[mixture * pointDim + dim] = X[j * pointDim + dim];
	}

	gmm->sigma = (double*)checkedCalloc(numMixtures * pointDim * pointDim, sizeof(double));
	for (int mixture = 0; mixture < numMixtures; ++mixture)
		for (int j = 0; j < pointDim; ++j)
			gmm->sigma[mixture * pointDim * pointDim + j * pointDim + j] = 1;

	gmm->sigmaL = (double*)checkedCalloc(numMixtures * pointDim * pointDim, sizeof(double));
	gmm->normalizer = (double*)checkedCalloc(numMixtures, sizeof(double));
	gmm->normBuff = (double*)checkedCalloc(3 * pointDim, sizeof(double));

	prepareCholesky(gmm);

	return gmm;
}

void freeGMM(struct GMM* gmm) {
	free(gmm->tau);
	free(gmm->mu);
	free(gmm->sigma);
	free(gmm->sigmaL);
	free(gmm->normalizer);
	free(gmm->normBuff);
	free(gmm);
}

void choleskyDecomposition(double* A, double* L, int N) {
	// Matrix A is positive semidefinite, perform Cholesky decomposition A = LL^T
	// p. 157-158., Cholesky Factorization, 4.2 LU and Cholesky Factorizations, Numerical Analysis by Kincaid, Cheney.

	for (int k = 0; k < N; ++k) {
		double sum = 0;
		for (int s = 0; s < k; ++s)
			sum += L[k * N + s] * L[k * N + s];

		sum = A[k * N + k] - sum;
		if (sum <= 0) {
			//throw new std::invalid_argument("A must be positive definite.");
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

void prepareCholesky(struct GMM* gmm) {
	int blockSize = gmm->pointDim * gmm->pointDim;

	// Perform cholesky factorization once each iteration instead of 
	// repeadily for each normDist execution.
	for (int mixture = 0; mixture < gmm->numMixtures; ++mixture)
		choleskyDecomposition(&(gmm->sigma[mixture * blockSize]), &(gmm->sigmaL[mixture * blockSize]), gmm->pointDim);

	// det(Sigma) = det(L L^T) = det(L)^2
	for (int mixture = 0; mixture < gmm->numMixtures; ++mixture) {
		double det = 1.0;
		for (int i = 0; i < gmm->pointDim; ++i)
			det *= gmm->sigmaL[mixture * blockSize + i * gmm->pointDim + i];

		det *= det;

		gmm->normalizer[mixture] = sqrt(pow(2.0 * PI, gmm->pointDim) * det);
	}
}

void solvePositiveSemidefinite(double* B, int numPoints, struct GMM* gmm, int mixture, double* X) {
	int N = gmm->pointDim;
	double* L = &(gmm->sigmaL[N * N * mixture]);
	double* Z = (double*)calloc(numPoints * gmm->pointDim, sizeof(double));

	// 2015-09-23 GEL play the access of L into L(F)orward and L(B)ackward. 
	// Found that sequential access improved runtime.
	double* LF = (double*)malloc(N * N * sizeof(double));
	for (int i = 0, lf = 0; i < N; i++) {
		double sum = 0;
		for (int j = 0; j <= i - 1; j++)
			LF[lf++] = L[i * N + j];

		LF[lf++] = L[i * N + i];
	}

	double* LB = (double*)malloc(N*N*sizeof(double));
	for (int i = N - 1, lb = 0; i >= 0; i--) {
		double sum = 0;
		for (int j = i + 1; j < N; j++)
			LB[lb++] = L[j * N + i];

		LB[lb++] = L[i * N + i];
	}


	/// Use forward subsitution to solve lower triangular matrix system Lz = b.
	/// p. 150., Easy-to-Solve Systems, 4.2 LU and Cholesky Factorizations, Numerical Analysis by Kincaid, Cheney.</remarks>
	for (int point = 0; point < numPoints; ++point) {
		double* b = &(B[point * gmm->pointDim]);
		double* z = &(Z[point * gmm->pointDim]);

		for (int i = 0, lf = 0; i < N; i++) {
			double sum = 0.0;
			for (int j = 0; j <= i - 1; j++)
				sum += LF[lf++] * z[j];

			z[i] = (b[i] - sum) / LF[lf++];
		}
	}

	// use backward subsitution to solve L^T x = z
	// p. 150., Easy-to-Solve Systems, 4.2 LU and Cholesky Factorizations, Numerical Analysis by Kincaid, Cheney.
	for (int point = 0; point < numPoints; ++point) {
		double* z = &(Z[point * gmm->pointDim]);
		double* x = &(X[point * gmm->pointDim]);

		for (int i = N - 1, lb = 0; i >= 0; i--) {
			double sum = 0;
			for (int j = i + 1; j < N; j++)
				// Want A^T so switch switch i,j
				sum += LB[lb++] * x[j];

			x[i] = (z[i] - sum) / LB[lb++];
		}
	}

	free(LF);
	free(LB);
	free(Z);
}

void mvNormDist(double* X, int numPoints, struct GMM* gmm, int mixture, double* P) {
	// 2015-09-23 GEL Through profiling (Sleepy CS), found that program was 
	// spending most of its time in this method. Decided to change from 
	// processing single point at a time to processing set of points at a time. 
	// Found this gave 1.36x improvement (~6ms -> ~4ms) on the Old Faithful 
	// dataset. Total 1.77x improvement.

	double* XM = (double*)malloc(numPoints * gmm->pointDim * sizeof(double));
	double* SXM = (double*)malloc(numPoints * gmm->pointDim * sizeof(double));
	double* innerProduct = (double*)malloc(numPoints * sizeof(double));

	double* mu = &(gmm->mu[mixture * gmm->pointDim]);

	// O(N)
	// (x - m)
	for (int point = 0; point < numPoints; ++point)
		for (int dim = 0; dim < gmm->pointDim; ++dim)
			XM[point * gmm->pointDim + dim] = X[point * gmm->pointDim + dim] - mu[dim];

	// --> O(N^2) <--
	// S y = (x - m)
	solvePositiveSemidefinite(XM, numPoints, gmm, mixture, SXM);

	// O(N)
	// (x - m)^T y
	for (int point = 0; point < numPoints; ++point) {
		innerProduct[point] = 0.0;
		for (int dim = 0; dim < gmm->pointDim; ++dim)
			innerProduct[point] += XM[point * gmm->pointDim + dim] * SXM[point * gmm->pointDim + dim];
	}

	// O(1)
	for (int point = 0; point < numPoints; ++point) {
		P[point] = exp(-0.5 * innerProduct[point]) / gmm->normalizer[mixture];
		if (P[point] < 1e-8)
			P[point] = 0.0;

		if (1.0 - P[point] < 1e-8)
			P[point] = 1.0;
	}

	free(XM);
	free(SXM);
	free(innerProduct);
}

double logLikelihood(double* prob, int numPoints, struct GMM* gmm) {
	double logL = 0.0;
	for (int point = 0; point < numPoints; ++point) {
		double inner = 0.0;
		for (int mixture = 0; mixture < gmm->numMixtures; ++mixture)
			inner += gmm->tau[mixture] * prob[mixture * numPoints + point];

		logL += log(inner);
	}

	return logL;
}

struct GMM* fit(double* X, int numPoints, int pointDim, int numMixtures) {
	struct GMM* gmm = initGMM(X, numPoints, pointDim, numMixtures);

	int maxIterations = 100;
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
		for (int mixture = 0; mixture < numMixtures; ++mixture)
			mvNormDist(X, numPoints, gmm, mixture, &(prob[mixture * numPoints]));

		for (int point = 0; point < numPoints; ++point) {
			double sum = 0.0;
			for (int mixture = 0; mixture < numMixtures; ++mixture)
				sum += prob[mixture * numPoints + point];

			if (sum > tolerance)
				for (int mixture = 0; mixture < numMixtures; ++mixture)
					prob[numPoints * mixture + point] /= sum;
		}

		// 2015-09-20 GEL Eliminated redundant mvNorm clac in logLikelihood by 
		// passing in precomputed prob values. Also moved loop termination here
		// since likelihood determines termination. Result: 1.3x improvement in 
		// execution time.  (~8 ms to ~6 ms on oldFaithful.dat)
		prevLogL = currentLogL;
		currentLogL = logLikelihood(prob, numPoints, gmm);
		if (--maxIterations <= 0 || !(maxIterations < 80 ? currentLogL > prevLogL : 1 == 1))
			break;

		// Let U[mixture] = \Sum_i T[mixture, i]
		memset(margin, 0, numMixtures * sizeof(double));
		for (int mixture = 0; mixture < numMixtures; ++mixture)
			for (int point = 0; point < numPoints; ++point)
				margin[mixture] += prob[mixture * numPoints + point];

		double normTerm = 0;
		for (int mixture = 0; mixture < numMixtures; ++mixture)
			normTerm += margin[mixture];

		// --- M-Step ---

		// Update tau
		for (int mixture = 0; mixture < numMixtures; ++mixture)
			gmm->tau[mixture] = margin[mixture] / normTerm;

		// Update mu
		memset(gmm->mu, 0, numMixtures * pointDim * sizeof(double));
		for (int mixture = 0; mixture < numMixtures; ++mixture)
			for (int point = 0; point < numPoints; ++point)
				for (int dim = 0; dim < pointDim; ++dim)
					gmm->mu[mixture * pointDim + dim] += prob[mixture * numPoints + point] * X[point * pointDim + dim];

		for (int mixture = 0; mixture < numMixtures; ++mixture)
			for (int dim = 0; dim < pointDim; ++dim)
				gmm->mu[mixture * pointDim + dim] /= margin[mixture];

		// Update sigma
		memset(gmm->sigma, 0, numMixtures * pointDim * pointDim * sizeof(double));
		for (int mixture = 0; mixture < numMixtures; ++mixture) {
			for (int point = 0; point < numPoints; ++point) {
				// (x - m)
				for (int dim = 0; dim < pointDim; ++dim)
					xm[dim] = X[point * pointDim + dim] - gmm->mu[mixture * pointDim + dim];

				// (x - m) (x - m)^T
				for (int row = 0; row < pointDim; ++row)
					for (int column = 0; column < pointDim; ++column)
						outerProduct[row * pointDim + column] = xm[row] * xm[column];

				for (int i = 0; i < pointDim * pointDim; ++i)
					gmm->sigma[mixture * pointDim * pointDim + i] += prob[mixture * numPoints + point] * outerProduct[i];
			}

			for (int i = 0; i < pointDim * pointDim; ++i)
				gmm->sigma[mixture * pointDim * pointDim + i] /= margin[mixture];
		}

		prepareCholesky(gmm);

	} while (1 == 1);

	free(prob);
	free(margin);

	free(xm);
	free(outerProduct);

	return gmm;
}

int getFileLength(FILE* handle, long* outFileLength) {
	if(fseek(handle, 0, SEEK_END) != 0) {
		perror("Failed to seek to end of file.");
	} else if ((*outFileLength = ftell(handle)) <= 0) {
		perror("Zero or negative file length encountered.");
	} else if (fseek(handle, 0, SEEK_SET) != 0) {
		perror("Failed to seek to start of file.");
	} else {
		return 1;
	}

	return 0;
}

char* readFile(const char* filePath, long* outFileLength) {
	char* contents = NULL;

	FILE* handle = fopen(filePath, "rb");
	if(handle == NULL) {
		perror(filePath);
	} else {
		if( getFileLength(handle, outFileLength) != 0 ) {
			contents = (char*)checkedCalloc(*outFileLength + 1, sizeof(char));
			assert(contents != NULL);

			long bytesRead = fread(contents, sizeof(char), *outFileLength, handle);
			if (bytesRead != *outFileLength) {
				perror("Number of bytes read does not match number of bytes expected.");
				if (contents != NULL) {
					free(contents);
					contents = NULL;
				}
			} else {
				contents[*outFileLength] = '\0';
			}
		}

		fclose(handle);
		handle = NULL;
	}

	return contents;
}

double* readDatFile(const char* filePath, int* numPoints) {
	long fileLength = 0;
	char* contents = readFile(filePath, &fileLength);
	if(contents == NULL) {
		return NULL;
	}

	int numLines = -1;
	for (int i = 0; i < fileLength; ++i)
		if (contents[i] == '\n')
			++numLines;

	assert(numLines >= 0);

	double* data = (double*)checkedCalloc(numLines * 2, sizeof(double));

	int actualPoints = 0;
	for (int i = 0; i < fileLength; ++i) {
		if (contents[i] != '#') {
			int j = i;
			while (++i < fileLength && (contents[i] != '\t' && contents[i] != '\n'));

			data[actualPoints * 2 + 0] = atof(&contents[j]);
			data[actualPoints * 2 + 1] = atof(&contents[i]);

			++actualPoints;
		}

		while (++i < fileLength && contents[i] != '\n');
	}

	free(contents);

	*numPoints = actualPoints;
	return data;
}

int main(int argc, char* argv[]) {
	if(argc != 2) {
		fprintf(stdout, "%s <.dat>\n", argv[0]);
		return EXIT_FAILURE;
	}

	fprintf(stdout, "Loading: %s\n", argv[1]);

	int numPoints = 0;
	double* data = readDatFile(argv[1], &numPoints);
	if(data == NULL) {
		return EXIT_FAILURE;
	}

	int pointDim = 2;
	int numMixtures = 2;

	struct GMM* gmm = fit(data, numPoints, pointDim, numMixtures);

	for (int mixture = 0; mixture < gmm->numMixtures; ++mixture) {
		fprintf(stdout, "Mixture %d:\n", mixture);

		fprintf(stdout, "\ttau: %.3f\n", gmm->tau[mixture]);

		fprintf(stdout, "\tmu: ");
		for (int dim = 0; dim < gmm->pointDim; ++dim)
			fprintf(stdout, "%.2f ", gmm->mu[mixture * pointDim + dim]);

		fprintf(stdout, "\n\tsigma: ");
		for (int dim = 0; dim < gmm->pointDim * gmm->pointDim; ++dim)
			fprintf(stdout, "%.2f ", gmm->sigma[mixture * gmm->pointDim * gmm->pointDim + dim]);

		fprintf(stdout, "\n\n");
	}

	freeGMM(gmm);

	free(data);

	return EXIT_SUCCESS;
}
