#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PI 3.141592653589793238

struct GMM {
	// Dimension of the data (each data point is a vector \in R^{pointDim})
	size_t pointDim;

	// Number of components
	size_t numMixtures;

	// Component weights: numMixtures x 1
	double* tau;

	// Component means: numMixtures x pointDim
	double* mu;

	// Component (upper triangular) covariances: numMixtures x pointDim^2
	double* sigma;

	// Component (lower triangular) covariance: numMixtures x pointDim^2
	double* sigmaL;

	// Leading normalizer on each component so prob integrates to 1: numMixtures x 1
	double* normalizer;
};

void prepareCholesky(struct GMM* gmm);

void* checkedCalloc(size_t count, size_t size) {
	errno = 0;
	void* result = calloc(count, size);
	if (errno != 0 || result == NULL) {
		perror("Failed to allocate memory.");
		if(result != NULL) {
			free(result);
		}
		exit(1);
	}

	return result;
}

struct GMM* initGMM(double* X, size_t numPoints, size_t numMixtures, size_t pointDim) {
	struct GMM* gmm = (struct GMM*)checkedCalloc(1, sizeof(struct GMM));
	gmm->pointDim = pointDim;
	gmm->numMixtures = numMixtures;

	// Initial guesses
	double uniformTau = 1.0 / numMixtures;
	gmm->tau = (double*)checkedCalloc(numMixtures, sizeof(double));
	for (size_t mixture = 0; mixture < numMixtures; ++mixture)
		gmm->tau[mixture] = uniformTau;

	gmm->mu = (double*)checkedCalloc(numMixtures * pointDim, sizeof(double));
	for (size_t mixture = 0; mixture < numMixtures; ++mixture) {
		size_t j = rand() % numPoints;
		for (size_t dim = 0; dim < pointDim; ++dim)
			gmm->mu[mixture * pointDim + dim] = X[j * pointDim + dim];
	}

	gmm->sigma = (double*)checkedCalloc(numMixtures * pointDim * pointDim, sizeof(double));
	for (size_t mixture = 0; mixture < numMixtures; ++mixture)
		for (size_t j = 0; j < pointDim; ++j)
			gmm->sigma[mixture * pointDim * pointDim + j * pointDim + j] = 1;

	gmm->sigmaL = (double*)checkedCalloc(numMixtures * pointDim * pointDim, sizeof(double));
	gmm->normalizer = (double*)checkedCalloc(numMixtures, sizeof(double));

	prepareCholesky(gmm);

	return gmm;
}

void freeGMM(struct GMM* gmm) {
	free(gmm->tau);
	free(gmm->mu);
	free(gmm->sigma);
	free(gmm->sigmaL);
	free(gmm->normalizer);
	free(gmm);
}

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

void prepareCholesky(struct GMM* gmm) {
	size_t blockSize = gmm->pointDim * gmm->pointDim;

	// Perform cholesky factorization once each iteration instead of 
	// repeadily for each normDist execution.
	for (size_t mixture = 0; mixture < gmm->numMixtures; ++mixture)
		choleskyDecomposition(&(gmm->sigma[mixture * blockSize]), &(gmm->sigmaL[mixture * blockSize]), gmm->pointDim);

	// det(Sigma) = det(L L^T) = det(L)^2
	for (size_t mixture = 0; mixture < gmm->numMixtures; ++mixture) {
		double det = 1.0;
		for (size_t i = 0; i < gmm->pointDim; ++i)
			det *= gmm->sigmaL[mixture * blockSize + i * gmm->pointDim + i];

		det *= det;

		gmm->normalizer[mixture] = sqrt(pow(2.0 * PI, gmm->pointDim) * det);
	}
}

void solvePositiveSemidefinite(double* B, size_t numPoints, struct GMM* gmm, size_t mixture, double* X) {
	size_t N = gmm->pointDim;
	double* L = &(gmm->sigmaL[N * N * mixture]);
	double* Z = (double*)calloc(numPoints * gmm->pointDim, sizeof(double));

	// 2015-09-23 GEL play the access of L into L(F)orward and L(B)ackward. 
	// Found that sequential access improved runtime.
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
		double* b = &(B[point * gmm->pointDim]);
		double* z = &(Z[point * gmm->pointDim]);

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
		double* z = &(Z[point * gmm->pointDim]);
		double* x = &(X[point * gmm->pointDim]);

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

void mvNormDist(double* X, size_t numPoints, struct GMM* gmm, size_t mixture, double* P) {
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
	for (size_t point = 0; point < numPoints; ++point)
		for (size_t dim = 0; dim < gmm->pointDim; ++dim)
			XM[point * gmm->pointDim + dim] = X[point * gmm->pointDim + dim] - mu[dim];

	// --> O(N^2) <--
	// S y = (x - m)
	solvePositiveSemidefinite(XM, numPoints, gmm, mixture, SXM);

	// O(N)
	// (x - m)^T y
	for (size_t point = 0; point < numPoints; ++point) {
		innerProduct[point] = 0.0;
		for (size_t dim = 0; dim < gmm->pointDim; ++dim)
			innerProduct[point] += XM[point * gmm->pointDim + dim] * SXM[point * gmm->pointDim + dim];
	}

	// O(1)
	for (size_t point = 0; point < numPoints; ++point) {
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
	struct GMM* gmm = initGMM(X, numPoints, pointDim, numMixtures);

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

		prepareCholesky(gmm);

	} while (1 == 1);

	free(prob);
	free(margin);

	free(xm);
	free(outerProduct);

	return gmm;
}

int getFileLength(FILE* handle, size_t* outFileLength) {
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

char* loadFile(const char* filePath, size_t* outFileLength) {
	char* contents = NULL;

	FILE* handle = fopen(filePath, "rb");
	if(handle == NULL) {
		perror(filePath);
	} else {
		if( getFileLength(handle, outFileLength) != 0 ) {
			contents = (char*)checkedCalloc(*outFileLength + 1, sizeof(char));
			assert(contents != NULL);

			size_t bytesRead = fread(contents, sizeof(char), *outFileLength, handle);
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

int isValidDatFile(const char* contents, const size_t fileLength, size_t* numLines, size_t* valuesPerLine) {
	assert(contents != NULL);
	assert(numLines != NULL);
	assert(valuesPerLine != NULL);

	*numLines = 0;
	*valuesPerLine = 0;

	size_t maxValuesPerLine = 0;
	size_t lastNewLine = 0;
	for(size_t i = 0; i < fileLength; ++i) {
		switch(contents[i]) {
			case '#': {
				// Ignore comments
				while(i < fileLength && contents[i] != '\n') 
					++i;
				*valuesPerLine = 0;
				break;
			}
			case '\t': {
				++(*valuesPerLine);
				break;
			}
			case '\n': {
				++(*valuesPerLine);

				if(maxValuesPerLine == 0) {
					maxValuesPerLine = *valuesPerLine;
				} else if (*valuesPerLine != maxValuesPerLine) {
					fprintf(stdout, "%.64s", &contents[lastNewLine]);
					fprintf(stdout, "Line: %zu\n", *numLines);
					fprintf(stdout, "Expect each line to have same number of values. %zu != %zu.\n", *valuesPerLine, maxValuesPerLine);
					return 0;
				}

				lastNewLine = i;
				*valuesPerLine = 0;
				++(*numLines);
				break;
			}
			default: {
				break;
			}
		}
	}

	if(*numLines == 0) {
		return 0;
	}

	*valuesPerLine = maxValuesPerLine;
	return 1;
}

double* parseDatFile(const char* filePath, size_t* numPoints, size_t* pointDim) {
	assert(filePath != NULL);
	assert(numPoints != NULL);
	assert(pointDim != NULL);

	size_t fileLength = 0;
	char* contents = loadFile(filePath, &fileLength);
	if(!isValidDatFile(contents, fileLength, numPoints, pointDim)) {
		free(contents);
		contents = NULL;
		return NULL;
	}

	double* data = (double*)checkedCalloc(*numPoints * *pointDim, sizeof(double));

	size_t currentPoint = 0;
	for(size_t i = 0, j = 0; i < fileLength; ++i) {
		switch(contents[i]) {
			case '#': {
				// Ignore comments
				while(i < fileLength && contents[i] != '\n') 
					++i;
				j = i;
				break;
			}
			case '\t': 
			case '\n': {
				data[currentPoint] = atof(&contents[j]);
				++currentPoint;
				j = i;
				break;
			}
			default: {
				break;
			}
		}
	}

	for(size_t i = 0; i < currentPoint; ++i) {
		assert( data[i] == data[i] );
		assert( data[i] != -INFINITY );
		assert( data[i] != +INFINITY );
	}

	if(contents != NULL) {
		free(contents);
		contents = NULL;
	}

	return data;
}

void usage(const char* programName) {
	fprintf(stdout, "%s <train.dat> <numComponents>\n", programName);
}

int main(int argc, char** argv) {
	if(argc != 3) {
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	errno = 0;
	size_t numMixtures = strtoul(argv[2], NULL, 0);
	if(errno != 0) {
		perror("Expected numComponents to be a positive integer.");
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	size_t numPoints = 0, pointDim = 0;
	double* data = parseDatFile(argv[1], &numPoints, &pointDim);
	if(data == NULL) {
		return EXIT_FAILURE;
	}

	struct GMM* gmm = fit(data, numPoints, pointDim, numMixtures);

	for (size_t mixture = 0; mixture < gmm->numMixtures; ++mixture) {
		fprintf(stdout, "Mixture %zu:\n", mixture);

		fprintf(stdout, "\ttau: %.3f\n", gmm->tau[mixture]);

		fprintf(stdout, "\tmu: ");
		for (size_t dim = 0; dim < gmm->pointDim; ++dim)
			fprintf(stdout, "%.2f ", gmm->mu[mixture * pointDim + dim]);

		fprintf(stdout, "\n\tsigma: ");
		for (size_t dim = 0; dim < gmm->pointDim * gmm->pointDim; ++dim)
			fprintf(stdout, "%.2f ", gmm->sigma[mixture * gmm->pointDim * gmm->pointDim + dim]);

		fprintf(stdout, "\n\n");
	}

	freeGMM(gmm);

	free(data);

	return EXIT_SUCCESS;
}
