#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaKernels.hu"

void calcDim(int N, cudaDeviceProp* devProp, dim3* block, dim3* grid) {
	assert(devProp != NULL);
	assert(block != NULL);
	assert(grid != NULL);

	// make a 2D grid of 1D blocks
	const int numThreadRows = 1;
	const int numThreadCols = devProp->maxThreadsPerBlock;
	block->x = min(numThreadCols, N);
	block->y = numThreadRows;

	const int numThreadsPerBlock = numThreadRows * numThreadCols;
	const int residualThreads = N % numThreadsPerBlock;
	int numBlocksPerGrid = (N - residualThreads) / numThreadsPerBlock;
	if(residualThreads > 0) {
		++numBlocksPerGrid;
	}

	const int numBlockCols = min( numBlocksPerGrid, devProp->maxGridSize[0] );
	const int residualBlocks = numBlocksPerGrid % numBlockCols;
	int numBlockRows = (numBlocksPerGrid - residualBlocks) / numBlockCols;
	if(residualBlocks > 0) {
		++numBlockRows;
	}

	grid->x = numBlockCols;
	grid->y = numBlockRows;

	assert(block->x * block->y * block->z > 0);
	assert(block->x * block->y * block->z <= devProp->maxThreadsPerBlock);

	assert(grid->x <= devProp->maxGridSize[0]);
	assert(grid->y <= devProp->maxGridSize[1]);
	assert(grid->z <= devProp->maxGridSize[2]);
}

void dimToConsole(dim3* block, dim3* grid) {
	assert(block != NULL);
	assert(grid != NULL);

	printf("block: (%d, %d, %d)\n", block->x, block->y, block->z);
	printf("grid: (%d, %d, %d)\n", grid->x, grid->y, grid->z);
}

void assertPowerOfTwo(size_t N) {
	int bit = 0;
	while(N > 0 && bit <= 1) {
		bit += N & 1;
		N >>= 1;
	}
	assert(bit <= 1);
}

__global__ void cudaGmmLogLikelihood(
	const size_t numPoints, const size_t numComponents,
	const double* logpi, double* logProb
) { 
	// Input:
	// logpi = 1 x numComponets
	// logProb = numComponents x numPoints

	// Output:
	// logProb[i] = log likelihood of single point i

	assert(numPoints > 0);
	assert(numComponents > 0);
	assert(logpi != NULL);
	assert(logProb != NULL);

	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	
	double maxArg = -INFINITY;
	for(size_t k = 0; k < numComponents; ++k) {
		const double logProbK = logpi[k] + logProb[k * numPoints + i];
		if(logProbK > maxArg) {
			maxArg = logProbK;
		}
	}

	double sum = 0.0;
	for (size_t k = 0; k < numComponents; ++k) {
		const double logProbK = logpi[k] + logProb[k * numPoints + i];
		sum = exp(logProbK - maxArg);
	}

	assert(sum >= 0);

	logProb[i] = maxArg + log(sum);
}

__device__ void devVecMinus(const size_t N, double* x, double* y) {
	assert(N > 0);
	assert(x != NULL);
	assert(y != NULL);
	// x == y allowed

	for(size_t i = 0; i < N; ++i) {
		x[i] -= y[i];
	}
}

__device__ void devSolveLowerTri(const size_t N, double* L, double* x, double* b) {
	assert(N > 0);
	assert(L != NULL);
	assert(x != NULL);
	assert(b != NULL);
	// x == b allowed

	for(size_t i = 0; i < N; ++i) {
		double sum = 0.0;
		if(i > 0) {
			for(size_t j = 0; j <= i - 1; ++j) {
				sum += L[i * N + j] * x[j];
			}
		}

		x[i] = (b[i] - sum) / L[i * N + i];
	}
}

__device__ void devSolveLowerTriT(const size_t N, double* L, double* x, double* b) {
	assert(N > 0);
	assert(L != NULL);
	assert(x != NULL);
	assert(b != NULL);
	// x == b allowed

	// treat L as an upper triangular matrix U
	for(size_t i = 0; i < N; i++) {
		size_t ip = N - 1 - i;
		double sum = 0;
		for(size_t j = ip + 1; j < N; ++j) {
			sum += L[j * N + ip] * x[j];
		}

		x[ip] = (b[ip] - sum) / L[ip * N + ip];
	}
}

__device__ double devVecDot(const size_t N, double* x, double* y) {
	assert(N > 0);
	assert(x != NULL);
	assert(y != NULL);

	double sum = 0;
	for(size_t i = 0; i < N; ++i) {
		sum += x[i] * y[i];
	}
	return sum;
}

__global__ void cudaLogMVNormDist(
	const size_t numPoints, const size_t pointDim, 
	double* X, double* mu, double* sigmaL, double logNormalizer,
	double* dest
) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i >= numPoints) {
		return;
	}

	double u[1024];

	double* x = & X[i * pointDim];
	devVecMinus(pointDim, x, mu); // x <- x - mu
	devSolveLowerTri(pointDim, sigmaL, u, x); // u <- u s.t. L u = (x - mu)
	devSolveLowerTriT(pointDim, sigmaL, u, u); // u <- v s.t. L^T v = u
	dest[i] = logNormalizer - 0.5 * devVecDot(pointDim, u, x);
}

__global__ void cudaSum(int N, double* dest, double* src) {
	// Assumes a 2D grid of 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	dest[i] += src[i];
}

__global__ void cudaReduceBlocks(double* dest) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;

	// Load into block shared memory
	__shared__ double localSum[1024];
	localSum[threadIdx.x] = dest[i];
	__syncthreads();	

	// Do all the calculations in block shared memory instead of global memory.
	for(int s = blockDim.x / 2; threadIdx.x < s; s /= 2) {
		localSum[threadIdx.x] += localSum[threadIdx.x + s];
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		// Just do one global write instead of 2048.
		dest[i] = localSum[0];
	}
}
