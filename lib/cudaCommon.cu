#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaCommon.hu"

__host__ void assertPowerOfTwo(size_t N) {
	int bit = 0;
	while(N > 0 && bit <= 1) {
		bit += N & 1;
		N >>= 1;
	}

	assert(bit <= 1);
}

__host__ size_t largestPowTwoLessThanEq(size_t N) {
	// Assigns the largest value (M = 2^n) < N to N and returns the residual.
	if(N == 0) {
		return 0;
	} // PC: N > 0

	size_t M = 1;
	while(M < N) {
		M *= 2;
	} // PC: M >= N

	if(M == N) {
		return M;
	} // PC: M > N
	
	return M / 2;
}

__host__ void calcDim(int N, cudaDeviceProp* devProp, dim3* block, dim3* grid) {
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

__host__ void dimToConsole(dim3* block, dim3* grid) {
	assert(block != NULL);
	assert(grid != NULL);

	printf("block: (%d, %d, %d)\n", block->x, block->y, block->z);
	printf("grid: (%d, %d, %d)\n", grid->x, grid->y, grid->z);
}

__host__ double* mallocOnGpu(const size_t N) {
	double* device_A;
	double ABytes = N * sizeof(double);
	check(cudaMalloc(&device_A, ABytes));
	return device_A;
}

__host__ double* sendToGpu(const size_t N, const double* host) {
	double* device;
	const size_t hostBytes = N * sizeof(double);
	check(cudaMalloc(&device, hostBytes));
	check(cudaMemcpy(device, host, hostBytes, cudaMemcpyHostToDevice));
	return device;
}

__host__ double* pinHostAndSendDevice(const size_t N, double* host) {
	double* device;
	const size_t hostBytes = N * sizeof(double);
	check(cudaHostRegister(host, hostBytes, cudaHostRegisterDefault));
	check(cudaMalloc(&device, hostBytes));
	check(cudaMemcpy(device, host, hostBytes, cudaMemcpyHostToDevice));
	return device;
}

__host__ void recvDeviceUnpinHost(double* device, double* host, const size_t N) {
	check(cudaMemcpy(host, device, N * sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(device);
	cudaHostUnregister(host);
}

__host__ void unpinHost(double* device, double* host) {
	cudaFree(device);
	cudaHostUnregister(host);
}
