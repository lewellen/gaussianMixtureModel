#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaCommon.hu"

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

__host__ void assertPowerOfTwo(size_t N) {
	int bit = 0;
	while(N > 0 && bit <= 1) {
		bit += N & 1;
		N >>= 1;
	}
	assert(bit <= 1);
}

__global__ void kernArraySum(int N, double* dest, double* src) {
	// Assumes a 2D grid of 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	dest[i] += src[i];
}

__global__ void kernReduceBlocks(double* dest) {
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

__host__ double cudaReduceSum(cudaDeviceProp* deviceProp, const size_t N, double* device_A) {
	// Parallel sum by continually folding the array in half and adding the right 
	// half to the left half until the fold size is 1024 (single block), then let
	// GPU reduce the remaining block to a single value and copy it over. O(log n).
	if(N >= 1024) {
		dim3 block, grid;
		for(size_t n = N/2; n >= 1024; n /= 2) {
			calcDim(n, deviceProp, &block, &grid);
			kernArraySum<<<grid, block>>>(n, device_A, device_A + n);
		}
		kernReduceBlocks<<<1, 1024>>>(device_A);
	} else {
		kernReduceBlocks<<<1, N>>>(device_A);
	}

	double sum = 0;
	check(cudaMemcpy(&sum, device_A, sizeof(double), cudaMemcpyDeviceToHost));
	return sum;
}
