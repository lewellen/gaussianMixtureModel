#include <assert.h>
#include <errno.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FLOAT_TYPE double
#define check(call) { cudaError_t __ABC123 = call; assert(__ABC123 == cudaSuccess); }

__global__ void cudaSum(int N, FLOAT_TYPE* dest, FLOAT_TYPE* src) {
	// Assumes a 2D grid of 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	dest[i] += src[i];
}

__global__ void cudaReduceBlocks(FLOAT_TYPE* dest) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;

	// Load into block shared memory
	__shared__ FLOAT_TYPE localSum[1024];
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

void initialize(FLOAT_TYPE* a, size_t N) {
	srand(time(NULL));
	for(size_t i = 0; i < N; ++i) {
		a[i] = i + 1; 
	}
}

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

FLOAT_TYPE parallelSum(FLOAT_TYPE* host_a, const size_t N) {
	assert(host_a != NULL);
	assert(N > 0);
	assertPowerOfTwo(N);

	const size_t NBytes = N * sizeof(FLOAT_TYPE);

	int deviceId;
	check(cudaGetDevice(&deviceId));

	cudaDeviceProp deviceProp;
	check(cudaGetDeviceProperties(&deviceProp, deviceId));

	FLOAT_TYPE *device_a;
	check(cudaMalloc(&device_a, NBytes));
	check(cudaMemcpy(device_a, host_a, NBytes, cudaMemcpyHostToDevice));

	// Parallel sum by continually folding the array in half and adding the right 
	// half to the left half until the fold size is 1024 (single block), then let
	// GPU reduce the remaining block to a single value and copy it over. O(log n).
	if(N >= 1024) {
		dim3 block, grid;
		for(size_t n = N/2; n >= 1024; n /= 2) {
			calcDim(n, &deviceProp, &block, &grid);
			cudaSum<<<grid, block>>>(n, device_a, device_a + n);
		}
		cudaReduceBlocks<<<1, 1024>>>(device_a);
	} else {
		cudaReduceBlocks<<<1, N>>>(device_a);
	}

	FLOAT_TYPE host_sum = 0;
	check(cudaMemcpy(&host_sum, device_a, sizeof(FLOAT_TYPE), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	cudaFree(device_a);

	return host_sum;
}

FLOAT_TYPE sequentialSum(FLOAT_TYPE* a, const size_t N) {
	assert(a != NULL);
	assert(N > 0);

	FLOAT_TYPE sum = 0;
	for(size_t i = 0; i < N; ++i) {
		sum += a[i];
	}
	return sum;
}

void testParallelSum() {
	int dev = 0;
	cudaSetDevice(dev);

	const size_t minN = 2;
	const size_t maxN = 16 * 1048576;
	for(size_t N = minN; N <= maxN; N *= 2) {
		FLOAT_TYPE* a = (FLOAT_TYPE*) malloc(N * sizeof(FLOAT_TYPE));
		initialize(a, N);

		double host_sum = sequentialSum(a, N);
		FLOAT_TYPE device_sum = parallelSum(a, N);

		double absDiff = fabs(host_sum - device_sum);
		if(absDiff > DBL_EPSILON) {
			printf("N: %zu, host_sum: %.15f, device_sum: %.15f, absDiff: %.15f\n", 
				N, host_sum, device_sum, absDiff
				);
			break;
		}

		free(a);
		cudaDeviceReset();
	}
}

int main(int argc, char** argv) {
	testParallelSum();
	return EXIT_SUCCESS;
}

