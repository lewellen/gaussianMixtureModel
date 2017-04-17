#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaCommon.hu"
#include "cudaFolds.hu"

__device__ void devVecAdd(size_t pointDim, double* dest, double* src) {
	for(size_t i = 0; i < pointDim; ++i) {
		dest[i] += src[i];
	}
}

__global__ void kernElementWiseSum(const size_t numPoints, const size_t pointDim, double* dest, double* src) {
	// Assumes a 2D grid of 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;

	if(i < numPoints) {
		devVecAdd(pointDim, &dest[i * pointDim], &src[i * pointDim]);
	}
}

__global__ void kernBlockWiseSum(const size_t numPoints, const size_t pointDim, double* dest) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;

	if(i < numPoints) {
		for(int s = blockDim.x / 2; threadIdx.x < s; s /= 2) {
			devVecAdd(pointDim, &dest[i * pointDim], &dest[(i + s) * pointDim]);
			__syncthreads();
		}
	}
}


__host__ void cudaArraySum(cudaDeviceProp* deviceProp, size_t numPoints, const size_t pointDim, double* device_A, cudaStream_t stream) {
	assert(deviceProp != NULL);
	assert(numPoints > 0);
	assert(pointDim > 0);
	assert(device_A != NULL);

	size_t M = largestPowTwoLessThanEq(numPoints);
	if(M != numPoints) {
		dim3 block , grid;
		calcDim(M, deviceProp, &block, &grid);
		kernElementWiseSum<<<grid, block, 0, stream>>>(
			numPoints - M, pointDim, device_A, device_A + M * pointDim
		);
		numPoints = M;
	}

	// Parallel sum by continually folding the array in half and adding the right 
	// half to the left half until the fold size is 1024 (single block), then let
	// GPU reduce the remaining block to a single value and copy it over. O(log n).
	if(numPoints > 1024) {
		dim3 block, grid;
		for(numPoints /= 2; numPoints >= 1024; numPoints /= 2) {
			calcDim(numPoints, deviceProp, &block, &grid);
			kernElementWiseSum<<<grid, block, 0, stream>>>(
				numPoints, pointDim, device_A, device_A + numPoints * pointDim
			);
		}
		numPoints *= 2;
	}

	assert(numPoints <= 1024);

	kernBlockWiseSum<<<1, numPoints, 0, stream>>>(
		numPoints, pointDim, device_A
	);
}

__host__ void cudaArraySum(cudaDeviceProp* deviceProp, size_t numPoints, const size_t pointDim, double* device_A, double* host_sum) {
	assert(host_sum != NULL);
	cudaArraySum(deviceProp, numPoints, pointDim, device_A);
	check(cudaMemcpy(host_sum, device_A, pointDim * sizeof(double), cudaMemcpyDeviceToHost));
}

__global__ void kernElementWiseMax(int N, double* dest, double* src) {
	// Assumes a 2D grid of 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;
	if(i < N) {
		if(dest[i] < src[i]) {
			dest[i] = src[i];
		}
	}
}

__global__ void kernBlockWiseMax(const size_t N, double* dest) {
	// Assumes a 2D grid of 1024x1 1D blocks
	int b = blockIdx.y * gridDim.x + blockIdx.x;
	int i = b * blockDim.x + threadIdx.x;

	// Load into block shared memory
	__shared__ double blockMax[1024];

	if(threadIdx.x >= N) {
		blockMax[threadIdx.x] = -INFINITY;
	} else {
		blockMax[threadIdx.x] = dest[i];
	}

	__syncthreads();	

	// Do all the calculations in block shared memory instead of global memory.
	for(int s = blockDim.x / 2; threadIdx.x < s; s /= 2) {
		if(blockMax[threadIdx.x] < blockMax[threadIdx.x + s]) {
			blockMax[threadIdx.x] = blockMax[threadIdx.x + s];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		// Just do one global write instead of 2048.
		dest[i] = blockMax[0];
	}
}

__host__ double cudaArrayMax(cudaDeviceProp* deviceProp, const size_t N, double* device_A) {
	// Parallel max by continually folding the array in half and maxing the right 
	// half to the left half until the fold size is 1024 (single block), then let
	// GPU reduce the remaining block to a single value and copy it over. O(log n).
	if(N > 1024) {
		assertPowerOfTwo(N);
		dim3 block, grid;
		for(size_t n = N/2; n >= 1024; n /= 2) {
			calcDim(n, deviceProp, &block, &grid);
			kernElementWiseMax<<<grid, block>>>(n, device_A, device_A + n);
			check((void)0);
		}
	}
 
	kernBlockWiseMax<<<1, 1024>>>(N, device_A);
	check((void)0);

	double maxValue = 0;
	check(cudaMemcpy(&maxValue, device_A, sizeof(double), cudaMemcpyDeviceToHost));
	check(cudaDeviceSynchronize());
	return maxValue;
}
