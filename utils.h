#pragma once

#include <cmath>
#include <cuda.h>

// ceil(x/y) for integers
#define DIVIDE_INTO(x, y) ((x + y - 1)/y)

#define EMUSYNC __syncthreads()

static __inline__ __device__ double atomicAdd_(double *addr, double val)
{
	double old=*addr, assumed;

	do {
		assumed = old;
		old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
					__double_as_longlong(assumed),
					__double_as_longlong(val+assumed)));
	} while( assumed!=old );

	return old;
}

// for a given number of blocks, return a 2D grid large enough to contain them
dim3 make_large_grid(const unsigned int num_blocks) {
	if(num_blocks <= 65535) {
		return dim3(num_blocks);
	} else {
		unsigned int side = (unsigned int) ceil(sqrt((double)num_blocks));
		return dim3(side, side);
	}
}

dim3 make_large_grid(const unsigned int num_threads, const unsigned int blocksize) {
	const unsigned int num_blocks = DIVIDE_INTO(num_threads, blocksize);
	if(num_blocks <= 65535) {
		// fit 1D grids
		return dim3(num_blocks);
	} else {
		// fit 2D grids
		unsigned int side = (unsigned int) ceil(sqrt((double)num_blocks));
		return dim3(side, side);
	}
}

#define large_grid_thread_id(void) ((blockDim.x * (blockIdx.x + blockIdx.y*gridDim.x) + threadIdx.x))
