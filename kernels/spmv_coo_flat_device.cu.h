#pragma once

#include "sparse_formats.h"
#include "utils.h"
#include "texture.h"
#include "kernels/spmv_coo_serial_device.cu.h"
#include "kernels/spmv_common_device.cu.h"

////////////////////////////////////////////////////////////////////////////
// COO SpMV kernel which flattens data irregularity (segmented reduction)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_coo_flat_device
//   The input coo_matrix must be sorted by row.

template <unsigned int BLOCK_SIZE, bool UseCache>
	__global__ void
spmv_coo_flat_kernel(const unsigned int num_nonzeros,
		const unsigned int interval_size,
		const unsigned int * I, 
		const unsigned int * J, 
		const double * V, 
		const double * x, 
		double * y)
{
	__shared__ unsigned int idx[BLOCK_SIZE];				// BLOCK_SIZE = 256
	__shared__ double val[BLOCK_SIZE];
	__shared__ unsigned int carry_idx[BLOCK_SIZE / 32];
	__shared__ double carry_val[BLOCK_SIZE / 32];

	const unsigned int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;     // global thread index
	const unsigned int thread_lane = threadIdx.x & (WARP_SIZE-1);               // thread index within the warp
	const unsigned int warp_id     = thread_id   / WARP_SIZE;                   // global warp index
	const unsigned int warp_lane   = threadIdx.x / WARP_SIZE;                   // warp index within the CTA

	const unsigned int begin = warp_id * interval_size + thread_lane;           // thread's offset into I,J,V
	const unsigned int end   = min(begin + interval_size, num_nonzeros);        // end of thread's work

	if(begin >= end) return;                                                 // warp has no work to do

	const unsigned int first_idx = I[warp_id * interval_size];                  // first row of this warp's interval

	if (thread_lane == 0){
		carry_idx[warp_lane] = first_idx; 
		carry_val[warp_lane] = 0;
	}

	for(unsigned int n = begin; n < end; n += WARP_SIZE){
		idx[threadIdx.x] = I[n];                                             // row index
		val[threadIdx.x] = V[n] * fetch_x<UseCache>(J[n], x);                // val = A[row,col] * x[col] 

		if (thread_lane == 0){
			if(idx[threadIdx.x] == carry_idx[warp_lane])
				val[threadIdx.x] += carry_val[warp_lane];                    // row continues into this warp's span
			else if(carry_idx[warp_lane] != first_idx)
				y[carry_idx[warp_lane]] += carry_val[warp_lane];             // row terminated, does not span boundary
			else
				atomicAdd_(y + carry_idx[warp_lane], carry_val[warp_lane]);   // row terminated, but spans iter-warp boundary
		}

		// segmented reduction in shared memory
		if( thread_lane >=  1 && idx[threadIdx.x] == idx[threadIdx.x - 1] ) { val[threadIdx.x] += val[threadIdx.x -  1]; EMUSYNC; } 
		if( thread_lane >=  2 && idx[threadIdx.x] == idx[threadIdx.x - 2] ) { val[threadIdx.x] += val[threadIdx.x -  2]; EMUSYNC; }
		if( thread_lane >=  4 && idx[threadIdx.x] == idx[threadIdx.x - 4] ) { val[threadIdx.x] += val[threadIdx.x -  4]; EMUSYNC; }
		if( thread_lane >=  8 && idx[threadIdx.x] == idx[threadIdx.x - 8] ) { val[threadIdx.x] += val[threadIdx.x -  8]; EMUSYNC; }
		if( thread_lane >= 16 && idx[threadIdx.x] == idx[threadIdx.x -16] ) { val[threadIdx.x] += val[threadIdx.x - 16]; EMUSYNC; }

		if( thread_lane == 31 ) {
			carry_idx[warp_lane] = idx[threadIdx.x];                         // last thread in warp saves its results
			carry_val[warp_lane] = val[threadIdx.x];
		}
		else if ( idx[threadIdx.x] != idx[threadIdx.x+1] ) {                 // row terminates here
			if(idx[threadIdx.x] != first_idx)
				y[idx[threadIdx.x]] += val[threadIdx.x];                     // row terminated, does not span inter-warp boundary
			else
				atomicAdd_(y + idx[threadIdx.x], val[threadIdx.x]);           // row terminated, but spans iter-warp boundary
		}

	}

	// final carry
	if(thread_lane == 31){
		atomicAdd_(y + carry_idx[warp_lane], carry_val[warp_lane]); 
	}
}

	template <bool UseCache>
void __spmv_coo_flat_device(const coo_matrix& d_coo, 
		const double * d_x, 
		double * d_y)
{
	if(d_coo.num_nonzeros == 0)
		return; //empty matrix

	const unsigned int BLOCK_SIZE      = 128;
	const unsigned int MAX_BLOCKS      = 4*MAX_THREADS / BLOCK_SIZE; // empirically  better on test cases
	const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

	const unsigned int num_units  = d_coo.num_nonzeros / WARP_SIZE; 
	const unsigned int num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
	const unsigned int num_blocks = DIVIDE_INTO(num_warps, WARPS_PER_BLOCK);
	const unsigned int num_iters  = DIVIDE_INTO(num_units, num_warps);

	const unsigned int interval_size = WARP_SIZE * num_iters;

	const unsigned int tail = num_units * WARP_SIZE; // do the last few nonzeros separately

	if (UseCache)
		bind_x(d_x);

	spmv_coo_flat_kernel<BLOCK_SIZE, UseCache> <<<num_blocks, BLOCK_SIZE>>>
		(tail, interval_size, d_coo.I, d_coo.J, d_coo.V, d_x, d_y);

	spmv_coo_serial_kernel<<<1,1>>>
		(d_coo.num_nonzeros - tail, d_coo.I + tail, d_coo.J + tail, d_coo.V + tail, d_x, d_y);

	if (UseCache)
		unbind_x(d_x);
}

void spmv_coo_flat_device(const coo_matrix& d_coo, 
						  const double * d_x, 
						  double * d_y)
{ 
	__spmv_coo_flat_device<false>(d_coo, d_x, d_y);
}

void spmv_coo_flat_tex_device(const coo_matrix& d_coo, 
							  const double * d_x, 
							  double * d_y)
{ 
	__spmv_coo_flat_device<true>(d_coo, d_x, d_y);
}
