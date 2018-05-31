
#pragma once

#include "sparse_formats.h"
#include "utils.h"
#include "texture.h"

// kernel for ELL format

template <bool UseCache>
__global__ void
spmv_ell_kernel(const unsigned int num_rows, 
				const unsigned int num_cols, 
				const unsigned int num_cols_per_row,
				const unsigned int stride,
				const unsigned int * Aj,
				const double * Ax, 
				const double * x, 
				double * y)
{
	const unsigned int row = large_grid_thread_id();
	
	// 同理， 一个线程处理一行
	if(row >= num_rows){ return; }

	double sum = y[row];

	Aj += row;  // Aj为列索引，大小为cols_per_row x stride，按列存储
	Ax += row;  // Ax为非零元索引，大小也为cols_per_row x stride，按列存储

	for(unsigned int n = 0; n < num_cols_per_row; n++){
		const double A_ij = *Ax;

		if (A_ij != 0){
			const unsigned int col = *Aj;
			sum += A_ij * fetch_x<UseCache>(col, x);
		}

		Aj += stride;
		Ax += stride;
	}

	y[row] = sum;
}

void spmv_ell_device(const ell_matrix & d_ell, 
					 const double * d_x, 
					 double * d_y)
{
	const unsigned int BLOCK_SIZE = 256;
	const dim3 grid = make_large_grid(d_ell.num_rows, BLOCK_SIZE);

	spmv_ell_kernel<false> <<<grid, BLOCK_SIZE>>>
		(d_ell.num_rows, d_ell.num_cols, d_ell.num_cols_per_row, d_ell.stride,
		 d_ell.Aj, d_ell.Ax,
		 d_x, d_y);
}

void spmv_ell_tex_device(const ell_matrix & d_ell, 
						 const double * d_x, 
						 double * d_y)
{
	const unsigned int BLOCK_SIZE = 256;
	const dim3 grid = make_large_grid(d_ell.num_rows, BLOCK_SIZE);

	bind_x(d_x);

	spmv_ell_kernel<true> <<<grid, BLOCK_SIZE>>>
		(d_ell.num_rows, d_ell.num_cols, d_ell.num_cols_per_row, d_ell.stride,
		 d_ell.Aj, d_ell.Ax,
		 d_x, d_y);

	unbind_x(d_x);
}

