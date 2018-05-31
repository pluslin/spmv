#pragma once

#include <stdio.h>

#include "sparse_formats.h"
#include "utils.h"
#include "texture.h"

////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a scalar model (one thread per row)
///////////////////////////////////////////////////////////////////////
//
// spmv_csr_scalar_device
//   Straightforward translation of standard CSR SpMV to CUDA
//   where each thread computes y[i] += A[i,:] * x 
//   (the dot product of the i-th row of A with the x vector)
//
// spmv_csr_scalar_tex_device
//   Same as spmv_csr_scalar_device, except x is accessed via texture cache.
//

template <bool UseCache>
__global__ void
spmv_csr_scalar_kernel(const unsigned int num_rows,
                       const unsigned int * Ap, 
                       const unsigned int * Aj, 
                       const double * Ax, 
                       const double * x, 
                             double * y)
{

    // row index
    const unsigned int row = large_grid_thread_id();
    
    if(row < num_rows){     
        double sum = y[row];

        const unsigned int row_start = Ap[row];
        const unsigned int row_end   = Ap[row+1];
    
        for (unsigned int jj = row_start; jj < row_end; jj++){             
            sum += Ax[jj] * fetch_x<UseCache>(Aj[jj], x);       
        }

        y[row] = sum;
    }
}

void spmv_csr_scalar_device(const csr_matrix & d_csr, 
                            const double * d_x, 
                                  double * d_y)
{
    const unsigned int BLOCK_SIZE = 256; 
    
    const dim3 grid = make_large_grid(d_csr.num_rows, BLOCK_SIZE);

    spmv_csr_scalar_kernel<false> <<<grid, BLOCK_SIZE>>> 
        (d_csr.num_rows, d_csr.Ap, d_csr.Aj, d_csr.Ax, d_x, d_y);   
}

void spmv_csr_scalar_tex_device(const csr_matrix & d_csr, 
                                const double * d_x, 
                                      double * d_y)
{
    const unsigned int BLOCK_SIZE = 256;
    
    const dim3 grid = make_large_grid(d_csr.num_rows, BLOCK_SIZE);
    
    bind_x(d_x);

    spmv_csr_scalar_kernel<true> <<<grid, BLOCK_SIZE>>> 
        (d_csr.num_rows, d_csr.Ap, d_csr.Aj, d_csr.Ax, d_x, d_y);   

    unbind_x(d_x);
}

