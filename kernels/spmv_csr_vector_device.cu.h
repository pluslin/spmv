/* Copyright 2008 NVIDIA Corporation.  All Rights Reserved */

#pragma once

#include "sparse_formats.h"
#include "texture.h"
#include "kernels/spmv_common_device.cu.h"

//////////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a vector model (one warp per row)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_csr_vector_device
//   Each row of the CSR matrix is assigned to a warp.  The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with 
//   the x vector, in parallel.  This division of work implies that 
//   the CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned).  On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of 
//   work.  Since an entire 32-thread warp is assigned to each row, many 
//   threads will remain idle when their row contains a small number 
//   of elements.  This code relies on implicit synchronization among 
//   threads in a warp.
//
// spmv_csr_vector_tex_device
//   Same as spmv_csr_vector_tex_device, except that the texture cache is 
//   used for accessing the x vector.

template <unsigned int BLOCK_SIZE, bool UseCache>
__global__ void
spmv_csr_vector_kernel(const unsigned int num_rows,
                       const unsigned int * Ap, 
                       const unsigned int * Aj, 
                       const double * Ax, 
                       const double * x, 
                             double * y)
{
    __shared__ double sdata[BLOCK_SIZE];
    __shared__ unsigned int ptrs[BLOCK_SIZE/WARP_SIZE][2];
    
    const unsigned int thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
    const unsigned int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
    const unsigned int warp_id     = thread_id   / WARP_SIZE;                // global warp index
    const unsigned int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
    const unsigned int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
 
    for(unsigned int row = warp_id; row < num_rows; row += num_warps){
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the more straightforward option
        if(thread_lane < 2) 
            ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];   // 将行指针转载到相应的ptrs共享变量中
        const unsigned int row_start = ptrs[warp_lane][0]; //same as: row_start = Ap[row];
        const unsigned int row_end   = ptrs[warp_lane][1]; //same as: row_end   = Ap[row+1];

        // compute local sum
        sdata[threadIdx.x] = 0;
        for(unsigned int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
            sdata[threadIdx.x] += Ax[jj] * fetch_x<UseCache>(Aj[jj], x);

        // reduce local sums to row sum (ASSUME: warpsize 32)
        // 当sdata计算完毕时，每个warp的值（处理一行）需要规约到sdata[threadIdx.x]中
        if (thread_lane < 16) { sdata[threadIdx.x] += sdata[threadIdx.x + 16]; EMUSYNC; }   // 其中宏定义EMUSYNC就是同步意思
        if (thread_lane <  8) { sdata[threadIdx.x] += sdata[threadIdx.x +  8]; EMUSYNC; }
        if (thread_lane <  4) { sdata[threadIdx.x] += sdata[threadIdx.x +  4]; EMUSYNC; }
        if (thread_lane <  2) { sdata[threadIdx.x] += sdata[threadIdx.x +  2]; EMUSYNC; }
        if (thread_lane <  1) { sdata[threadIdx.x] += sdata[threadIdx.x +  1]; EMUSYNC; }

        // first thread writes warp result
        // 计算完毕之后，将共享空间中的数据写回到全部存储的y向量中去
        if (thread_lane == 0)
            y[row] += sdata[threadIdx.x];
    }
}

void spmv_csr_vector_device(const csr_matrix & d_csr, 
                            const double * d_x, 
                                  double * d_y)
{
    const unsigned int BLOCK_SIZE = 256;
    const unsigned int NUM_BLOCKS = MAX_THREADS/BLOCK_SIZE;
    
    spmv_csr_vector_kernel<BLOCK_SIZE, false> <<<NUM_BLOCKS, BLOCK_SIZE>>> 
        (d_csr.num_rows, d_csr.Ap, d_csr.Aj, d_csr.Ax, d_x, d_y);	

}

void spmv_csr_vector_tex_device(const csr_matrix & d_csr, 
                                const double * d_x, 
                                      double * d_y)
{
    const unsigned int BLOCK_SIZE = 256;
    const unsigned int NUM_BLOCKS = MAX_THREADS/BLOCK_SIZE;
    
    bind_x(d_x);
    
    spmv_csr_vector_kernel<BLOCK_SIZE, true> <<<NUM_BLOCKS, BLOCK_SIZE>>> 
        (d_csr.num_rows, d_csr.Ap, d_csr.Aj, d_csr.Ax, d_x, d_y);	

    unbind_x(d_x);
}

