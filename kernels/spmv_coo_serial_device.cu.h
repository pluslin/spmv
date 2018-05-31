# pragma once

#include "sparse_formats.h"

// only use one thread to run kernel
__global__ void
spmv_coo_serial_kernel(const unsigned int num_nonzeros,
                       const unsigned int *I,
                       const unsigned int *J,
                       const double *V,
                       const double *x,
                             double *y)
{
    for(int n = 0; n < num_nonzeros; n++) {
        y[I[n]] += V[n] * x[J[n]];
    }
}

void spmv_coo_serial_device(const coo_matrix &d_coo, 
                            const double * d_x,
                            double * d_y)
{
    spmv_coo_serial_kernel<<<1,1>>>(d_coo.num_nonzeros, d_coo.I, d_coo.J, d_coo.V, d_x, d_y);
}
