/*******************************************
 * ! spmv in host memory
 * ! CPU spmv kernel
 * *****************************************/

#pragma once

#include "sparse_formats.h"


// COO host_kernel
void __spmv_coo_serial_host(const unsigned int num_nonzeros,
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

void spmv_coo_serial_host(const coo_matrix &h_coo, 
                          const double * h_x,
                          double * h_y)
{
    __spmv_coo_serial_host(h_coo.num_nonzeros, h_coo.I, h_coo.J, h_coo.V, h_x, h_y);
}

// CSR host_kernel
void __spmv_csr_serial_host(const unsigned int num_rows,
                            const unsigned int *Ap,
                            const unsigned int *Aj,
                            const double *Ax,
                            const double *x,
                            double *y)
{
    unsigned int row_start;
    unsigned int row_end;
    unsigned int i,j;
    double sum;

    for(i = 0; i < num_rows; i++) {
        row_start = Ap[i];
        row_end = Ap[i+1];
        sum = y[i];

        for(j = row_start; j < row_end; j++) {
            sum += Ax[j] * x[Aj[j]];
        }

        y[i] = sum;           
    }
}

void spmv_csr_serial_host(const csr_matrix &h_csr, 
                          const double * h_x,
                          double * h_y)
{
    __spmv_csr_serial_host(h_csr.num_rows, h_csr.Ap, h_csr.Aj, h_csr.Ax, h_x, h_y);
}
