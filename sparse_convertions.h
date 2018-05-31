/********************************************************************
 * ! convert CSR format to other format
 * ! also include coo_to_csr() be called in 'sparse_io.h'
 * *****************************************************************/
#pragma once

#include <algorithm>
#include "sparse_formats.h"

void coo_to_csr(const unsigned int * I,
                const unsigned int * J,
                const double * V,
                const unsigned int num_rows,
                const unsigned int num_cols,
                const unsigned int num_nonzeros,
                unsigned int * Ap,
                unsigned int * Aj,
                double * Ax)
{
    unsigned int i, cumsum;
    for(i = 0; i < num_rows; i++) 
        Ap[i] = 0;

    for(i = 0; i < num_nonzeros; i++)
        Ap[I[i]]++;

    // cumsum the nnz per row
    for(i = 0, cumsum = 0; i < num_rows; i++) {
        unsigned int temp = Ap[i];
        Ap[i] = cumsum;
        cumsum += temp;
    }
    Ap[num_rows] = num_nonzeros;

    // Aj and Ax are not changed
    for(i = 0; i < num_nonzeros; i++) {
        Aj[i] = J[i];
        Ax[i] = V[i];
    }
}


// allocate memory for csr
csr_matrix coo_to_csr(const coo_matrix &coo, bool compact = false) {

    csr_matrix csr;

    csr.num_rows = coo.num_rows;
    csr.num_cols = coo.num_cols;
    csr.num_nonzeros = coo.num_nonzeros;

    csr.Ap = new_host_array<unsigned int>(csr.num_rows + 1);
    csr.Aj = new_host_array<unsigned int>(csr.num_nonzeros);
    csr.Ax = new_host_array<double>(csr.num_nonzeros);

    coo_to_csr(coo.I, coo.J, coo.V,
               coo.num_rows, coo.num_cols, coo.num_nonzeros,
               csr.Ap, csr.Aj, csr.Ax);
    
    return csr;
}


void csr_to_coo(const unsigned int * Ap,
                const unsigned int * Aj,
                const double * Ax,
                const unsigned int rows,
                const unsigned int cols,
                const unsigned int num_nonzeros,
                unsigned int * I,
                unsigned int * J,
                double * V)
{
    unsigned int i, j;

    for(i = 0; i < rows; i++) {
        for(j = Ap[i]; j < Ap[i+1]; j++) {
            I[j] = i;
        }
    }

    for(i = 0; i < num_nonzeros; i++) {
        J[i] = Aj[i];
        V[i] = Ax[i];
    }
}

// allocate memory for coo
coo_matrix csr_to_coo(const csr_matrix &csr) {

    coo_matrix coo;

    coo.num_rows = csr.num_rows;
    coo.num_cols = csr.num_cols;
    coo.num_nonzeros = csr.num_nonzeros;

    coo.I = new_host_array<unsigned int>(coo.num_nonzeros);
    coo.J = new_host_array<unsigned int>(coo.num_nonzeros);
    coo.V = new_host_array<double>(coo.num_nonzeros);

    csr_to_coo(csr.Ap, csr.Aj, csr.Ax,
               csr.num_rows, csr.num_cols, csr.num_nonzeros,
               coo.I, coo.J, coo.V);
    
    return coo;
}


// allocate memory for ell
ell_matrix csr_to_ell(const csr_matrix &csr, const unsigned max_cols_per_row, const unsigned int alignment = 32) {
	
	unsigned int num_cols_per_row = 0;
	for(int i = 0; i < csr.num_rows; i++) {
		num_cols_per_row = std::max(num_cols_per_row, csr.Ap[i+1] - csr.Ap[i]);
	}

	if(num_cols_per_row >= max_cols_per_row) {
		ell_matrix ell;

		ell.Aj = NULL;
		ell.Ax = NULL;
		ell.num_rows = 0;
		ell.num_cols = 0;
		ell.num_nonzeros = 0;
		ell.num_cols_per_row = num_cols_per_row;
		ell.stride = 0;
		printf("Covert to ell failed, too many columns: %d\n", num_cols_per_row);
		return ell;
	} else {
		ell_matrix ell;

		ell.num_rows = csr.num_rows;
		ell.num_cols = csr.num_cols;
		ell.num_nonzeros = csr.num_nonzeros;

		// stride is the ceil number of multiple of alignment
		ell.stride = alignment * ((ell.num_rows + alignment - 1) / alignment);
		ell.num_cols_per_row = num_cols_per_row;
		
		ell.Aj = new_host_array<unsigned int>(ell.num_cols_per_row * ell.stride);
		ell.Ax = new_host_array<double>(ell.num_cols_per_row * ell.stride);

		// Pad the Aj and Ax
		std::fill(ell.Aj, ell.Aj + ell.num_cols_per_row * ell.stride, 0);
		std::fill(ell.Ax, ell.Ax + ell.num_cols_per_row * ell.stride, 0);

		for(int i = 0; i < ell.num_rows; i++) {
			unsigned int n = 0;
			for(unsigned int jj = csr.Ap[i]; jj < csr.Ap[i+1]; jj++) {
				ell.Aj[ell.stride * n + i] = csr.Aj[jj];
				ell.Ax[ell.stride * n + i] = csr.Ax[jj];
				n++;
			}
		}

		return ell;
	}


}
