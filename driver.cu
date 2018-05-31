#include <stdio.h>
#include <stdlib.h>
#include "sparse_io.h"
#include "tests.h"

void run_all_kernels(int argc, char **argv)
{
    char * mm_filename = NULL;
    mm_filename = argv[1];

	csr_matrix csr;
    csr = read_csr_matrix(mm_filename);
    printf("Using %d-by-%d matrix with %d nonzero values\n", csr.num_rows, csr.num_cols, csr.num_nonzeros);

    // fill matrix with random values: some matrices have extreme values, 
    // which makes correctness testing difficult
    for(unsigned i = 0; i < csr.num_nonzeros; i++)
        csr.Ax[i] = rand() / (RAND_MAX + 1.0);
    
    // To do test
    test_coo_matrix_kernels(csr);
    test_csr_matrix_kernels(csr);
    test_ell_matrix_kernels(csr);

    delete_host_matrix(csr);
}


int main(int argc, char** argv) {
    // actually there will be some check operation

    run_all_kernels(argc, argv);

    return 0;
}
