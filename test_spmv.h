/******************************************
 * ! test the kernel's correcty
 * ! general test program
 * ****************************************/
#pragma once

#include<algorithm>
#include<assert.h>
#include "mem.h"


double maximum_relative_error(double * A, double * B, unsigned int N)
{
    unsigned int i;
    double max_error = 0;

    for(i = 0; i < N; i++) {
        double a = A[i];
        double b = B[i];

        double error = std::abs(a - b);
        if(error != 0) {
            max_error = std::max(max_error, error/(std::abs(a) + std::abs(b)));
        }
    }

    return max_error;
}


// Compare the result of two SpMV kernels
// And this compare operation should copy one more data(when in device)
// compare with variant _loc
template <typename SparseMatrix1, typename SpMV1,
          typename SparseMatrix2, typename SpMV2>
void compare_spmv_kernels(const SparseMatrix1 & sm1_host, SpMV1 spmv1, const memory_location loc1,
                          const SparseMatrix2 & sm2_host, SpMV2 spmv2, const memory_location loc2)
{
    // first checking
    assert(sm1_host.num_rows == sm2_host.num_rows);
    assert(sm1_host.num_cols == sm2_host.num_cols);
    assert(sm1_host.num_nonzeros == sm2_host.num_nonzeros);

    const unsigned int num_rows = sm1_host.num_rows;
    const unsigned int num_cols = sm1_host.num_cols;

    // transfer matrix from host to destination mem
    SparseMatrix1 sm1_loc = (loc1 == DEVICE_MEMORY) ? copy_matrix_to_device(sm1_host) : sm1_host;
    SparseMatrix2 sm2_loc = (loc2 == DEVICE_MEMORY) ? copy_matrix_to_device(sm2_host) : sm2_host;

    // initialize host vectors of x\y
    double * x_host = new_host_array<double>(num_cols);
    double * y_host = new_host_array<double>(num_rows);

    for(int i = 0; i < num_cols; i++) 
        x_host[i] = rand() / (RAND_MAX + 1.0);
    for(int i = 0; i < num_rows; i++)
        y_host[i] = rand() / (RAND_MAX + 1.0);
    
    // create vectors in appropriate locations
    double * x_loc1 = copy_array<double>(x_host, num_cols, HOST_MEMORY, loc1);
    double * y_loc1 = copy_array<double>(y_host, num_rows, HOST_MEMORY, loc1);
    double * x_loc2 = copy_array<double>(x_host, num_cols, HOST_MEMORY, loc2);
    double * y_loc2 = copy_array<double>(y_host, num_rows, HOST_MEMORY, loc2);

    // compute y = A*x
    spmv1(sm1_loc, x_loc1, y_loc1);
    spmv2(sm2_loc, x_loc2, y_loc2);

    // transfer the result to host
    double * y_sm1_result = copy_array<double>(y_loc1, num_rows, loc1, HOST_MEMORY);
    double * y_sm2_result = copy_array<double>(y_loc2, num_rows, loc2, HOST_MEMORY);

    double max_error = maximum_relative_error(y_sm1_result, y_sm2_result, num_rows);
    printf(" [max error %9f]", max_error);

    // 1e-4 means 10^-4
    if ( max_error > 1e-4 )
        printf(" FAILURE");

    // cleanup -- free the temp memory
    if (loc1 == DEVICE_MEMORY) delete_device_matrix(sm1_loc);
    if (loc2 == DEVICE_MEMORY) delete_device_matrix(sm2_loc);
    delete_host_array(x_host);
    delete_host_array(y_host);
    delete_array(x_loc1, loc1);
    delete_array(y_loc1, loc1);
    delete_array(x_loc2, loc2);
    delete_array(y_loc2, loc2);
    delete_host_array(y_sm1_result);
    delete_host_array(y_sm2_result);
}


// A general template to reload the format and kernel function
// SPMV1 is the reference
template <typename SparseMatrix1, typename SpMV1,
          typename SparseMatrix2, typename SpMV2>
void test_spmv_kernel(const SparseMatrix1 & sm1_host, SpMV1 spmv1, const memory_location loc1,
                      const SparseMatrix2 & sm2_host, SpMV2 spmv2, const memory_location loc2,
                      const char * method_name)
{
    printf("\ttesting %-26s", method_name);
    if(loc2 == HOST_MEMORY)
        printf("[cpu]:");
    else
        printf("[gpu]:");

    compare_spmv_kernels(sm1_host, spmv1, loc1, sm2_host, spmv2, loc2);

    printf("\n");
}
