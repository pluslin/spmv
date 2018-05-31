#pragma once

#include "spmv_device.h"
#include "sparse_io.h"
#include "spmv_host.h"
#include "test_spmv.h"
#include "benchmark_spmv.h"
#include "mem.h"

void test_coo_matrix_kernels(const csr_matrix &csr)
{
    printf("\n######     Testing COO Kernels     #####\n");

    // Create COO Matrix, will allocate memory when convert format
    printf("\tcreating coo_matrix:");
    coo_matrix coo = csr_to_coo(csr);
    printf("\n");

    // Check out the correcty of kernel
    test_spmv_kernel(csr, spmv_csr_serial_host, HOST_MEMORY,
                     coo, spmv_coo_serial_host, HOST_MEMORY,
                     "coo_serial");

    // Test kernels
    test_spmv_kernel(csr, spmv_csr_serial_host, HOST_MEMORY,
                     coo, spmv_coo_flat_device, DEVICE_MEMORY,
                     "coo_matrix_flat");
    
	test_spmv_kernel(csr, spmv_csr_serial_host, HOST_MEMORY,
                     coo, spmv_coo_flat_tex_device, DEVICE_MEMORY,
                     "coo_matrix_flat_tex");

    // Benchmark kernel(test esplased time)
    benchmark_spmv_on_device(coo, spmv_coo_flat_device, "coo_matrix_flat");
    benchmark_spmv_on_device(coo, spmv_coo_flat_tex_device, "coo_matrix_flat_tex");

    delete_host_matrix(coo);
}

void test_csr_matrix_kernels(const csr_matrix &csr)
{
    printf("\n######     Testing CSR Kernels     #####\n");

    // Test kernels
    test_spmv_kernel(csr, spmv_csr_serial_host, HOST_MEMORY,
                     csr, spmv_csr_scalar_device, DEVICE_MEMORY,
                     "csr_scalar");
    
	test_spmv_kernel(csr, spmv_csr_serial_host, HOST_MEMORY,
                     csr, spmv_csr_scalar_tex_device, DEVICE_MEMORY,
                     "csr_scalar_tex");

    test_spmv_kernel(csr, spmv_csr_serial_host, HOST_MEMORY,
                     csr, spmv_csr_vector_device, DEVICE_MEMORY,
                     "csr_vector");
    
	test_spmv_kernel(csr, spmv_csr_serial_host, HOST_MEMORY,
                     csr, spmv_csr_vector_tex_device, DEVICE_MEMORY,
                     "csr_vector_tex");

    // Benchmark kernel(test esplased time)
    benchmark_spmv_on_device(csr, spmv_csr_scalar_device, "csr_scalar");
    benchmark_spmv_on_device(csr, spmv_csr_scalar_tex_device, "csr_scalar_tex");
    benchmark_spmv_on_device(csr, spmv_csr_vector_device, "csr_vector");
    benchmark_spmv_on_device(csr, spmv_csr_vector_tex_device, "csr_vector_tex");
}

void test_ell_matrix_kernels(const csr_matrix &csr, float max_fill = 3.0)
{
    printf("\n######     Testing ELL Kernels     #####\n");

	// triple long
	unsigned int max_cols_per_row = (max_fill * csr.num_nonzeros) / csr.num_rows + 1;

	printf("Creating ell_matrix:");
	ell_matrix ell = csr_to_ell(csr, max_cols_per_row);
	printf("\n");

    // Test kernels
    test_spmv_kernel(csr, spmv_csr_serial_host, HOST_MEMORY,
                     ell, spmv_ell_device, DEVICE_MEMORY,
                     "ell");
    
	test_spmv_kernel(csr, spmv_csr_serial_host, HOST_MEMORY,
                     ell, spmv_ell_tex_device, DEVICE_MEMORY,
                     "ell_tex");

    // Benchmark kernel(test esplased time)
    benchmark_spmv_on_device(ell, spmv_ell_device, "ell");
    benchmark_spmv_on_device(ell, spmv_ell_tex_device, "ell_tex");

}

