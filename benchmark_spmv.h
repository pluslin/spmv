#pragma once
#include<stdio.h>
#include<vector>
#include "sparse_formats.h"
#include "timer.h"

char * BENCHMARK_OUTPUT_FILE_NAME = "benchmark_output.log";

size_t bytes_per_spmv(const ell_matrix & mtx)
{
	size_t bytes = 0;
	bytes += sizeof(unsigned int) * mtx.num_cols_per_row * mtx.stride;	//Aj
	bytes += sizeof(double) * mtx.num_cols_per_row * mtx.stride;		//Ax
	bytes += sizeof(double) * mtx.num_nonzeros;							//xj
	bytes += 2*sizeof(double) * mtx.num_rows;							//yi += yi + ...
	return bytes;
}


size_t bytes_per_spmv(const csr_matrix & mtx)
{
	size_t bytes = 0;
	bytes += 2*sizeof(unsigned int) * mtx.num_rows;     // row pointer
	bytes += 1*sizeof(unsigned int) * mtx.num_nonzeros; // column index
	bytes += 2*sizeof(double) * mtx.num_nonzeros; // A[i,j] and x[j]
	bytes += 2*sizeof(double) * mtx.num_rows;     // y[i] = y[i] + ...
	return bytes;
}

size_t bytes_per_spmv(const coo_matrix mtx)
{
	size_t bytes = 0;
	bytes += 2*sizeof(unsigned int) * mtx.num_nonzeros; // row and column indices;
	bytes += 2*sizeof(double) * mtx.num_nonzeros;       // A[i,j] and x[j]

	std::vector<size_t> occupied_rows(mtx.num_rows, 0);
	for(size_t n = 0; n < mtx.num_nonzeros; n++)
		occupied_rows[mtx.I[n]] = 1;
	for(size_t n = 0; n < mtx.num_rows; n++) {
		if(occupied_rows[n] == 1)
			bytes += 2*sizeof(double);                  // y[i] += y[i] + ...
	}
	return bytes;
}

	template <typename SparseMatrix, typename SpMV>
double benchmark_spmv(const SparseMatrix & sp_host, SpMV spmv, const memory_location loc, const char * method_name, const size_t num_iterations)
{
	SparseMatrix sp_loc;
	if(loc == DEVICE_MEMORY) {
		sp_loc = copy_matrix_to_device(sp_host);
	} else {
		sp_loc = sp_host;
	}

	//initialize host arrays
	double * x_host = new_host_array<double>(sp_host.num_cols);
	double * y_host = new_host_array<double>(sp_host.num_rows);
	for(unsigned int i = 0; i < sp_host.num_cols; i++)
		x_host[i] = rand() / (RAND_MAX + 1.0); 
	std::fill(y_host, y_host + sp_host.num_rows, 0);

	//initialize device arrays
	double * x_loc = copy_array(x_host, sp_host.num_cols, HOST_MEMORY, loc); 
	double * y_loc = copy_array(y_host, sp_host.num_rows, HOST_MEMORY, loc);

	// warmup    
	spmv(sp_loc, x_loc, y_loc);

	// time several SpMV iterations
	timer t;
	for(size_t i = 0; i < num_iterations; i++)
		spmv(sp_loc, x_loc, y_loc);
	cudaThreadSynchronize();
	double msec_per_iteration = t.milliseconds_elapsed() / (double) num_iterations;
	double sec_per_iteration = msec_per_iteration / 1000.0;
	double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) sp_host.num_nonzeros / sec_per_iteration) / 1e9;
	double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_spmv(sp_host) / sec_per_iteration) / 1e9;

	const char * location = (loc == HOST_MEMORY) ? "cpu" : "gpu";
	printf("\tbenchmarking %-20s [%s]: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", \
			method_name, location, msec_per_iteration, GFLOPs, GBYTEs); 

	//record results to file
	FILE * fid = fopen(BENCHMARK_OUTPUT_FILE_NAME, "a");
	fprintf(fid, "kernel=%s gflops=%f gbytes=%f msec=%f\n", method_name, GFLOPs, GBYTEs, msec_per_iteration); 
	fclose(fid);

	//deallocate buffers
	delete_host_array(x_host);
	delete_host_array(y_host);
	delete_array(x_loc, loc);
	delete_array(y_loc, loc);

	if (loc == DEVICE_MEMORY) delete_device_matrix(sp_loc);

	return msec_per_iteration;
}

	template <typename SparseMatrix, typename SpMV>
double benchmark_spmv_on_device(const SparseMatrix & sp_host, SpMV spmv,
		const char * method_name = NULL, const size_t num_iterations = 500)
{
	return benchmark_spmv<SparseMatrix, SpMV>(sp_host, spmv, DEVICE_MEMORY, method_name, num_iterations);
}

	template <typename SparseMatrix, typename SpMV>
double benchmark_spmv_on_host(const SparseMatrix & sp_host, SpMV spmv,
		const char * method_name = NULL, const size_t num_iterations = 500)
{
	return benchmark_spmv<SparseMatrix, SpMV>(sp_host, spmv, HOST_MEMORY, method_name, num_iterations);
}
