#pragma once

#include "mem.h"

///////////////////////////////////////////////////
// define the sparse format
///////////////////////////////////////////////////

struct matrix_shape
{
	unsigned int num_rows, num_cols, num_nonzeros;
};

typedef struct : public matrix_shape
{
	unsigned int * I;	// row indices
	unsigned int * J;	// col indices
	double       * V;	// nonzero values
}coo_matrix;

typedef struct : public matrix_shape
{
	unsigned int * Ap;	// row pointer
	unsigned int * Aj;	// column indices
	double       * Ax;	// nonzeros
}csr_matrix;

typedef struct : public matrix_shape
{
	unsigned int stride;
	unsigned int num_cols_per_row;

	unsigned int * Aj;	// column indices stored in a (num_cols_per_row x stride) matrix
	double * Ax;		// nonzero values stored in a (num_cols_per_row x stride) matrix
}ell_matrix;

///////////////////////////////////////////////////
// sparse matrix memory management
///////////////////////////////////////////////////

void delete_coo_matrix(coo_matrix &coo, const memory_location loc) {
	delete_array(coo.I, loc);  delete_array(coo.J, loc);  delete_array(coo.V, loc);
}

void delete_csr_matrix(csr_matrix &csr, const memory_location loc) {
	delete_array(csr.Ap, loc);  delete_array(csr.Aj, loc);  delete_array(csr.Ax, loc);
}

void delete_ell_matrix(ell_matrix &ell, const memory_location loc) {
	delete_array(ell.Aj, loc);  delete_array(ell.Ax, loc);
}

///////////////////////////////////////////////////
// ! host function（函数重载）
///////////////////////////////////////////////////

void delete_host_matrix(coo_matrix &coo) { delete_coo_matrix(coo, HOST_MEMORY); }
void delete_host_matrix(csr_matrix &csr) { delete_csr_matrix(csr, HOST_MEMORY); }
void delete_host_matrix(ell_matrix &ell) { delete_ell_matrix(ell, HOST_MEMORY); }

///////////////////////////////////////////////////
// ! device function
///////////////////////////////////////////////////

void delete_device_matrix(coo_matrix &coo) { delete_coo_matrix(coo, DEVICE_MEMORY); }
void delete_device_matrix(csr_matrix &csr) { delete_csr_matrix(csr, DEVICE_MEMORY); }
void delete_device_matrix(ell_matrix &ell) { delete_ell_matrix(ell, DEVICE_MEMORY); }

///////////////////////////////////////////////////
// ! copy to device(函数重载)
///////////////////////////////////////////////////

coo_matrix copy_matrix_to_device(const coo_matrix &h_coo)
{
	coo_matrix d_coo = h_coo;	// copy fields（将h_coo的matrix_shape的变量赋值到d_coo中）
	d_coo.I = copy_array_to_device<unsigned int>(h_coo.I, h_coo.num_nonzeros);
	d_coo.J = copy_array_to_device<unsigned int>(h_coo.J, h_coo.num_nonzeros);
	d_coo.V = copy_array_to_device<double>(h_coo.V, h_coo.num_nonzeros);
	return d_coo;
}

csr_matrix copy_matrix_to_device(const csr_matrix &h_csr)
{
	csr_matrix d_csr = h_csr;
	d_csr.Ap = copy_array_to_device<unsigned int>(h_csr.Ap, h_csr.num_rows + 1);
	d_csr.Aj = copy_array_to_device<unsigned int>(h_csr.Aj, h_csr.num_nonzeros);
	d_csr.Ax = copy_array_to_device<double>(h_csr.Ax, h_csr.num_nonzeros);
	return d_csr;
}	

ell_matrix copy_matrix_to_device(const ell_matrix &h_ell) 
{
	ell_matrix d_ell = h_ell;
	d_ell.Aj = copy_array_to_device<unsigned int>(h_ell.Aj, h_ell.num_cols_per_row * h_ell.stride);
	d_ell.Ax = copy_array_to_device<double>(h_ell.Ax, h_ell.num_cols_per_row * h_ell.stride);
	return d_ell;
}

