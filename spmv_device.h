#pragma once

////////////////////////////////////////////////////////////////////////////////
//! GPU SpMV kernels
//////////////////////////////////////////////////////////////////////////////////


#include "kernels/spmv_common_device.cu.h"
#include "kernels/spmv_coo_serial_device.cu.h"
#include "kernels/spmv_coo_flat_device.cu.h"
#include "kernels/spmv_csr_scalar_device.cu.h"
#include "kernels/spmv_csr_vector_device.cu.h"
#include "kernels/spmv_ell_device.cu.h"
