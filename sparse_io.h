#pragma once

#include "sparse_convertions.h"

#include <stdio.h>
#include <stdlib.h>

extern "C"
{
    #include "mmio.h"
}

// 将稀疏矩阵文件读成coo格式文件
coo_matrix read_coo_matrix(const char *mm_filename)
{
    coo_matrix coo;

    FILE * fid;
    MM_typecode matcode;

	fid = fopen(mm_filename, "r");

    if(fid == NULL) {
        printf("Unable to open file %s\n", mm_filename);
        exit(1);
    }

    // 读入稀疏矩阵文件，并设置该矩阵的MM_typecode,其中banner为  '%%MatrixMarket'
    // Done: 读入矩阵文件第一行数据并填充matcode
    if(mm_read_banner(fid, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    // 判断该矩阵的MM_typecode是否有效
    if(!mm_is_valid(matcode)) {
        printf("Invalid Matrix Market file.\n");
        exit(1);
    }

    // 只支持稀疏的real和pattern coordinate的矩阵
    if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode))){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        printf("Only sparse real-valued or pattern coordinate matrices are supported\n");
        exit(1);
    }

    int num_rows, num_cols, num_nonzeros;
    // 将稀疏矩阵中的rows、cols、nonzeros读入到变量中，读入稀疏矩阵的第二、三行
    if(mm_read_mtx_crd_size(fid, &num_rows, &num_cols, &num_nonzeros) != 0) {
        exit(1);
    }

    // 将这几个值赋值到coo矩阵中
    coo.num_rows = num_rows;
    coo.num_cols = num_cols;
    coo.num_nonzeros = num_nonzeros;

    coo.I = new_host_array<unsigned int>(coo.num_nonzeros);
    coo.J = new_host_array<unsigned int>(coo.num_nonzeros);
    coo.V = new_host_array<double>(coo.num_nonzeros);

    printf("Reading sparse matrix from file (%s):", mm_filename);   // 注意没有\n
    // fflush的作用是刷新缓冲区，fflush(stdout)的作用是刷新输出缓冲区：即将当前的输出缓冲区的内容输出并清空输出缓冲区
    fflush(stdout);

    // pattern matrix only defines sparsity pattern, but not value
    if(mm_is_pattern(matcode)) {
        for(unsigned int i = 0; i < coo.num_nonzeros; i++) {
            fscanf(fid, " %d %d \n", &(coo.I[i]), &(coo.J[i]));
            coo.I[i]--;     // adjust from 1-based to 0-based index
            coo.J[i]--;
            coo.V[i] = 1.0; //  use value 1.0 for all nonzeros entries
        }
    }else if(mm_is_real(matcode) || mm_is_integer(matcode)) {
        for(unsigned int i = 0; i < coo.num_nonzeros; i++) {
            unsigned int I, J;
            double V;           // always read in a double and convert later if nessary
    
            fscanf(fid, " %d %d %lf \n", &I, &J, &V);
            coo.I[i] = I - 1;
            coo.J[i] = J - 1;
            coo.V[i] = V;
        }
    }else {
        printf("Unrecognized data type\n");
        exit(1);
    }

    fclose(fid);
    printf("done\n");

    // 当矩阵文件标记为symmetric时，说明真正的矩阵是对称，即我们需要把上(下)三角矩阵变为一个对称的完整的矩阵
    if(mm_is_symmetric(matcode)) {
        unsigned int off_diagonals = 0;
        for(unsigned int i = 0; i < coo.num_nonzeros; i++) {
            if(coo.I[i] != coo.J[i])
                off_diagonals++;
        }

        unsigned true_nonzeros = 2 * off_diagonals + (coo.num_nonzeros - off_diagonals);

        unsigned int * new_I = new_host_array<unsigned int>(true_nonzeros);
        unsigned int * new_J = new_host_array<unsigned int>(true_nonzeros);
        double * new_V = new_host_array<double>(true_nonzeros);

        unsigned int ptr = 0;
        for(unsigned int i = 0; i < coo.num_nonzeros; i++) {
            if(coo.I[i] != coo.J[i]) {
                new_I[ptr] = coo.I[i];  new_J[ptr] = coo.J[i];  new_V[ptr] = coo.V[i];
                ptr++;
                new_I[ptr] = coo.J[i];  new_J[ptr] = coo.I[i];  new_V[ptr] = coo.V[i];
                ptr++;
            }else {
                new_I[ptr] = coo.I[i];  new_J[ptr] = coo.J[i];  new_V[ptr] = coo.V[i];
                ptr++; 
            }
        }
        delete_host_array(coo.I);   delete_host_array(coo.J);   delete_host_array(coo.V);
        coo.I = new_I;  coo.J = new_J;  coo.V = new_V;
        coo.num_nonzeros = true_nonzeros;
    }

    return coo;
}

// 会被调用的稀疏矩阵读取文件，分为以下两步：
// 1、调用read_coo_matrix
// 2、调用coo_to_csr进行格式的转换
csr_matrix read_csr_matrix(const char * mm_filename, bool compact = false)
{
    coo_matrix coo = read_coo_matrix(mm_filename);

    // compact 默认为 false
    // coo_to_csr included in 'sparse_conversions.h'
    csr_matrix csr = coo_to_csr(coo, compact);

    delete_host_matrix(coo);

    return csr;
}
