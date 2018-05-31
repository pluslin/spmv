#!/bin/bash

nvcc -o driver.o -c -arch=sm_30 -O3 -I. driver.cu
gcc -o mmio.o -c -O3 -I. mmio.c
gcc -o spmv driver.o mmio.o -L/usr/local/cuda/lib64 -lcudart -lm -lstdc++

