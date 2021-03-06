#pragma once

#include "cuda.h"

// These textures are (optionally) used to cache the 'x' vector in y += A*x
texture<float,1> tex_x_float;
texture<int2,1> tex_x_double;

// use int2 to pul double through texture cache
void bind_x(const float * x)
{   CUDA_SAFE_CALL(cudaBindTexture(NULL, tex_x_float, x));   }

void bind_x(const double * x)
{   CUDA_SAFE_CALL(cudaBindTexture(NULL, tex_x_double, x));   }

void unbind_x(const float * x)
{   CUDA_SAFE_CALL(cudaUnbindTexture(tex_x_float)); }
void unbind_x(const double * x)
{   CUDA_SAFE_CALL(cudaUnbindTexture(tex_x_double)); }


template <bool UseCache>
__inline__ __device__ double fetch_x(const int& i, const double * x)
{
	if (UseCache){
		int2 v = tex1Dfetch(tex_x_double, i);
		return __hiloint2double(v.y, v.x);
	} else {
		return x[i];
	}
}
