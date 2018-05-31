#pragma once

// A simple timer class

#include "mem.h"
#include <cuda.h>

class timer
{
	cudaEvent_t start;
	cudaEvent_t end;

	public:
	timer()
	{ 
		CUDA_SAFE_CALL(cudaEventCreate(&start)); 
		CUDA_SAFE_CALL(cudaEventCreate(&end));
		CUDA_SAFE_CALL(cudaEventRecord(start,0));
	}

	~timer()
	{
		CUDA_SAFE_CALL(cudaEventDestroy(start));
		CUDA_SAFE_CALL(cudaEventDestroy(end));
	}

	float milliseconds_elapsed()
	{ 
		float elapsed_time;
		CUDA_SAFE_CALL(cudaEventRecord(end, 0));
		CUDA_SAFE_CALL(cudaEventSynchronize(end));
		CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, end));
		return elapsed_time;
	}
	float seconds_elapsed()
	{ 
		return 1000.0 * milliseconds_elapsed();
	}
};
