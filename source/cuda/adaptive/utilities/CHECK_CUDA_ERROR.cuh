#pragma once

#include "cuda_runtime.h"

#include <cstdio>  // for fprintf
#include <cstdlib> // for exit(error)

// Shared by the adaptive and acc_nugrid CUDA headers.
#ifndef LISFLOOD_CUDA_ASSERT_DEFINED
#define LISFLOOD_CUDA_ASSERT_DEFINED

#define CHECK_CUDA_ERROR(ans) { CUDAAssert( (ans), __FILE__, __LINE__); }

inline void CUDAAssert(cudaError_t error, const char* file, int line, bool abort = true)
{
	if (error != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s, %s, %d\n", cudaGetErrorString(error), file, line);

		if (abort) exit(error);
	}
}

#endif
