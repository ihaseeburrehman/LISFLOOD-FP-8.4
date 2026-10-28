#include "cuda_rain.cuh"
#include "cuda_device_functions.cuh"
#include "../rain/rain.h"
#include "cuda_unifiedallocator.cuh"
#include "cuda_util.cuh"

#if _NETCDF == 1
	#include "../rain/rain.tpp"
#else
	#include "../rain/rain_stub.tpp"
#endif

template class DynamicRain<lis::cuda::UnifiedAllocator<NUMERIC_TYPE>>;

__device__ double lis::cuda::rain_volume = 0.0;

double lis::cuda::rain_volume_total()
{
	double v = 0.0;
	checkCudaErrors(cudaMemcpyFromSymbol(&v, rain_volume, sizeof(double)));
	return v;
}

__device__ 
NUMERIC_TYPE lis::cuda::rate_at_cell
(
	NUMERIC_TYPE* rain_data,
	int i,
	int j
)
{
	return detail::rate_at_cell(rain_data, i, j);
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
lis::cuda::DynamicRain::update
(
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* H,
	NUMERIC_TYPE* rain_data
)
{
#if _NETCDF == 1
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = cuda::fv1_owned_row_begin()
		+ blockIdx.y*blockDim.y + threadIdx.y;
	double rate_sum = 0.0; // m/s summed over the cells, for rain_volume

	for (int j=global_j; j<cuda::fv1_owned_row_end(); j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			const int ti = i - threadIdx.x, tj = j - threadIdx.y;
			if (cuda::fv1_dead(ti, tj, ti + blockDim.x - 1, tj + blockDim.y - 1)) continue;
			NUMERIC_TYPE Z = DEM[j*cuda::pitch + i];
            if (FABS(Z - cuda::solver_params.nodata_elevation) < C(1e-6)) continue;

			NUMERIC_TYPE& Hval = H[j*cuda::pitch + i];
			const NUMERIC_TYPE rate = rate_at_cell(rain_data, i, j);
            Hval += rate * cuda::dt;
			rate_sum += rate;
		}
	}
	add_block_sum(rate_sum*cuda::dt*cuda::geometry.dx*cuda::geometry.dy, &rain_volume);
#endif
}


__device__ 
NUMERIC_TYPE lis::cuda::rate_at_cell_ACC
(
	NUMERIC_TYPE* rain_data,
	int i,
	int j
)
{
    int tile_i = (i) * cuda::geometry.dx / cuda::rain_geometry.dx;

    NUMERIC_TYPE top_gap = cuda::geometry.tly - cuda::rain_geometry.tly;
    int tile_j = (j - top_gap/cuda::geometry.dy) *
		cuda::geometry.dy / cuda::rain_geometry.dy;

    if (tile_i < cuda::rain_geometry.xsz && tile_j >= 0)
    {
        return rain_data[tile_j*cuda::rain_geometry.xsz + tile_i];
    }
    else
    {
        return C(0.0);
    }
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
lis::cuda::DynamicRain::updateACC
(
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* H,
	NUMERIC_TYPE* rain_data
)
{
#if _NETCDF == 1
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
	double rate_sum = 0.0; // m/s summed over the cells, for rain_volume

	for (int j=global_j; j<cuda::geometry.ysz; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<cuda::geometry.xsz; i+=blockDim.x*gridDim.x)
		{
			NUMERIC_TYPE Z = DEM[j*cuda::geometry.xsz + i];
            if (FABS(Z - cuda::solver_params.nodata_elevation) < C(1e-6)) continue;

			NUMERIC_TYPE& Hval = H[j*cuda::geometry.xsz + i];
			const NUMERIC_TYPE rate = rate_at_cell_ACC(rain_data, i, j);
            Hval += rate * cuda::dt;
			rate_sum += rate;
		}
	}
	add_block_sum(rate_sum*cuda::dt*cuda::geometry.dx*cuda::geometry.dy, &rain_volume);
#endif
}
