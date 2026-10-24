#pragma once
#define CUDA_BLOCK_SIZE_X 16
#define CUDA_BLOCK_SIZE_Y 16
#define CUDA_BLOCK_SIZE 256

#include <algorithm>
#include <cfloat>
#include "../geometry.h"
#include "cuda_boundary.cuh"
#include "cuda_stats.cuh"
#include "params.h"
#include "sample.h"

namespace lis
{
namespace cuda
{

extern __constant__ Geometry geometry;
extern __constant__ Geometry rain_geometry;
extern __constant__ int pitch;
extern __constant__ PhysicalParams physical_params;
extern __constant__ SolverParams solver_params;
extern __constant__ BoundaryConditions boundaries;
extern __constant__ SamplePoints sample_points;
extern __managed__ NUMERIC_TYPE dt;
extern __managed__ int sample_buf_idx;

#if defined(LISFLOOD_CUDA_MPI)
extern __constant__ int fv1_row_begin;
extern __constant__ int fv1_row_end;
extern __constant__ int fv1_partition_active;
extern __constant__ int fv1_owns_north_ghost;
extern __constant__ int fv1_owns_south_ghost;
__device__ __forceinline__ bool fv1_owns_row(int j)
{
	return !fv1_partition_active || (j >= fv1_row_begin && j < fv1_row_end);
}
__device__ __forceinline__ bool fv1_owns_cfl_row(int j)
{
	return fv1_owns_row(j) || (j == 0 && fv1_owns_north_ghost) ||
		(j == cuda::geometry.ysz+1 && fv1_owns_south_ghost);
}
__device__ __forceinline__ int fv1_owned_row_begin()
{
	return fv1_partition_active ? fv1_row_begin : 1;
}
__device__ __forceinline__ int fv1_owned_row_end()
{
	return fv1_partition_active ? fv1_row_end : cuda::geometry.ysz + 1;
}
#else
__device__ __forceinline__ bool fv1_owns_row(int) { return true; }
__device__ __forceinline__ bool fv1_owns_cfl_row(int) { return true; }
__device__ __forceinline__ int fv1_owned_row_begin() { return 1; }
__device__ __forceinline__ int fv1_owned_row_end() { return cuda::geometry.ysz + 1; }
#endif

extern const dim3 block_size;

/** *a = min(*a, v) for v >= 0: the bit patterns of non-negative floats order as the values */
__device__ inline void atomic_min_nonnegative(float* a, float v)
{
	atomicMin(reinterpret_cast<unsigned int*>(a), __float_as_uint(v));
}

__device__ inline void atomic_min_nonnegative(double* a, double v)
{
	atomicMin(reinterpret_cast<unsigned long long*>(a),
			static_cast<unsigned long long>(__double_as_longlong(v)));
}

/**
 * minimum of v (>= 0) over the threads of a block of CUDA_BLOCK_SIZE threads,
 * taken into *min (atomically); within each warp, then over the warps
 */
__device__ inline void block_min_to(NUMERIC_TYPE v, NUMERIC_TYPE* min)
{
	__shared__ NUMERIC_TYPE warp_min[CUDA_BLOCK_SIZE/32];
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	for (int s=16; s>0; s>>=1) v = FMIN(v, __shfl_down_sync(0xffffffff, v, s));
	if ((tid & 31) == 0) warp_min[tid/32] = v;
	__syncthreads();
	if (tid < 32)
	{
		v = (tid < (blockDim.x*blockDim.y)/32) ? warp_min[tid] : FLT_MAX;
		for (int s=16; s>0; s>>=1) v = FMIN(v, __shfl_down_sync(0xffffffff, v, s));
		if (tid == 0) atomic_min_nonnegative(min, v);
	}
}

/**
 * blocks for the kernels of block_size threads over nx by ny cells, allowing
 * for tiles that overlap by one cell (FV1, FV2); at most 128 x 128, larger
 * grids are covered by the kernels' grid-stride loops
 */
inline dim3 tile_grid(int nx, int ny)
{
	auto blocks = [](int n, int b) { return std::min(128, std::max(1, (n + b - 3)/(b - 1) + 1)); };
	return dim3(blocks(nx, CUDA_BLOCK_SIZE_X), blocks(ny, CUDA_BLOCK_SIZE_Y));
}

/**
 * blocks for the cell-wise kernels of CUDA_BLOCK_SIZE threads in a row over
 * nx by ny cells: one block row per grid row, at most 128 x 128
 */
inline dim3 row_grid(int nx, int ny)
{
	return dim3(std::min(128, std::max(1, (nx + CUDA_BLOCK_SIZE - 1)/CUDA_BLOCK_SIZE)),
			std::min(128, std::max(1, ny)));
}

__device__
void update_mass_stats_x
(
	MassStats* stats,
	NUMERIC_TYPE FH,
	int i,
	int j,
	NUMERIC_TYPE time_stage_fraction = C(1.0)
);

__device__
void update_mass_stats_y
(
	MassStats* stats,
	NUMERIC_TYPE FH,
	int i,
	int j,
	NUMERIC_TYPE time_stage_fraction = C(1.0)
);

template<typename F>
class Solver
{
	public:
	virtual void zero_ghost_cells() = 0;

	virtual void update_ghost_cells() = 0;

	virtual void update_uniform_rain(NUMERIC_TYPE  rain_rate) = 0;

	virtual void updateMaxFieldACC(NUMERIC_TYPE t) = 0;
	
	virtual void zero_thin_depth_slopes() = 0;

	virtual void FloodplainQ() = 0; 

	virtual F& update_flow_variables
	(
		MassStats* mass_stats
	) = 0;

	virtual F& d_U() = 0;

	virtual void update_dt_per_element
	(
		NUMERIC_TYPE* dt_field
	) const = 0;

	/**
	 * the minimum of the per-element time steps over elements
	 * [first, first + count) into the device scalar min_dt, in one kernel
	 * and without the dt field; false if the solver has no such kernel
	 */
	virtual bool reduce_dt
	(
		NUMERIC_TYPE* min_dt,
		int first,
		int count
	) const { return false; }

	virtual ~Solver() {};
};

template<typename F>
class DynamicTimestep
{
public:
	DynamicTimestep
	(
	 	NUMERIC_TYPE& dt,
		Geometry& geometry,
		NUMERIC_TYPE max_dt,
		int adaptive_ts,
		Solver<F>& solver,
		int element_offset = 0,
		int element_count = 0
	);

	NUMERIC_TYPE update_dt();

	~DynamicTimestep();

private:
	bool adaptive;
	NUMERIC_TYPE& dt;
	NUMERIC_TYPE max_dt;
	const Solver<F>& solver;
	NUMERIC_TYPE* dt_field;
	const int element_offset;
	const int elements;
	void* d_temp;
	size_t bytes;
	NUMERIC_TYPE* d_reduced_dt;
#if defined(LISFLOOD_CUDA_MPI)
	NUMERIC_TYPE* h_reduced_dt;
#endif
};


template<typename F>
class DynamicTimestepACC
{
public:
	DynamicTimestepACC
	(
		NUMERIC_TYPE& dt,
		Geometry& geometry,
		NUMERIC_TYPE max_dt,
		int adaptive_ts,
		Solver<F>& solver
	);

	NUMERIC_TYPE update_dt_ACC();

	~DynamicTimestepACC();

private:
	bool adaptive;
	NUMERIC_TYPE& dt;
	NUMERIC_TYPE max_dt;
	const Solver<F>& solver;
	NUMERIC_TYPE* dt_field;
	const int elements;
	void* d_temp;
	size_t bytes;
};


}
}

#include "cuda_solver.templates.cu"
