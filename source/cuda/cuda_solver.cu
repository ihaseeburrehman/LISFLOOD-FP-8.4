#include "cuda_solver.cuh"
#include "cuda_device_functions.cuh"
#include "cuda_atomic.cuh"

__constant__ Geometry lis::cuda::geometry;
__constant__ Geometry lis::cuda::rain_geometry;
__constant__ int lis::cuda::pitch;
__constant__ lis::PhysicalParams lis::cuda::physical_params;
__constant__ lis::SolverParams lis::cuda::solver_params;
__constant__ lis::cuda::BoundaryConditions lis::cuda::boundaries;
__constant__ lis::SamplePoints lis::cuda::sample_points;
__managed__ NUMERIC_TYPE lis::cuda::dt;
__managed__ int lis::cuda::sample_buf_idx = 0;
#if defined(LISFLOOD_CUDA_MPI)
__constant__ int lis::cuda::fv1_row_begin;
__constant__ int lis::cuda::fv1_row_end;
__constant__ int lis::cuda::fv1_partition_active = 0;
__constant__ int lis::cuda::fv1_owns_north_ghost = 0;
__constant__ int lis::cuda::fv1_owns_south_ghost = 0;
#endif

__constant__ int lis::cuda::fv1_skip_dead = 0;
__constant__ const int* lis::cuda::fv1_live_sat = nullptr;

const dim3 lis::cuda::block_size(CUDA_BLOCK_SIZE_X, CUDA_BLOCK_SIZE_Y);

__device__ void lis::cuda::update_mass_stats_x
(
	MassStats* stats,
	NUMERIC_TYPE FH,
	int i,
	int j,
	NUMERIC_TYPE time_stage_fraction
)
{
	detail::update_mass_stats_x(stats, FH, i, j, time_stage_fraction);
}

__device__ void lis::cuda::update_mass_stats_y
(
	MassStats* stats,
	NUMERIC_TYPE FH,
	int i,
	int j,
	NUMERIC_TYPE time_stage_fraction
)
{
	detail::update_mass_stats_y(stats, FH, i, j, time_stage_fraction);
}
