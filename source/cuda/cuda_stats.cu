#include "cuda_stats.cuh"
#include "cuda_solver.cuh"
#include "cuda_util.cuh"
#include <limits>
#include <cstdlib>
#if defined(LISFLOOD_CUDA_MPI)
#include <mpi.h>
#endif

namespace lis
{
namespace cuda
{

__global__ void accumulate_mass
(
	MassStats* cumulative,
	MassStats* instantaneous
)
{
	cumulative->in += instantaneous->in * cuda::dt;
	cumulative->out += instantaneous->out * cuda::dt;
}

/**
 * sums[0] += depth summed over the real cells this rank owns (no ghost
 * cells), sums[1] += how many of them are wet (depth >= threshold)
 */
__global__ void domain_sums
(
	const NUMERIC_TYPE* H,
	int ghost,
	NUMERIC_TYPE threshold,
	double* sums
)
{
	const int nx = cuda::geometry.xsz;
	const int row_begin = ghost ? cuda::fv1_owned_row_begin() : 0;
	const int rows = (ghost ? cuda::fv1_owned_row_end() : cuda::geometry.ysz) - row_begin;
	const int pitch = ghost ? nx + 2 : nx;
	double depth = 0.0, wet = 0.0;
	for (long k = blockIdx.x*(long)blockDim.x + threadIdx.x; k < (long)nx*rows;
			k += (long)blockDim.x*gridDim.x)
	{
		const int i = (int)(k % nx) + (ghost ? 1 : 0);
		const int j = (int)(k / nx) + row_begin;
		const NUMERIC_TYPE h = H[j*pitch + i];
		depth += h;
		if (h >= threshold) wet += 1.0;
	}
	add_block_sum(depth, &sums[0]);
	add_block_sum(wet, &sums[1]);
}

}
}

lis::cuda::StatsCollector::StatsCollector
(
	Geometry& geometry,
	NUMERIC_TYPE DepthThresh,
	int acceleration,
	int element_offset,
	int element_count
)
:
geometry(geometry),
depth_threshold(DepthThresh),
ghost(acceleration == 0),
distributed(element_count > 0)
{
	d_sums = static_cast<double*>(malloc_device(2*sizeof(double)));

	d_instantaneous_mass = static_cast<MassStats*>(
			malloc_device(sizeof(MassStats)));
	d_cumulative_mass = static_cast<MassStats*>(
			malloc_device(sizeof(MassStats)));
	zero_cumulative_mass();
}

void lis::cuda::StatsCollector::start(NUMERIC_TYPE* H)
{
	NUMERIC_TYPE initial_area;
	measure(H, previous_volume, initial_area);
}

void lis::cuda::StatsCollector::zero_instantaneous_mass()
{
	checkCudaErrors(cudaMemsetAsync(d_instantaneous_mass, 0,
			sizeof(MassStats)));
}

lis::MassStats* lis::cuda::StatsCollector::instantaneous_mass()
{
	return d_instantaneous_mass;
}

void lis::cuda::StatsCollector::accumulate_mass()
{
	accumulate_mass(cuda::dt);
}

void lis::cuda::StatsCollector::accumulate_mass(NUMERIC_TYPE step_dt)
{
	lis::cuda::accumulate_mass<<<1, 1>>>(d_cumulative_mass,
			d_instantaneous_mass);
	cumulative_time += step_dt;
}

lis::StatsEntry lis::cuda::StatsCollector::create_entry
(
	::Solver& solver,
	NUMERIC_TYPE* H
)
{
	NUMERIC_TYPE current_volume, current_area;
	measure(H, current_volume, current_area);
	NUMERIC_TYPE volume_change = current_volume - previous_volume;
	previous_volume = current_volume;

	double rain_total = rain_volume_total();

	MassStats h_instantaneous_mass;
	cuda::copy(&h_instantaneous_mass, d_instantaneous_mass, sizeof(MassStats));

	MassStats h_cumulative_mass;
	cuda::copy(&h_cumulative_mass, d_cumulative_mass, sizeof(MassStats));
#if defined(LISFLOOD_CUDA_MPI)
	int initialized = 0;
	MPI_Initialized(&initialized);
	if (initialized) {
		if (std::getenv("LISFLOOD_CUDA_MPI_TRACE")) {
			int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			fprintf(stderr, "MPI trace rank=%d local in=%g out=%g cumulative_in=%g cumulative_out=%g\n",
				rank, (double)h_instantaneous_mass.in, (double)h_instantaneous_mass.out,
				(double)h_cumulative_mass.in, (double)h_cumulative_mass.out);
			fflush(stderr);
		}
		MassStats global_instantaneous{}, global_cumulative{};
		MPI_Allreduce(&h_instantaneous_mass, &global_instantaneous, 2,
			sizeof(NUMERIC_TYPE) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&h_cumulative_mass, &global_cumulative, 2,
			sizeof(NUMERIC_TYPE) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		h_instantaneous_mass = global_instantaneous;
		h_cumulative_mass = global_cumulative;
		double global_rain = 0.0;
		MPI_Allreduce(&rain_total, &global_rain, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		rain_total = global_rain;
	}
#endif
	const double interval_rain = rain_total - previous_rain;
	previous_rain = rain_total;

	NUMERIC_TYPE time = cumulative_time;

	NUMERIC_TYPE net_source_flux =
		h_cumulative_mass.in - h_cumulative_mass.out;

	zero_cumulative_mass();

	// rain, infiltration and evaporation are sources and sinks, not errors
	NUMERIC_TYPE volume_error = (NUMERIC_TYPE)((double)net_source_flux
		+ interval_rain - interval_loss - (double)volume_change);
	interval_loss = 0.0;

	return {
		solver.t,
		cuda::dt,
		solver.itCount,
		solver.MinTstep,
		current_area,
		current_volume,
		h_instantaneous_mass,
		volume_error,
		volume_error/time,
		(NUMERIC_TYPE)total_loss,
		(NUMERIC_TYPE)rain_total
	};
}

lis::cuda::StatsCollector::~StatsCollector()
{
	free_device(d_sums);
	free_device(d_instantaneous_mass);
	free_device(d_cumulative_mass);
}

void lis::cuda::StatsCollector::measure
(
	NUMERIC_TYPE* H,
	NUMERIC_TYPE& volume,
	NUMERIC_TYPE& area
)
{
	checkCudaErrors(cudaMemsetAsync(d_sums, 0, 2*sizeof(double)));
	const long cells = (long)geometry.xsz*geometry.ysz;
	const int blocks = (int)std::min<long>(1024, std::max<long>(1, (cells + CUDA_BLOCK_SIZE - 1)/CUDA_BLOCK_SIZE));
	domain_sums<<<blocks, CUDA_BLOCK_SIZE>>>(H, ghost ? 1 : 0, depth_threshold, d_sums);
	double sums[2];
	cuda::copy(sums, d_sums, 2*sizeof(double));
#if defined(LISFLOOD_CUDA_MPI)
	if (distributed) {
		double global_sums[2] = {0.0, 0.0};
		MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		sums[0] = global_sums[0];
		sums[1] = global_sums[1];
	}
#endif
	volume = (NUMERIC_TYPE)(sums[0] * geometry.dx * geometry.dy);
	area = (NUMERIC_TYPE)(sums[1] * geometry.dx * geometry.dy);
}

void lis::cuda::StatsCollector::zero_cumulative_mass()
{
	MassStats temp_stats = {};
	cuda::copy(d_cumulative_mass, &temp_stats, sizeof(MassStats));

	cumulative_time = C(0.0);
}
