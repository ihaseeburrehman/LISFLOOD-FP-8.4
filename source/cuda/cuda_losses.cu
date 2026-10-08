#include "cuda_losses.cuh"
#include "cuda_geometry.cuh"
#include "cuda_util.cuh"
#include "../cuda/ghostraster.h"
#include "io.h"
#include <cstring>

namespace lis
{
namespace cuda
{

/** sum of v over the threads of a block, added once per block to *total */
__device__ void add_block_sum(double v, double* total)
{
	__shared__ double partial[CUDA_BLOCK_SIZE];
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	partial[tid] = v;
	__syncthreads();
	for (int s = (blockDim.x*blockDim.y)/2; s > 0; s >>= 1)
	{
		if (tid < s) partial[tid] += partial[tid + s];
		__syncthreads();
	}
	if (tid == 0 && partial[0] > 0.0) atomicAdd(total, partial[0]);
}

/**
 * depth after losing rate*dt plus the cell's residual (not below zero), or h
 * if the cell is dry; residual keeps the part of the loss that rounding to
 * single precision dropped, for the next step
 */
__device__ NUMERIC_TYPE depth_after_loss
(
	NUMERIC_TYPE h,
	NUMERIC_TYPE rate,
	NUMERIC_TYPE& residual
)
{
	if (h <= cuda::solver_params.DepthThresh || rate <= C(0.0))
	{
		residual = C(0.0);
		return h;
	}
	NUMERIC_TYPE wanted = rate*cuda::dt + residual;
	NUMERIC_TYPE h_new = h - wanted;
	if (h_new <= C(0.0))
	{
		residual = C(0.0);
		return C(0.0);
	}
	residual = wanted - (h - h_new); // h - h_new is exact (Sterbenz)
	return h_new;
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
apply_losses_fv1
(
	NUMERIC_TYPE* H,
	NUMERIC_TYPE* HU,
	NUMERIC_TYPE* HV,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* rate_grid,
	NUMERIC_TYPE* residual,
	NUMERIC_TYPE uniform_rate,
	NUMERIC_TYPE evap_rate,
	double* volume
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
	double removed = 0.0;

	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			int idx = j*cuda::pitch + i;
			if (FABS(DEM[idx] - cuda::solver_params.nodata_elevation) < C(1e-6)) continue;

			NUMERIC_TYPE h = H[idx];
			NUMERIC_TYPE rate = ((rate_grid != nullptr) ? rate_grid[idx] : uniform_rate)
				+ evap_rate;
			NUMERIC_TYPE h_new = depth_after_loss(h, rate, residual[idx]);
			if (h_new == h) continue;

			H[idx] = h_new;
			if (h_new > cuda::solver_params.DepthThresh)
			{
				// keep the velocity: discharge falls with the depth
				NUMERIC_TYPE scale = h_new/h;
				HU[idx] *= scale;
				HV[idx] *= scale;
			}
			else
			{
				HU[idx] = C(0.0);
				HV[idx] = C(0.0);
			}
			removed += (double)(h - h_new);
		}
	}

	add_block_sum(removed*cuda::geometry.dx*cuda::geometry.dy, volume);
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
apply_losses_acc
(
	NUMERIC_TYPE* H,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* rate_grid,
	NUMERIC_TYPE* residual,
	NUMERIC_TYPE uniform_rate,
	NUMERIC_TYPE evap_rate,
	double* volume
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
	double removed = 0.0;

	for (int j=global_j; j<cuda::geometry.ysz; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<cuda::geometry.xsz; i+=blockDim.x*gridDim.x)
		{
			int idx = j*cuda::geometry.xsz + i;
			if (FABS(DEM[idx] - cuda::solver_params.nodata_elevation) < C(1e-6)) continue;

			NUMERIC_TYPE h = H[idx];
			NUMERIC_TYPE rate = ((rate_grid != nullptr) ? rate_grid[idx] : uniform_rate)
				+ evap_rate;
			NUMERIC_TYPE h_new = depth_after_loss(h, rate, residual[idx]);
			if (h_new == h) continue;

			H[idx] = h_new;
			removed += (double)(h - h_new);
		}
	}

	add_block_sum(removed*cuda::geometry.dx*cuda::geometry.dy, volume);
}

}
}

lis::cuda::Losses::Losses
(
	Fnames& filenames,
	States& states,
	Pars& pars,
	Geometry& geometry,
	int pitch,
	int offset,
	int verbose
)
:
acc(states.acceleration == ON)
{
	infiltration = (states.calc_infiltration == ON);
	evaporation = (states.calc_evap == ON);
	if (!enabled()) return;

	if (infiltration)
	{
		if (strlen(filenames.infilfilename) > 0)
		{
			// per-cell rate, mm/h -> m/s
			NUMERIC_TYPE* rate = acc ? lis::GhostRaster::allocate_H(geometry)
				: lis::GhostRaster::allocate(geometry);
			FILE* file = fopen_or_die(filenames.infilfilename, "rb",
					"Loading infilfile\n", verbose);
			Geometry file_geometry;
			NUMERIC_TYPE no_data_value;
			AsciiRaster::read_header(file, file_geometry, no_data_value);
			AsciiRaster::match_cell_dimensions_or_die(geometry, file_geometry,
					"infilfile");
			AsciiRaster::read(file, rate, geometry, pitch, offset);
			fclose(file);
			AsciiRaster::replace_no_data(rate, geometry, pitch, offset,
					no_data_value, C(0.0));
			for (int j=0; j<geometry.ysz; j++)
			{
				for (int i=0; i<geometry.xsz; i++)
				{
					NUMERIC_TYPE& r = rate[j*pitch + i + offset];
					r = (r > C(0.0)) ? r/C(3600000.0) : C(0.0);
				}
			}
			if (acc)
			{
				d_rate = GhostRaster::allocate_device_H(geometry);
				GhostRaster::copy_H(d_rate, rate, geometry);
			}
			else
			{
				d_rate = GhostRaster::allocate_device(geometry);
				GhostRaster::copy(d_rate, rate, geometry);
			}
			delete[] rate;
			if (verbose) printf("Infiltration: per-cell rates from %s\n",
					filenames.infilfilename);
		}
		else
		{
			uniform_rate = pars.InfilRate;
			if (verbose) printf("Infiltration: uniform rate %.10" NUM_FMT" m/s\n",
					uniform_rate);
		}
	}

	if (evaporation)
	{
		// the CPU loader: time series in mm/day, converted to m/s
		Arrays arrays;
		arrays.evap = nullptr;
		LoadEvap(&filenames, &arrays, verbose);
		evap = arrays.evap;
	}

	// residuals start at zero, in the layout of H
	NUMERIC_TYPE* zeros = acc ? lis::GhostRaster::allocate_H(geometry)
		: lis::GhostRaster::allocate(geometry);
	if (acc)
	{
		d_residual = GhostRaster::allocate_device_H(geometry);
		GhostRaster::copy_H(d_residual, zeros, geometry);
	}
	else
	{
		d_residual = GhostRaster::allocate_device(geometry);
		GhostRaster::copy(d_residual, zeros, geometry);
	}
	delete[] zeros;

	d_volume = static_cast<double*>(malloc_device(sizeof(double)));
	double zero = 0.0;
	copy(d_volume, &zero, sizeof(double));
}

NUMERIC_TYPE lis::cuda::Losses::evaporation_rate
(
	NUMERIC_TYPE t
)
{
	return evaporation ? InterpolateTimeSeries(evap, t) : C(0.0);
}

void lis::cuda::Losses::apply_fv1
(
	NUMERIC_TYPE* H,
	NUMERIC_TYPE* HU,
	NUMERIC_TYPE* HV,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE t,
	dim3 grid_size
)
{
	apply_losses_fv1<<<grid_size, CUDA_BLOCK_SIZE>>>(H, HU, HV, DEM, d_rate,
			d_residual, infiltration ? uniform_rate : C(0.0), evaporation_rate(t), d_volume);
}

void lis::cuda::Losses::apply_acc
(
	NUMERIC_TYPE* H,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE t,
	dim3 grid_size
)
{
	apply_losses_acc<<<grid_size, CUDA_BLOCK_SIZE>>>(H, DEM, d_rate,
			d_residual, infiltration ? uniform_rate : C(0.0), evaporation_rate(t), d_volume);
}

double lis::cuda::Losses::total_volume()
{
	if (d_volume == nullptr) return 0.0;
	double v;
	copy(&v, d_volume, sizeof(double));
	return v;
}

double lis::cuda::Losses::take_interval_volume()
{
	double total = total_volume();
	double interval = total - reported;
	reported = total;
	return interval;
}

lis::cuda::Losses::~Losses()
{
	free_device(d_rate);
	free_device(d_residual);
	free_device(d_volume);
}
