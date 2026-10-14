#include "cuda_losses.cuh"
#include "cuda_geometry.cuh"
#include "cuda_util.cuh"
#include "../cuda/ghostraster.h"
#include "io.h"
#include "adaptive/classes/AssembledSolution.h"
#include "acc_nugrid/AssembledSolution.h"
#include "adaptive/zorder/compact.cuh"
#include "adaptive/utilities/get_lvl_idx.cuh"
#include <cmath>
#include <cstdlib>
#include <cstring>
#if defined(LISFLOOD_CUDA_MPI)
#include <mpi.h>
#endif

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
	NUMERIC_TYPE& residual,
	NUMERIC_TYPE step_dt, NUMERIC_TYPE threshold
)
{
	if (h <= threshold || rate <= C(0.0))
	{
		residual = C(0.0);
		return h;
	}
	NUMERIC_TYPE wanted = rate*step_dt + residual;
	NUMERIC_TYPE h_new = h - wanted;
	if (h_new <= C(0.0))
	{
		residual = C(0.0);
		return C(0.0);
	}
	residual = wanted - (h - h_new); // h - h_new is exact (Sterbenz)
	return h_new;
}

/**
 * Green-Ampt: depth after this step's infiltration (and evaporation) on a
 * cell with water; updates F, the depth infiltrated so far
 */
__device__ NUMERIC_TYPE depth_after_greenampt
(
	const LossParams& p,
	int idx,
	NUMERIC_TYPE h, NUMERIC_TYPE step_dt
)
{
	NUMERIC_TYPE& residual = p.residual[idx];
	if (h <= C(0.0))
	{
		residual = C(0.0);
		return h;
	}

	double dt = step_dt;
	double F = p.F[idx];
	double ks = p.ks[idx];
	double dF = 0.0;
	if (ks > 0.0)
	{
		// implicit Euler: F_new = F + dt Ks (1 + (psi + h) dtheta / F_new)
		double M = ((double)p.psi[idx] + (double)h) * (double)p.dtheta[idx];
		double a = F + ks*dt;
		dF = 0.5*(a + sqrt(a*a + 4.0*ks*dt*M)) - F;
		if (p.fmax != nullptr)
		{
			dF = fmin(dF, fmax((double)p.fmax[idx] - F, 0.0));
		}
	}
	double e = (double)p.evap_rate * dt;
	if (dF + e <= 0.0)
	{
		residual = C(0.0);
		return h;
	}

	NUMERIC_TYPE wanted = (NUMERIC_TYPE)(dF + e) + residual;
	NUMERIC_TYPE h_new = h - wanted;
	if (h_new <= C(0.0))
	{
		// not enough water: infiltration takes it first, evaporation the rest
		p.F[idx] = F + fmin(dF, (double)h);
		residual = C(0.0);
		return C(0.0);
	}
	residual = wanted - (h - h_new); // h - h_new is exact (Sterbenz)
	p.F[idx] = F + dF;
	return h_new;
}

/** depth of cell idx after this step's losses */
__device__ NUMERIC_TYPE depth_after_losses
(
	const LossParams& p,
	int idx,
	NUMERIC_TYPE h, NUMERIC_TYPE step_dt, NUMERIC_TYPE threshold
)
{
	if (p.greenampt) return depth_after_greenampt(p, idx, h, step_dt);
	NUMERIC_TYPE rate = ((p.rate_grid != nullptr) ? p.rate_grid[idx] : p.uniform_rate)
		+ p.evap_rate;
	return depth_after_loss(h, rate, p.residual[idx], step_dt, threshold);
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
apply_losses_fv1
(
	NUMERIC_TYPE* H,
	NUMERIC_TYPE* HU,
	NUMERIC_TYPE* HV,
	NUMERIC_TYPE* DEM,
	LossParams p,
	double* volume
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
	double removed = 0.0;

	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		if (!cuda::fv1_owns_row(j)) continue;
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			int idx = j*cuda::pitch + i;
			if (FABS(DEM[idx] - cuda::solver_params.nodata_elevation) < C(1e-6)) continue;

			NUMERIC_TYPE h = H[idx];
			NUMERIC_TYPE h_new = depth_after_losses(p, idx, h, cuda::dt, cuda::solver_params.DepthThresh);
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
apply_losses_slopes
(
	NUMERIC_TYPE* H, NUMERIC_TYPE* H1x, NUMERIC_TYPE* H1y,
	NUMERIC_TYPE* HU, NUMERIC_TYPE* HU1x, NUMERIC_TYPE* HU1y,
	NUMERIC_TYPE* HV, NUMERIC_TYPE* HV1x, NUMERIC_TYPE* HV1y,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE face,
	LossParams p,
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
			NUMERIC_TYPE h_new = depth_after_losses(p, idx, h, cuda::dt, cuda::solver_params.DepthThresh);
			if (h_new == h) continue;

			H[idx] = h_new;
			bool faces_dry = face*FABS(H1x[idx]) >= h_new || face*FABS(H1y[idx]) >= h_new;
			if (h_new > cuda::solver_params.DepthThresh && !faces_dry)
			{
				// keep the velocity: discharge falls with the depth
				NUMERIC_TYPE scale = h_new/h;
				HU[idx] *= scale; HU1x[idx] *= scale; HU1y[idx] *= scale;
				HV[idx] *= scale; HV1x[idx] *= scale; HV1y[idx] *= scale;
			}
			else if (h_new > cuda::solver_params.DepthThresh)
			{
				// a face would become dry (cell on a wet/dry edge): make the cell
				// first order, so every face has the cell's depth and velocity
				// (a face with discharge but no depth would stop the time step)
				NUMERIC_TYPE scale = h_new/h;
				H1x[idx] = C(0.0); H1y[idx] = C(0.0);
				HU[idx] *= scale; HU1x[idx] = C(0.0); HU1y[idx] = C(0.0);
				HV[idx] *= scale; HV1x[idx] = C(0.0); HV1y[idx] = C(0.0);
			}
			else
			{
				H1x[idx] = C(0.0); H1y[idx] = C(0.0);
				HU[idx] = C(0.0); HU1x[idx] = C(0.0); HU1y[idx] = C(0.0);
				HV[idx] = C(0.0); HV1x[idx] = C(0.0); HV1y[idx] = C(0.0);
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
	LossParams p,
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
			NUMERIC_TYPE h_new = depth_after_losses(p, idx, h, cuda::dt, cuda::solver_params.DepthThresh);
			if (h_new == h) continue;

			H[idx] = h_new;
			removed += (double)(h - h_new);
		}
	}

	add_block_sum(removed*cuda::geometry.dx*cuda::geometry.dy, volume);
}


/** Preserve refinement wherever the physical mask or soil parameters vary.
 * Otherwise a flat basin would merge before spatial losses can create detail. */
__global__ void preflag_loss_details(bool* details, int L, Geometry geom,
	int pitch, int offset, LossParams p, const NUMERIC_TYPE* dem, NUMERIC_TYPE nodata)
{
	int node = blockIdx.x*blockDim.x + threadIdx.x;
	if (node >= get_lvl_idx(L)) return;
	int level = 0;
	while (level+1 < L && node >= get_lvl_idx(level+1)) ++level;
	unsigned code = (node-get_lvl_idx(level)) << (2*(L-level));
	int x0 = compact(code), y0 = compact(code >> 1);
	int side = 1 << (L-level);
	int first = y0 < geom.ysz && x0 < geom.xsz ? (geom.ysz-1-y0)*pitch+x0+offset : -1;
	if (first >= 0 && FABS(dem[first]-nodata) < C(1e-6)) first = -1;
	for (int y=y0; y<y0+side; ++y)
	for (int x=x0; x<x0+side; ++x)
	{
		int idx = y < geom.ysz && x < geom.xsz ? (geom.ysz-1-y)*pitch+x+offset : -1;
		if (idx >= 0 && FABS(dem[idx]-nodata) < C(1e-6)) idx = -1;
		bool different = (idx < 0) != (first < 0);
		if (idx >= 0 && first >= 0)
		{
			if (p.rate_grid) different |= p.rate_grid[idx] != p.rate_grid[first];
			if (p.greenampt)
			{
				different |= p.ks[idx] != p.ks[first] || p.psi[idx] != p.psi[first]
					|| p.dtheta[idx] != p.dtheta[first];
				if (p.fmax) different |= p.fmax[idx] != p.fmax[first];
			}
		}
		if (different) { details[node] = true; return; }
	}
}

/** Each leaf owns disjoint finest children. Evaluate nonlinear soil losses on
 * those children, then restrict the depth change to the leaf average. F and
 * compensation stay attached to raster cells through all mesh changes. */
__global__ void apply_losses_adaptive(
	AssembledSolution sol, int L, Geometry geom, int pitch, int offset,
	LossParams p, NUMERIC_TYPE step_dt, NUMERIC_TYPE threshold,
	const NUMERIC_TYPE* dem, NUMERIC_TYPE nodata, double* volume, double* water)
{
	int cell = blockIdx.x * blockDim.x + threadIdx.x;
	double removed = 0.0, stored = 0.0;
	if (cell < sol.length)
	{
		int level = sol.levels[cell];
		unsigned code = (sol.act_idcs[cell] - get_lvl_idx(level)) << (2 * (L-level));
		int x0 = compact(code), y0 = compact(code >> 1);
		int side = 1 << (L-level);
		NUMERIC_TYPE h = sol.h0[cell];
		double loss = 0.0;
		int valid = 0;
		for (int y=y0; y<y0+side && y<geom.ysz; ++y)
		for (int x=x0; x<x0+side && x<geom.xsz; ++x)
		{
			// Adaptive coordinates start at the bottom; ASCII rasters at the top.
			int idx = (geom.ysz-1-y)*pitch + x + offset;
			if (FABS(dem[idx]-nodata) < C(1e-6)) continue;
			NUMERIC_TYPE child = depth_after_losses(p, idx, h, step_dt, threshold);
			loss += (double)(h-child);
			++valid;
		}
		// Padding/NoData is excluded from the physical water and soil area.
		double count = valid;
		NUMERIC_TYPE next = valid ? (NUMERIC_TYPE)((double)h-loss/count) : h;
		// Correct the extra rounding introduced by restricting to the leaf.
		NUMERIC_TYPE correction = valid ? (NUMERIC_TYPE)((loss-count*(double)(h-next))/valid) : C(0.0);
		for (int y=y0; y<y0+side && y<geom.ysz; ++y)
		for (int x=x0; x<x0+side && x<geom.xsz; ++x)
		{
			int idx = (geom.ysz-1-y)*pitch + x + offset;
			if (FABS(dem[idx]-nodata) < C(1e-6)) continue;
			p.residual[idx] += correction;
		}
		if (next != h)
		{
			sol.h0[cell] = next;
			NUMERIC_TYPE scale = next > threshold ? next/h : C(0.0);
			sol.qx0[cell] *= scale; sol.qy0[cell] *= scale;
			if (sol.h1x != nullptr)
			{
				const NUMERIC_TYPE face = SQRT(C(3.0));
				bool dry = next <= threshold || face*FABS(sol.h1x[cell]) >= next
					|| face*FABS(sol.h1y[cell]) >= next;
				if (dry) { sol.h1x[cell] = C(0.0); sol.h1y[cell] = C(0.0); }
				sol.qx1x[cell] *= dry ? C(0.0) : scale;
				sol.qx1y[cell] *= dry ? C(0.0) : scale;
				sol.qy1x[cell] *= dry ? C(0.0) : scale;
				sol.qy1y[cell] *= dry ? C(0.0) : scale;
			}
		}
		removed = (double)(h-next)*count*geom.dx*geom.dy;
		stored = (double)next*valid*geom.dx*geom.dy;
	}
	add_block_sum(removed, volume);
	add_block_sum(stored, water);
}

/** Apply raster-scale losses to the static non-uniform ACC leaves. The soil
 * state stays at finest-grid resolution, while the leaf stores their average
 * depth. */
__global__ void apply_losses_acc_nugrid(
	acc_nugrid::AssembledSolution sol, int L, Geometry geom, int pitch, int offset,
	LossParams p, NUMERIC_TYPE step_dt, NUMERIC_TYPE threshold,
	const NUMERIC_TYPE* dem, NUMERIC_TYPE nodata, double* volume, double* water)
{
	int cell = blockIdx.x * blockDim.x + threadIdx.x;
	double removed = 0.0, stored = 0.0;
	if (cell < sol.length)
	{
		int level = sol.levels[cell];
		unsigned code = (sol.act_idcs[cell] - get_lvl_idx(level)) << (2 * (L-level));
		int x0 = compact(code), y0 = compact(code >> 1);
		int side = 1 << (L-level);
		NUMERIC_TYPE h = sol.h[cell];
		double loss = 0.0;
		int valid = 0;
		for (int y=y0; y<y0+side && y<geom.ysz; ++y)
		for (int x=x0; x<x0+side && x<geom.xsz; ++x)
		{
			int idx = (geom.ysz-1-y)*pitch + x + offset;
			if (FABS(dem[idx]-nodata) < C(1e-6)) continue;
			NUMERIC_TYPE child = depth_after_losses(p, idx, h, step_dt, threshold);
			loss += (double)(h-child);
			++valid;
		}
		double count = valid;
		NUMERIC_TYPE next = valid ? (NUMERIC_TYPE)((double)h-loss/count) : h;
		NUMERIC_TYPE correction = valid
			? (NUMERIC_TYPE)((loss-count*(double)(h-next))/valid) : C(0.0);
		for (int y=y0; y<y0+side && y<geom.ysz; ++y)
		for (int x=x0; x<x0+side && x<geom.xsz; ++x)
		{
			int idx = (geom.ysz-1-y)*pitch + x + offset;
			if (FABS(dem[idx]-nodata) < C(1e-6)) continue;
			p.residual[idx] += correction;
		}
		if (next != h) sol.h[cell] = next;
		removed = (double)(h-next)*count*geom.dx*geom.dy;
		stored = (double)next*count*geom.dx*geom.dy;
	}
	add_block_sum(removed, volume);
	add_block_sum(stored, water);
}

/** true and the value if spec is a number (and nothing else) */
static bool parse_number(const char* spec, double& value)
{
	char* end = nullptr;
	value = strtod(spec, &end);
	if (end == spec) return false;
	while (*end == ' ' || *end == '\t') end++;
	return *end == '\0';
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
geometry(geometry),
pitch(pitch),
offset(offset),
acc(states.acceleration == ON)
{
	infiltration = (states.calc_infiltration == ON);
	greenampt = (states.calc_greenampt == ON);
	evaporation = (states.calc_evap == ON);
	if (!enabled()) return;

	if (infiltration)
	{
		if (strlen(filenames.infilfilename) > 0)
		{
			// per-cell rate, mm/h -> m/s
			NUMERIC_TYPE* rate = host_grid(filenames.infilfilename, C(3600000.0),
					"infilfile", verbose);
			d_rate = to_device(rate);
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

	if (greenampt)
	{
		const char* missing = nullptr;
		if (strlen(filenames.greenampt_ks) == 0) missing = "greenampt_ks";
		else if (strlen(filenames.greenampt_psi) == 0) missing = "greenampt_psi";
		else if (strlen(filenames.greenampt_porosity) == 0) missing = "greenampt_porosity";
		else if (strlen(filenames.greenampt_theta_init) == 0) missing = "greenampt_theta_init";
		if (missing != nullptr)
		{
			fprintf(stderr, "ERROR: infiltration greenampt needs %s (a number or a grid file)\n",
					missing);
			exit(-1);
		}
		NUMERIC_TYPE* ks = host_grid(filenames.greenampt_ks, C(3600000.0),
				"greenampt_ks", verbose);       // mm/h -> m/s
		NUMERIC_TYPE* psi = host_grid(filenames.greenampt_psi, C(1000.0),
				"greenampt_psi", verbose);      // mm -> m
		NUMERIC_TYPE* porosity = host_grid(filenames.greenampt_porosity, C(1.0),
				"greenampt_porosity", verbose);
		NUMERIC_TYPE* theta = host_grid(filenames.greenampt_theta_init, C(1.0),
				"greenampt_theta_init", verbose);
		NUMERIC_TYPE* depth = (strlen(filenames.greenampt_soil_depth) > 0)
			? host_grid(filenames.greenampt_soil_depth, C(1.0),
					"greenampt_soil_depth", verbose)
			: nullptr;

		// dtheta = porosity - initial moisture (not below 0); fmax = depth x dtheta
		long wetter = 0;
		for (int j=0; j<geometry.ysz; j++)
		{
			for (int i=0; i<geometry.xsz; i++)
			{
				int idx = j*pitch + i + offset;
				NUMERIC_TYPE d = porosity[idx] - theta[idx];
				if (d < C(0.0)) { d = C(0.0); wetter++; }
				porosity[idx] = d;
				if (depth != nullptr) depth[idx] *= d;
			}
		}
		if (wetter > 0)
		{
			fprintf(stderr, "WARNING: greenampt_theta_init > greenampt_porosity in %ld cells; "
					"there the soil is taken as saturated\n", wetter);
		}
		d_ks = to_device(ks);
		d_psi = to_device(psi);
		d_dtheta = to_device(porosity);
		if (depth != nullptr) d_fmax = to_device(depth);
		delete[] ks; delete[] psi; delete[] porosity; delete[] theta; delete[] depth;

		int elements = acc ? lis::GhostRaster::elements_H(geometry)
			: lis::GhostRaster::elements(geometry);
		double* zeros = new double[elements]();
		d_F = static_cast<double*>(malloc_device(elements*sizeof(double)));
		copy(d_F, zeros, elements*sizeof(double));
		delete[] zeros;
		if (verbose) printf("Infiltration: Green-Ampt%s\n",
				(d_fmax != nullptr) ? ", with soil depth" : "");
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
	d_residual = to_device(zeros);
	delete[] zeros;

	d_volume = static_cast<double*>(malloc_device(sizeof(double)));
	double zero = 0.0;
	copy(d_volume, &zero, sizeof(double));
}

NUMERIC_TYPE* lis::cuda::Losses::host_grid
(
	const char* spec,
	NUMERIC_TYPE divisor,
	const char* keyword,
	int verbose
)
{
	NUMERIC_TYPE* grid = acc ? lis::GhostRaster::allocate_H(geometry)
		: lis::GhostRaster::allocate(geometry);
	double number;
	if (parse_number(spec, number))
	{
		NUMERIC_TYPE v = (NUMERIC_TYPE)number;
		v = (v > C(0.0)) ? v/divisor : C(0.0);
		for (int j=0; j<geometry.ysz; j++)
		{
			for (int i=0; i<geometry.xsz; i++) grid[j*pitch + i + offset] = v;
		}
		if (verbose) printf("%s = %s (uniform)\n", keyword, spec);
		return grid;
	}

	FILE* file = fopen_or_die(spec, "rb", keyword, verbose);
	Geometry file_geometry;
	NUMERIC_TYPE no_data_value;
	AsciiRaster::read_header(file, file_geometry, no_data_value);
	AsciiRaster::match_cell_dimensions_or_die(geometry, file_geometry, keyword);
	AsciiRaster::read(file, grid, geometry, pitch, offset);
	fclose(file);
	AsciiRaster::replace_no_data(grid, geometry, pitch, offset, no_data_value, C(0.0));
	for (int j=0; j<geometry.ysz; j++)
	{
		for (int i=0; i<geometry.xsz; i++)
		{
			NUMERIC_TYPE& r = grid[j*pitch + i + offset];
			r = (r > C(0.0)) ? r/divisor : C(0.0);
		}
	}
	return grid;
}

NUMERIC_TYPE* lis::cuda::Losses::to_device
(
	NUMERIC_TYPE* host
)
{
	NUMERIC_TYPE* d;
	if (acc)
	{
		d = GhostRaster::allocate_device_H(geometry);
		GhostRaster::copy_H(d, host, geometry);
	}
	else
	{
		d = GhostRaster::allocate_device(geometry);
		GhostRaster::copy(d, host, geometry);
	}
	return d;
}

lis::cuda::LossParams lis::cuda::Losses::params
(
	NUMERIC_TYPE t
)
{
	LossParams p;
	p.greenampt = greenampt;
	p.rate_grid = d_rate;
	p.uniform_rate = infiltration ? uniform_rate : C(0.0);
	p.evap_rate = evaporation ? InterpolateTimeSeries(evap, t) : C(0.0);
	p.residual = d_residual;
	p.F = d_F;
	p.ks = d_ks;
	p.psi = d_psi;
	p.dtheta = d_dtheta;
	p.fmax = d_fmax;
	return p;
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
	apply_losses_fv1<<<grid_size, CUDA_BLOCK_SIZE>>>(H, HU, HV, DEM, params(t), d_volume);
}

void lis::cuda::Losses::apply_slopes
(
	NUMERIC_TYPE* H, NUMERIC_TYPE* H1x, NUMERIC_TYPE* H1y,
	NUMERIC_TYPE* HU, NUMERIC_TYPE* HU1x, NUMERIC_TYPE* HU1y,
	NUMERIC_TYPE* HV, NUMERIC_TYPE* HV1x, NUMERIC_TYPE* HV1y,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE face,
	NUMERIC_TYPE t,
	dim3 grid_size
)
{
	apply_losses_slopes<<<grid_size, CUDA_BLOCK_SIZE>>>(H, H1x, H1y, HU, HU1x, HU1y,
			HV, HV1x, HV1y, DEM, face, params(t), d_volume);
}

void lis::cuda::Losses::apply_acc
(
	NUMERIC_TYPE* H,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE t,
	dim3 grid_size
)
{
	apply_losses_acc<<<grid_size, CUDA_BLOCK_SIZE>>>(H, DEM, params(t), d_volume);
}

void lis::cuda::Losses::preflag_adaptive(bool* details, int L,
	const NUMERIC_TYPE* dem, NUMERIC_TYPE nodata)
{
	int nodes = get_lvl_idx(L);
	preflag_loss_details<<<(nodes+CUDA_BLOCK_SIZE-1)/CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(
		details, L, geometry, pitch, offset, params(C(0.0)), dem, nodata);
}

void lis::cuda::Losses::apply_adaptive(
	AssembledSolution& sol, int max_level, NUMERIC_TYPE step_dt,
	NUMERIC_TYPE threshold, NUMERIC_TYPE t, const NUMERIC_TYPE* dem, NUMERIC_TYPE nodata)
{
	if (d_water == nullptr) d_water = static_cast<double*>(malloc_device(sizeof(double)));
	double zero = 0.0;
	copy(d_water, &zero, sizeof(double));
	apply_losses_adaptive<<<(sol.length+CUDA_BLOCK_SIZE-1)/CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(
		sol, max_level, geometry, pitch, offset, params(t), step_dt, threshold,
		dem, nodata, d_volume, d_water);
}

void lis::cuda::Losses::apply_acc_nugrid(
	acc_nugrid::AssembledSolution& sol, int max_level, NUMERIC_TYPE step_dt,
	NUMERIC_TYPE threshold, NUMERIC_TYPE t, const NUMERIC_TYPE* dem, NUMERIC_TYPE nodata)
{
	if (d_water == nullptr) d_water = static_cast<double*>(malloc_device(sizeof(double)));
	double zero = 0.0;
	copy(d_water, &zero, sizeof(double));
	apply_losses_acc_nugrid<<<(sol.length+CUDA_BLOCK_SIZE-1)/CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(
		sol, max_level, geometry, pitch, offset, params(t), step_dt, threshold,
		dem, nodata, d_volume, d_water);
}

double lis::cuda::Losses::adaptive_water_volume()
{
	double water = 0.0;
	if (d_water != nullptr) copy(&water, d_water, sizeof(double));
	return water;
}

double lis::cuda::Losses::total_volume()
{
	if (d_volume == nullptr) return 0.0;
	double v;
	copy(&v, d_volume, sizeof(double));
#if defined(LISFLOOD_CUDA_MPI)
	int initialized = 0;
	MPI_Initialized(&initialized);
	if (initialized) {
		double global = 0.0;
		MPI_Allreduce(&v, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		v = global;
	}
#endif
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
	free_device(d_F);
	free_device(d_ks);
	free_device(d_psi);
	free_device(d_dtheta);
	free_device(d_fmax);
	free_device(d_volume);
	free_device(d_water);
}
