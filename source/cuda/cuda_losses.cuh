#pragma once
#include "../lisflood.h"
#include "../geometry.h"
#include "cuda_solver.cuh"

namespace lis
{
namespace cuda
{

/**
 * Floodplain infiltration and evaporation for the CUDA FV1 and ACC solvers,
 * as in the CPU routines FPInfiltration and Evaporation (infevap.cpp):
 * every wet cell (depth > DepthThresh) outside NoData loses
 * (infiltration rate + evaporation rate) * dt of depth, never below zero.
 *
 * .par keywords, as for the CPU solvers:
 *   infiltration <rate>   uniform infiltration rate, m/s
 *   infilfile <file>      infiltration rate per cell, ESRI ASCII grid in mm/h
 *                         (same size as the DEM; NoData and negative values: 0);
 *                         used instead of the uniform rate
 *   evaporation <file>    evaporation time series in mm/day, uniform in space
 *
 * The volume removed is accumulated on the GPU for the .mass file.
 *
 * Depth is single precision, and the loss per time step is often close to
 * its resolution (e.g. 2 mm/h x 0.5 s = 3e-7 m on a 1 m deep cell), so every
 * subtraction is rounded, the same way each step. Each cell therefore keeps
 * the part of the loss that rounding dropped and adds it to its next step
 * (compensated subtraction): over many steps the depth lost equals
 * rate x time to within one rounding step.
 */
class Losses
{
public:
	Losses
	(
		Fnames& filenames,
		States& states,
		Pars& pars,
		Geometry& geometry,
		int pitch,
		int offset,
		int verbose
	);

	bool enabled() const { return infiltration || evaporation; }

	/** FV1: ghost-cell layout; scales HU and HV with the depth */
	void apply_fv1
	(
		NUMERIC_TYPE* H,
		NUMERIC_TYPE* HU,
		NUMERIC_TYPE* HV,
		NUMERIC_TYPE* DEM,
		NUMERIC_TYPE t,
		dim3 grid_size
	);

	/** ACC: layout without ghost cells; discharges sit on the faces */
	void apply_acc
	(
		NUMERIC_TYPE* H,
		NUMERIC_TYPE* DEM,
		NUMERIC_TYPE t,
		dim3 grid_size
	);

	/** volume removed since the previous call (m3) */
	double take_interval_volume();

	/** volume removed since the start (m3) */
	double total_volume();

	~Losses();

private:
	NUMERIC_TYPE evaporation_rate(NUMERIC_TYPE t);

	bool infiltration = false;
	bool evaporation = false;
	bool acc = false;
	NUMERIC_TYPE uniform_rate = C(0.0);   /**< m/s */
	NUMERIC_TYPE* d_rate = nullptr;      /**< m/s per cell, or nullptr */
	NUMERIC_TYPE* d_residual = nullptr;  /**< loss not yet applied (rounding), m */
	::TimeSeries* evap = nullptr;          /**< m/s */
	double* d_volume = nullptr;          /**< cumulative volume removed, m3 */
	double reported = 0.0;
};

}
}
