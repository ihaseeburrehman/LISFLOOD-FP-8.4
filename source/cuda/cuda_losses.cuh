#pragma once
#include "../lisflood.h"
#include "../geometry.h"
#include "cuda_solver.cuh"

struct AssembledSolution;

namespace lis
{
namespace cuda
{

namespace acc_nugrid { struct AssembledSolution; }

/**
 * What the loss kernels need for one time step (passed by value).
 *
 * Constant rate (greenampt == false): rate_grid or uniform_rate, m/s.
 * Green-Ampt (greenampt == true): per-cell grids ks (m/s), psi (m),
 * dtheta (porosity - initial moisture) and, optionally, fmax (m, the most
 * the soil can take: soil depth x dtheta), and the state F (m, infiltrated
 * depth so far, double precision).
 * Both: evap_rate (m/s, uniform), and residual (m, the part of the loss that
 * rounding to single precision dropped, carried to the next step).
 */
struct LossParams
{
	bool greenampt;
	NUMERIC_TYPE* rate_grid;
	NUMERIC_TYPE uniform_rate;
	NUMERIC_TYPE evap_rate;
	NUMERIC_TYPE* residual;
	double* F;
	NUMERIC_TYPE* ks;
	NUMERIC_TYPE* psi;
	NUMERIC_TYPE* dtheta;
	NUMERIC_TYPE* fmax;
};

/**
 * Floodplain infiltration and evaporation for the CUDA FV1, FV2, DG2 and ACC
 * solvers (8.4). Two infiltration models, chosen in the .par file:
 *
 * Constant rate, as the CPU routines FPInfiltration and Evaporation
 * (infevap.cpp): every wet cell (depth > DepthThresh, not NoData) loses
 * (infiltration rate + evaporation rate) * dt of depth, never below zero.
 *   infiltration <rate>   uniform rate, m/s
 *   infilfile <file>      rate per cell, ESRI ASCII grid in mm/h (NoData and
 *                         negative values: 0); used instead of the uniform rate
 *
 * Green-Ampt (infiltration greenampt), on every cell with water:
 *   f = Ks (1 + (psi + h) dtheta / F),  dtheta = porosity - initial moisture,
 * with h the ponded depth and F the depth infiltrated so far, advanced with
 * implicit Euler, which has a closed form per step:
 *   F_new = (a + sqrt(a^2 + 4 Ks dt (psi + h) dtheta)) / 2,  a = F + Ks dt,
 * limited by the water available and, optionally, by the soil depth.
 *   greenampt_ks <number|file>          saturated conductivity, mm/h
 *   greenampt_psi <number|file>         wetting-front suction, mm
 *   greenampt_porosity <number|file>    saturated moisture content, m3/m3
 *   greenampt_theta_init <number|file>  initial moisture content, m3/m3
 *   greenampt_soil_depth <number|file>  optional: soil depth, m (storage cap)
 * The soil does not recover between rain events (F never decreases).
 *
 * Evaporation, with either model:
 *   evaporation <file>    time series in mm/day, uniform in space
 *
 * Depth is single precision and the loss per step is often close to its
 * resolution, so each cell carries the part of the loss that rounding dropped
 * to its next step (compensated subtraction). The volume removed is summed on
 * the GPU for the .mass file.
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

	bool enabled() const { return infiltration || greenampt || evaporation; }

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

	/**
	 * FV2 and DG2: the loss lowers the cell average of the depth; the depth
	 * slopes are kept (a uniform loss keeps the water surface flat) and the
	 * discharge average and slopes are scaled with the depth; if a face
	 * would become dry (wet/dry edge), the cell becomes first order (all
	 * slopes zero); a cell that dries loses its slopes and discharge.
	 * face: the face value is average +- face x slope (FV2: 0.5, DG2: sqrt(3)).
	 */
	void apply_slopes
	(
		NUMERIC_TYPE* H, NUMERIC_TYPE* H1x, NUMERIC_TYPE* H1y,
		NUMERIC_TYPE* HU, NUMERIC_TYPE* HU1x, NUMERIC_TYPE* HU1y,
		NUMERIC_TYPE* HV, NUMERIC_TYPE* HV1x, NUMERIC_TYPE* HV1y,
		NUMERIC_TYPE* DEM,
		NUMERIC_TYPE face,
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

	/** Adaptive mesh: soil state remains in the original raster layout. */
	void apply_adaptive(AssembledSolution& solution, int max_level,
		NUMERIC_TYPE dt, NUMERIC_TYPE threshold, NUMERIC_TYPE t,
		const NUMERIC_TYPE* dem, NUMERIC_TYPE nodata);
	/** Static non-uniform ACC mesh; evaluates losses on finest raster children. */
	void apply_acc_nugrid(acc_nugrid::AssembledSolution& solution, int max_level,
		NUMERIC_TYPE dt, NUMERIC_TYPE threshold, NUMERIC_TYPE t,
		const NUMERIC_TYPE* dem, NUMERIC_TYPE nodata);
	double adaptive_water_volume();
	void preflag_adaptive(bool* details, int max_level,
		const NUMERIC_TYPE* dem, NUMERIC_TYPE nodata);

	/** volume removed since the previous call (m3) */
	double take_interval_volume();

	/** volume removed since the start (m3) */
	double total_volume();

	~Losses();

private:
	LossParams params(NUMERIC_TYPE t);

	/** a grid in the layout of H, on the host: a number for every cell, or a
	 * file (NoData and negative values: 0); values divided by divisor */
	NUMERIC_TYPE* host_grid(const char* spec, NUMERIC_TYPE divisor,
			const char* keyword, int verbose);
	NUMERIC_TYPE* to_device(NUMERIC_TYPE* host);

	Geometry& geometry;
	int pitch;
	int offset;
	bool infiltration = false;
	bool greenampt = false;
	bool evaporation = false;
	bool acc = false;
	NUMERIC_TYPE uniform_rate = C(0.0);   /**< m/s */
	NUMERIC_TYPE* d_rate = nullptr;      /**< m/s per cell, or nullptr */
	NUMERIC_TYPE* d_residual = nullptr;  /**< loss not yet applied (rounding), m */
	double* d_F = nullptr;               /**< Green-Ampt: infiltrated depth, m */
	NUMERIC_TYPE* d_ks = nullptr;        /**< Green-Ampt: m/s */
	NUMERIC_TYPE* d_psi = nullptr;       /**< Green-Ampt: m */
	NUMERIC_TYPE* d_dtheta = nullptr;    /**< Green-Ampt: - */
	NUMERIC_TYPE* d_fmax = nullptr;      /**< Green-Ampt: m, or nullptr */
	::TimeSeries* evap = nullptr;        /**< m/s */
	double* d_volume = nullptr;          /**< cumulative volume removed, m3 */
	double* d_water = nullptr;
	double reported = 0.0;
};

}
}
