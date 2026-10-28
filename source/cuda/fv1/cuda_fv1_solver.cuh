#pragma once
#include "../lisflood.h"
#include "params.h"
#include "stats.h"
#include "fv1/cuda_fv1_flow.cuh"
#include "cuda_sample.cuh"
#include "cuda_solver.cuh"
#include "cuda_losses.cuh"

namespace lis
{
namespace cuda
{
namespace fv1
{

#if defined(LISFLOOD_CUDA_MPI)
void configure_mpi_partition(int row_begin, int row_end, int rank, int size);
void exchange_mpi_halos(struct Flow& U, int pitch);
#endif

class Solver : public cuda::Solver<Flow>
{
public:
	Solver
	(
		Flow& U,
		NUMERIC_TYPE* DEM,
		NUMERIC_TYPE* Zstar_x,
		NUMERIC_TYPE* Zstar_y,
		NUMERIC_TYPE* manning,
		Geometry& geometry,
		PhysicalParams& physical_params,
		dim3 grid_size
	);

	void FloodplainQ(); 
	void zero_thin_depth_slopes();

	void zero_ghost_cells();

	void update_ghost_cells();
	
	void update_uniform_rain(NUMERIC_TYPE  rain_rate);

	void updateMaxFieldACC(NUMERIC_TYPE t);

	Flow& update_flow_variables
	(
		MassStats* mass_stats
	);

	Flow& d_U();

	void update_dt_per_element
	(
		NUMERIC_TYPE* dt_field
	) const;

	bool reduce_dt
	(
		NUMERIC_TYPE* min_dt,
		int first,
		int count
	) const;

	~Solver();

private:
	Flow U1;
	Flow U2;
	Flow Ux;
	Flow& Uold;
	Flow& U;
	NUMERIC_TYPE* DEM;
	NUMERIC_TYPE* Zstar_x;
	NUMERIC_TYPE* Zstar_y;
	NUMERIC_TYPE* manning;
	bool friction;
	bool fused;
	bool march;
	dim3 march_grid;
	dim3 march_grid_x;
	int strip;
public:
	/**
	 * the next reduce_dt also applies the dynamic rain (rain_data, or none if
	 * nullptr) and updates the ghost cells: the caller skips
	 * DynamicRain::update and update_ghost_cells for this step
	 */
	void prepare_fused_dt(NUMERIC_TYPE* rain_data, dim3 cell_grid)
	{
		fused_rain = rain_data;
		fused_cell_grid = cell_grid;
		fused_dt_pending = true;
	}
	bool fused_path() const { return fused; }
	/**
	 * set before update_flow_variables to apply the cell-wise updates of
	 * Losses::apply_fv1_cells inside the y flux kernel; returns false if the
	 * separate kernels run (then the caller applies them)
	 */
	bool set_cell_updates(NUMERIC_TYPE* maxH, bool drain, bool losses,
			const LossParams& p, double* volume)
	{
		cells = { maxH, drain, losses, p, volume };
		return fused;
	}
private:
	struct CellUpdates
	{
		NUMERIC_TYPE* maxH;
		bool drain;
		bool losses;
		LossParams p;
		double* volume;
	} cells = {};
	NUMERIC_TYPE* fused_rain = nullptr;
	dim3 fused_cell_grid;
	mutable bool fused_dt_pending = false;
	const dim3 grid_size;
	const int raster_pitch;
};

}
}
}
