#pragma once
#include "cuda_dg2_dem.cuh"
#include "cuda_dg2_flow.cuh"
#include "cuda_dg2_slope_limit.cuh"
#include "cuda_solver.cuh"
#if defined(LISFLOOD_CUDA_MPI)
#include <mpi.h>
#endif

namespace lis
{
namespace cuda
{
namespace dg2
{

extern __managed__ NUMERIC_TYPE maxH;

class Solver : public cuda::Solver<Flow>
{
public:
	Solver
	(
		Flow& U,
		DeviceTopography& DEM,
		NUMERIC_TYPE* manning,
		Geometry& geometry,
		PhysicalParams& physical_params,
		bool limit_slopes,
		dim3 grid_size
	);

	void FloodplainQ(); 

	void zero_ghost_cells();

	void update_ghost_cells();
	
	void update_uniform_rain(NUMERIC_TYPE  rain_rate);
	
	void zero_thin_depth_slopes();

	void updateMaxFieldACC(NUMERIC_TYPE t);

	Flow& update_flow_variables
	(
		MassStats* mass_stats
	);

	Flow& d_U();
	Flow& d_Uint();

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
	void update_ghost_cells
	(
		Flow& U
	);
	
	void zero_thin_depth_slopes
	(
		Flow& U
	);	

	void zero_perimeter_slopes
	(
		DeviceTopography& DEM,
		Flow& flow
	);

	Flow U1;
	Flow Ux;
	Flow Uint;
	Flow& Uold;
	DeviceTopography& DEM;
	NUMERIC_TYPE* manning;
	Slopes slopes;
	MaxH maxH;
	bool friction;
	bool limit_slopes = false;
	const dim3 grid_size;
};

}
}
}

#if defined(LISFLOOD_CUDA_MPI)
namespace lis { namespace cuda { namespace dg2 {
void configure_mpi_partition(int row_begin, int row_end, int rank, int size, int pitch);
void exchange_mpi_halos(Flow& U, int pitch);
void gather_dg2_rows(Flow& U, const Geometry& geometry, int pitch, int rank, int size);
}}}
#endif
