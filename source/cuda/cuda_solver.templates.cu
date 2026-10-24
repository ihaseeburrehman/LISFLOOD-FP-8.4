#include "ghostraster.h"
#include "cuda_geometry.cuh"
#include "cuda_util.cuh"
#include <helper_cuda.h>
#include <cub/cub.cuh>
#if defined(LISFLOOD_CUDA_MPI)
#include <mpi.h>
#endif

template<typename F>
lis::cuda::DynamicTimestep<F>::DynamicTimestep
(
	NUMERIC_TYPE& dt,
	Geometry& geometry,
	NUMERIC_TYPE max_dt,
	int adaptive_ts,
	Solver<F>& solver,
	int element_offset,
	int element_count
)
:
dt(dt),
max_dt(max_dt),
adaptive(adaptive_ts == ON),
solver(solver),
element_offset(element_offset),
elements(element_count > 0 ? element_count : lis::GhostRaster::elements(geometry))
{
	if (adaptive)
	{
		d_temp = nullptr;
		NUMERIC_TYPE* dummy_in = nullptr;
		NUMERIC_TYPE* dummy_out = nullptr;

		checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
					dummy_in, dummy_out, elements));

		d_temp = malloc_device(bytes);

		dt_field = cuda::GhostRaster::allocate_device(geometry);
		d_reduced_dt = static_cast<NUMERIC_TYPE*>(malloc_device(sizeof(NUMERIC_TYPE)));
#if defined(LISFLOOD_CUDA_MPI)
		checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_reduced_dt),
				sizeof(NUMERIC_TYPE)));
#endif
	}
	else
	{
		dt = max_dt;
	}
}

template<typename F>
lis::cuda::DynamicTimestepACC<F>::DynamicTimestepACC
(
	NUMERIC_TYPE& dt,
	Geometry& geometry,
	NUMERIC_TYPE max_dt,
	int adaptive_ts,
	Solver<F>& solver
)
	:
	dt(dt),
	max_dt(max_dt),
	adaptive(adaptive_ts == ON),
	solver(solver),
	elements(lis::GhostRaster::elements_H(geometry))
{

		d_temp = nullptr;
		NUMERIC_TYPE* dummy_in = nullptr;
		NUMERIC_TYPE* dummy_out = nullptr;

		checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
			dummy_in, dummy_out, elements));

		d_temp = malloc_device(bytes);

		dt_field = cuda::GhostRaster::allocate_device_H(geometry);


}

template<typename F>
NUMERIC_TYPE lis::cuda::DynamicTimestep<F>::update_dt()
{
	if (adaptive)
	{
#if defined(LISFLOOD_CUDA_MPI)
		if (!solver.reduce_dt(d_reduced_dt, element_offset, elements))
		{
			solver.update_dt_per_element(dt_field);
			checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
						dt_field + element_offset, d_reduced_dt, elements));
		}
		checkCudaErrors(cudaMemcpyAsync(h_reduced_dt, d_reduced_dt,
				sizeof(NUMERIC_TYPE), cudaMemcpyDeviceToHost));
		// Wait only for the default stream that produced the reduction. This
		// avoids migrating the managed timestep to the CPU or synchronizing
		// unrelated device work.
		checkCudaErrors(cudaStreamSynchronize(0));
		int mpi_initialized = 0;
		MPI_Initialized(&mpi_initialized);
		if (mpi_initialized) {
			NUMERIC_TYPE global_dt;
			MPI_Datatype type = sizeof(NUMERIC_TYPE) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
			MPI_Allreduce(h_reduced_dt, &global_dt, 1, type, MPI_MIN, MPI_COMM_WORLD);
			*h_reduced_dt = global_dt;
		}
		checkCudaErrors(cudaMemcpyToSymbolAsync(lis::cuda::dt, h_reduced_dt,
				sizeof(NUMERIC_TYPE)));
		return *h_reduced_dt;
#else
		if (solver.reduce_dt(d_reduced_dt, element_offset, elements))
		{
			NUMERIC_TYPE h_dt;
			checkCudaErrors(cudaMemcpy(&h_dt, d_reduced_dt, sizeof(NUMERIC_TYPE),
					cudaMemcpyDeviceToHost));
			dt = h_dt;
		}
		else
		{
			solver.update_dt_per_element(dt_field);
			checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
						dt_field + element_offset, &dt, elements));
			// the host reads the managed dt next: wait for the reduction
			checkCudaErrors(cudaStreamSynchronize(0));
		}
#endif
	}

	return dt;
}

template<typename F>
NUMERIC_TYPE lis::cuda::DynamicTimestepACC<F>::update_dt_ACC()
{
		solver.update_dt_per_element(dt_field);
		checkCudaErrors(cub::DeviceReduce::Min(d_temp, bytes,
					dt_field, &dt, elements));

	cuda::sync(); 

	return dt;
}

template<typename F>
lis::cuda::DynamicTimestep<F>::~DynamicTimestep()
{
	if (adaptive)
	{
		cuda::free_device(dt_field);
		free_device(d_temp);
		free_device(d_reduced_dt);
#if defined(LISFLOOD_CUDA_MPI)
		checkCudaErrors(cudaFreeHost(h_reduced_dt));
#endif
	}
}

template<typename F>
lis::cuda::DynamicTimestepACC<F>::~DynamicTimestepACC()
{

		cuda::free_device(dt_field);
		free_device(d_temp);

}
