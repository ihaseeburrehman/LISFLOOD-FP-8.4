#include "cuda_dem.cuh"
#include "cuda_sample.cuh"
#include "cuda_simulate.cuh"
#include "cuda_solver.cuh"
#include "cuda_util.cuh"
#include "cuda_fv2_dem.cuh"
#include "cuda_fv2_flow.cuh"
#include "cuda_fv2_simulate.cuh"
#include "cuda_fv2_snapshot.cuh"
#include "cuda_fv2_solver.cuh"
#include "cuda_max_field.cuh"
#include "cuda_rain.cuh"
#include "cuda_stats.cuh"
#include "cuda_losses.cuh"
#include "cuda_unifiedallocator.cuh"
#include "../rain/rain.h"
#include "io.h"
#include <algorithm>
#include <cstdio>
#include <ctime>
#if defined(LISFLOOD_CUDA_MPI)
#include <mpi.h>
#include <stdexcept>
#endif

void lis::cuda::fv2::Simulation::run
(
	Fnames& filenames,
	States& states,
	Pars& pars,
	::Solver& solver,
	int verbose
)
{
#if defined(LISFLOOD_CUDA_MPI)
	int initialized = 0;
	MPI_Initialized(&initialized);
	const bool initialized_here = !initialized;
	if (initialized_here) MPI_Init(nullptr, nullptr);
	int mpi_rank = 0, mpi_size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	if (mpi_size != states.cuda_mpi_gpus) {
		fprintf(stderr,
			"ERROR: parameter GPUs %d requires launching exactly %d MPI ranks; "
			"current launch has %d. Example: mpirun -np %d lisflood case.par\n",
			states.cuda_mpi_gpus, states.cuda_mpi_gpus, mpi_size,
			states.cuda_mpi_gpus);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	int visible_gpus = 0;
	checkCudaErrors(cudaGetDeviceCount(&visible_gpus));
	if (visible_gpus < 1) throw std::runtime_error("No CUDA device visible to MPI rank");
	const int device = mpi_rank % visible_gpus;
	checkCudaErrors(cudaSetDevice(device));
	cudaDeviceProp device_properties{};
	checkCudaErrors(cudaGetDeviceProperties(&device_properties, device));
	printf("CUDA FV2 MPI rank %d/%d bound to GPU %d: %s\n",
		mpi_rank, mpi_size, device, device_properties.name);
#endif
	if (
#if defined(LISFLOOD_CUDA_MPI)
		mpi_rank == 0 &&
#endif
		verbose) print_device_info();
	dim3 grid_size(128, 128);

	Geometry geometry;
	int pitch = 0;
	int offset = 0;

	Topography DEM(filenames.demfilename, geometry, pitch, offset,
			pars.nodata_elevation, states.acceleration, verbose);

	update_geometry(pars, geometry);
#if defined(LISFLOOD_CUDA_MPI)
	int mpi_row_begin = 1 + (geometry.ysz * mpi_rank) / mpi_size;
	int mpi_row_end   = 1 + (geometry.ysz * (mpi_rank + 1)) / mpi_size;
	if (mpi_row_end <= mpi_row_begin)
		throw std::runtime_error("More MPI ranks than FV2 rows");
	configure_mpi_partition(mpi_row_begin, mpi_row_end, mpi_rank, mpi_size, pitch);
	const int owned_rows = mpi_row_end - mpi_row_begin;
#else
	const int owned_rows = geometry.ysz;
#endif
	// launch only enough blocks for this rank's cells: the kernels stride over
	// larger grids, and blocks beyond the domain would start only to exit
	grid_size = cuda::tile_grid(geometry.xsz, owned_rows);
	const dim3 cell_grid = cuda::row_grid(geometry.xsz, owned_rows);
	cuda::copy_to_symbol(cuda::geometry, &geometry, sizeof(Geometry));
	cuda::copy_to_symbol(cuda::pitch, &pitch, sizeof(int));

	DeviceTopography d_DEM;
	d_DEM.initialise(DEM, geometry);

	cuda::sync();

    ::DynamicRain<cuda::UnifiedAllocator<NUMERIC_TYPE>> rain(
			filenames.dynamicrainfilename, verbose);
	cuda::copy_to_symbol(cuda::rain_geometry, &rain.geometry(), sizeof(Geometry));

	Flow U;
	Flow::allocate_pinned(U, geometry);
	initialise_H(U.H, filenames.startfilename, states, DEM._0,
			geometry, pitch, offset, verbose);
	initialise_discharge(U.HU, U.HV, filenames.startfilename, states,
			geometry, pitch, offset, verbose);

	NUMERIC_TYPE* manning = nullptr;
	NUMERIC_TYPE* d_manning = nullptr;
	if (strlen(filenames.nfilename) > 0)
	{
		manning = lis::GhostRaster::allocate(geometry);
		initialise_manning(manning, filenames.nfilename, geometry,
				pitch, offset, verbose);
		d_manning = lis::cuda::GhostRaster::allocate_device(geometry);
		GhostRaster::copy(d_manning, manning, geometry);
	}

	PhysicalParams physical_params(pars, solver);
	cuda::copy_to_symbol(cuda::physical_params, &physical_params,
			sizeof(PhysicalParams));

	SolverParams solver_params(pars, solver, states);
	cuda::copy_to_symbol(cuda::solver_params, &solver_params,
			sizeof(SolverParams));

	BoundCs boundCs;
	load_boundaries(filenames, states, pars, boundCs, verbose);
	cuda::Boundary::initialise(cuda::boundaries, boundCs, pitch, offset);

	auto solve = unique_ptr<cuda::Solver<Flow>>(new Solver(U, d_DEM, d_manning,
				geometry, physical_params, pars.limit_slopes, grid_size));

	DynamicTimestep<Flow> dynamic_dt(cuda::dt, geometry, solver.InitTstep,
			states.adaptive_ts, *solve
#if defined(LISFLOOD_CUDA_MPI)
			, mpi_row_begin * pitch, (mpi_row_end - mpi_row_begin) * pitch
#endif
			);
	solver.MinTstep = solver.InitTstep;

	StatsCollector stats_collector(geometry, solver.DepthThresh, states.acceleration
#if defined(LISFLOOD_CUDA_MPI)
			, (mpi_row_begin - (mpi_rank == 0 ? 1 : 0)) * pitch,
			(mpi_row_end - mpi_row_begin + (mpi_rank == 0 ? 1 : 0)
			 + (mpi_rank + 1 == mpi_size ? 1 : 0)) * pitch
#endif
			);
	Losses losses(filenames, states, pars, geometry, pitch, offset, verbose);
	MaxField max_field(filenames.resrootname, geometry, pitch, offset,
			cell_grid, pars.output_precision);

	Stats stats(filenames.resrootname, pars.MassInt, pars.MassTotal, states.checkpoint, solver.t
#if defined(LISFLOOD_CUDA_MPI)
			, mpi_rank == 0
#else
			, true
#endif
			);
	stats.write_header(solver.t);

	Sampler sampler(cuda::sample_points, cuda::sample_buf_idx, verbose);
	if (states.save_stages == ON)
	{
		sampler.load_sample_points(filenames.stagefilename, geometry,
				pitch, offset);
#if defined(LISFLOOD_CUDA_MPI)
		if (mpi_rank == 0) {
#endif
		sampler.open_stage_file(filenames.resrootname, states.checkpoint, solver.t);
		sampler.write_stage_header(DEM._0, filenames.stagefilename, states.checkpoint, solver.t);

		if (states.voutput_stage == ON)
		{
			sampler.open_gauge_file(filenames.resrootname, states.checkpoint, solver.t);
			sampler.write_gauge_header(DEM._0, filenames.gaugefilename, solver.t);
		}
#if defined(LISFLOOD_CUDA_MPI)
		}
#endif
	}

	Snapshot snapshot(filenames.resrootname, pars.SaveInt,
			pars.SaveTotal, pars.SaveNo, DEM, U, geometry, pitch, offset,
			verbose, pars.output_precision);
	if (states.save_elev == ON)
	{
		snapshot.enable_elevation_writer();
	}
	if (states.voutput == ON)
	{
		snapshot.enable_velocity_writer(solver.DepthThresh);
	}
	if (states.save_Qs == ON)
	{
		snapshot.enable_discharge_writer();
	}

	Flow& d_U = solve->d_U();

	time_t loop_start;
	time(&loop_start);

	while (solver.t < solver.Sim_Time)
	{
		if (rain.enabled())
		{
			rain.update_time(solver.t);
			lis::cuda::DynamicRain::update<<<cell_grid, CUDA_BLOCK_SIZE>>>(
					d_DEM._0, d_U.H, rain.data());
		}
		stats_collector.zero_instantaneous_mass();
		cuda::Boundary::update_time_series(solver.t);
		cuda::Boundary::update_point_sources(d_U.H, DEM._0,
				stats_collector.instantaneous_mass());
#if defined(LISFLOOD_CUDA_MPI)
		exchange_mpi_halos(d_U, pitch);
#endif
		stats_collector.accumulate_mass();
		stats_collector.zero_instantaneous_mass();
		solve->update_ghost_cells();
		dynamic_dt.update_dt();
		solver.MinTstep = FMIN(solver.MinTstep, cuda::dt);

		if (
#if defined(LISFLOOD_CUDA_MPI)
			mpi_rank == 0 &&
#endif
			verbose) printf("t=%f\tdt=%f\n", solver.t, cuda::dt);
		solver.t += cuda::dt;
		solver.itCount++;

		d_U = solve->update_flow_variables(
				stats_collector.instantaneous_mass());
		if (pars.drain_nodata)
		{
			cuda::Boundary::drain_nodata_water(d_U.H, d_DEM._0,
					stats_collector.instantaneous_mass(), cell_grid);
		}
		if (losses.enabled())
		{
			losses.apply_slopes(d_U.H, d_U.H1x, d_U.H1y, d_U.HU, d_U.HU1x, d_U.HU1y,
					d_U.HV, d_U.HV1x, d_U.HV1y, d_DEM._0, C(0.5), solver.t, cell_grid);
		}
		stats_collector.accumulate_mass();
		max_field.update(d_U.H);

		bool gathered_depth = false;
		bool gathered_full_state = false;
		if (stats.need_to_write(solver.t))
		{
#if defined(LISFLOOD_CUDA_MPI)
			if (states.save_stages == ON) {
				gather_fv2_rows(d_U, geometry, pitch, mpi_rank, mpi_size);
				gathered_depth = true;
				gathered_full_state = true;
			}
			if (mpi_rank == 0) {
#endif
			sampler.sample(d_U.H, d_U.HU, d_U.HV, solver.t);
#if defined(LISFLOOD_CUDA_MPI)
			}
#endif
			solve->zero_ghost_cells();
			cuda::sync();

			if (losses.enabled())
			{
				stats_collector.interval_loss = losses.take_interval_volume();
				stats_collector.total_loss = losses.total_volume();
			}
			stats.write(stats_collector.create_entry(solver, d_U.H));
#if defined(LISFLOOD_CUDA_MPI)
			if (mpi_rank == 0)
#endif
			sampler.write_if_buffer_full();
		}

#if defined(LISFLOOD_CUDA_MPI)
		const bool snapshot_due = solver.t >= snapshot.next_save;
		if (snapshot_due && !gathered_full_state) {
			gather_fv2_rows(d_U, geometry, pitch, mpi_rank, mpi_size);
			gathered_depth = true;
			gathered_full_state = true;
		}
		if (mpi_rank == 0)
			snapshot.write_if_needed(d_U, solver.t, states.call_gzip);
		else if (snapshot_due)
			snapshot.next_save += pars.SaveInt;
#else
		snapshot.write_if_needed(d_U, solver.t, states.call_gzip);
#endif
	}

	time_t loop_end;
	time(&loop_end);
	NUMERIC_TYPE seconds = difftime(loop_end, loop_start);
	if (
#if defined(LISFLOOD_CUDA_MPI)
		mpi_rank == 0
#else
		true
#endif
	) printf("loop time %lf\n", seconds);

#if defined(LISFLOOD_CUDA_MPI)
	if (mpi_rank == 0)
#endif
	sampler.write();
	snapshot.wait();
	max_field.write();

	d_DEM.free_device();
	delete[] manning;
	cuda::free_device(d_manning);
	Flow::free_pinned(U);
	cuda::Boundary::free_device(boundaries);

#if defined(LISFLOOD_CUDA_MPI)
	if (initialized_here) MPI_Finalize();
#endif
}
