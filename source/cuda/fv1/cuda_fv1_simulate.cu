#include "cuda_dem.cuh"
#include "cuda_sample.cuh"
#include "cuda_simulate.cuh"
#include "cuda_fv1_flow.cuh"
#include "cuda_fv1_simulate.cuh"
#include "cuda_fv1_snapshot.cuh"
#include "cuda_fv1_solver.cuh"
#include "cuda_rain.cuh"
#include "cuda_max_field.cuh"
#include "cuda_stats.cuh"
#include "cuda_losses.cuh"
#include "cuda_unifiedallocator.cuh"
#include "../rain/rain.h"
#include "io.h"
#include <cstdio>
#include <ctime>
#if defined(LISFLOOD_CUDA_MPI)
#include <mpi.h>
#include <vector>
#include <stdexcept>
#endif
#include "../../rain/Rainfall_uniform.h"

#if defined(LISFLOOD_CUDA_MPI)
namespace {
void gather_fv1_rows(lis::cuda::fv1::Flow& flow, const Geometry& geometry,
		int pitch, int rank, int size)
{
	if (size <= 1) return;
	const int first = 1 + geometry.ysz * rank / size;
	const int last = 1 + geometry.ysz * (rank + 1) / size;
	std::vector<int> counts(size), displs(size);
	for (int r=0; r<size; ++r) {
		int lo = geometry.ysz * r / size, hi = geometry.ysz * (r+1) / size;
		counts[r] = (hi-lo)*geometry.xsz; displs[r] = lo*geometry.xsz;
	}
	const MPI_Datatype type = sizeof(NUMERIC_TYPE) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
	auto gather = [&](NUMERIC_TYPE* device) {
		std::vector<NUMERIC_TYPE> local((last-first)*geometry.xsz);
		std::vector<NUMERIC_TYPE> global(geometry.ysz*geometry.xsz);
		for (int j=first; j<last; ++j)
			lis::cuda::copy(local.data()+(j-first)*geometry.xsz,
				device+j*pitch+1, geometry.xsz*sizeof(NUMERIC_TYPE));
		MPI_Allgatherv(local.data(), local.size(), type, global.data(),
			counts.data(), displs.data(), type, MPI_COMM_WORLD);
		if (rank == 0)
			for (int j=1; j<=geometry.ysz; ++j)
				lis::cuda::copy(device+j*pitch+1, global.data()+(j-1)*geometry.xsz,
					geometry.xsz*sizeof(NUMERIC_TYPE));
	};
	gather(flow.H); gather(flow.HU); gather(flow.HV);
}
}
#endif

void lis::cuda::fv1::Simulation::run
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
			"ERROR: parameter GPUs %d requires launching exactly %d MPI ranks; current launch has %d. Example: mpirun -np %d lisflood case.par\n",
			states.cuda_mpi_gpus, states.cuda_mpi_gpus, mpi_size,
			states.cuda_mpi_gpus);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	int visible_gpus = 0;
	checkCudaErrors(cudaGetDeviceCount(&visible_gpus));
	if (visible_gpus < 1) throw std::runtime_error("No CUDA device visible to MPI rank");
	if (mpi_size > visible_gpus) {
		fprintf(stderr,
			"ERROR: requested %d CUDA FV1 ranks but this MPI launch exposes only %d GPU(s) per rank.\n",
			mpi_size, visible_gpus);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	const int device = mpi_rank % visible_gpus;
	checkCudaErrors(cudaSetDevice(device));
	cudaDeviceProp device_properties{};
	checkCudaErrors(cudaGetDeviceProperties(&device_properties, device));
	printf("CUDA FV1 MPI rank %d/%d bound to GPU %d: %s\n",
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

	NUMERIC_TYPE* DEM = Topography::load(filenames.demfilename, geometry,
			pitch, offset, pars.nodata_elevation, states.acceleration, verbose);
	Topography::clamp_boundary_values(DEM, geometry, pitch);

	update_geometry(pars, geometry);
#if defined(LISFLOOD_CUDA_MPI)
	int mpi_row_begin = 1 + (geometry.ysz * mpi_rank) / mpi_size;
	int mpi_row_end = 1 + (geometry.ysz * (mpi_rank + 1)) / mpi_size;
	if (mpi_row_end <= mpi_row_begin) throw std::runtime_error("More MPI ranks than FV1 rows");
	configure_mpi_partition(mpi_row_begin, mpi_row_end, mpi_rank, mpi_size);
#endif
	cuda::copy_to_symbol(cuda::geometry, &geometry, sizeof(Geometry));
	cuda::copy_to_symbol(cuda::pitch, &pitch, sizeof(int));

	NUMERIC_TYPE* d_DEM = GhostRaster::allocate_device(geometry);
	GhostRaster::copy(d_DEM, DEM, geometry);

	NUMERIC_TYPE* Zstar_x = GhostRaster::allocate_device(geometry);
	Topography::initialise_Zstar_x(Zstar_x, d_DEM);

	NUMERIC_TYPE* Zstar_y = GhostRaster::allocate_device(geometry);
	Topography::initialise_Zstar_y(Zstar_y, d_DEM);

	cuda::sync();

    ::DynamicRain<cuda::UnifiedAllocator<NUMERIC_TYPE>> rain(
			filenames.dynamicrainfilename, verbose);
	cuda::copy_to_symbol(cuda::rain_geometry, &rain.geometry(), sizeof(Geometry));

	fv1::Flow U;
	fv1::Flow::allocate_pinned(U, geometry);
	initialise_H(U.H, filenames.startfilename, states, DEM,
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

	Rainfall_uniform   rainfall_uniform(C(0.0));

	BoundCs boundCs;
	load_boundaries(filenames, states, pars, boundCs, verbose);
	cuda::Boundary::initialise(cuda::boundaries, boundCs, pitch, offset);

	auto solve = unique_ptr<cuda::Solver<Flow>>(new Solver(
				U, d_DEM, Zstar_x, Zstar_y, d_manning, geometry,
				physical_params, grid_size));

	DynamicTimestep<fv1::Flow> dynamic_dt(cuda::dt, geometry, solver.InitTstep,
			states.adaptive_ts, *solve);
	solver.MinTstep = solver.InitTstep;

	StatsCollector stats_collector(geometry, solver.DepthThresh, states.acceleration);
	Losses losses(filenames, states, pars, geometry, pitch, offset, verbose);
	MaxField max_field(filenames.resrootname, geometry, pitch, offset,
			grid_size, pars.output_precision);

	Stats stats(filenames.resrootname, pars.MassInt, pars.MassTotal, states.checkpoint, solver.t,
#if defined(LISFLOOD_CUDA_MPI)
			mpi_rank == 0
#else
			true
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
		sampler.write_stage_header(DEM, filenames.stagefilename, states.checkpoint, solver.t);

		if (states.voutput_stage == ON)
		{
			sampler.open_gauge_file(filenames.resrootname, states.checkpoint, solver.t);
			sampler.write_gauge_header(DEM, filenames.gaugefilename, solver.t);
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
			lis::cuda::DynamicRain::update<<<grid_size, CUDA_BLOCK_SIZE>>>(
					d_DEM, d_U.H, rain.data());
		}

		if (states.rainfall == ON) {

			rainfall_uniform.update_rainfall(filenames.rainfilename, solver.t);
			solve->update_uniform_rain(rainfall_uniform.value);
		}

		stats_collector.zero_instantaneous_mass();
		cuda::Boundary::update_time_series(solver.t);
		cuda::Boundary::update_point_sources(d_U.H, DEM,
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
			cuda::Boundary::drain_nodata_water(d_U.H, d_DEM,
					stats_collector.instantaneous_mass(), grid_size);
		}
		if (losses.enabled())
		{
			losses.apply_fv1(d_U.H, d_U.HU, d_U.HV, d_DEM, solver.t, grid_size);
		}
		stats_collector.accumulate_mass();
		max_field.update(d_U.H);

		bool gathered_for_output = false;
		if (stats.need_to_write(solver.t))
		{
#if defined(LISFLOOD_CUDA_MPI)
			gather_fv1_rows(d_U, geometry, pitch, mpi_rank, mpi_size);
			gathered_for_output = true;
			if (mpi_rank == 0) {
#endif
			sampler.sample(d_U.H, d_U.HU, d_U.HV, solver.t);
#if defined(LISFLOOD_CUDA_MPI)
			}
#endif
			solve->zero_ghost_cells();

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
		if (snapshot_due && !gathered_for_output)
			gather_fv1_rows(d_U, geometry, pitch, mpi_rank, mpi_size);
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

	delete[] DEM;
	delete[] manning;
	cuda::free_device(d_DEM);
	cuda::free_device(d_manning);
	cuda::free_device(Zstar_x);
	cuda::free_device(Zstar_y);
	Flow::free_pinned(U);
	cuda::Boundary::free_device(boundaries);
#if defined(LISFLOOD_CUDA_MPI)
	if (initialized_here) MPI_Finalize();
#endif
}
