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
#include <algorithm>
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
		int pitch, int rank, int size, bool gather_depth = true,
		bool gather_momentum = true)
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
		std::vector<NUMERIC_TYPE> global;
		if (rank == 0) global.resize(geometry.ysz*geometry.xsz);
		for (int j=first; j<last; ++j)
			lis::cuda::copy(local.data()+(j-first)*geometry.xsz,
				device+j*pitch+1, geometry.xsz*sizeof(NUMERIC_TYPE));
		MPI_Gatherv(local.data(), local.size(), type,
			rank == 0 ? global.data() : nullptr, counts.data(), displs.data(),
			type, 0, MPI_COMM_WORLD);
		if (rank == 0)
			for (int j=1; j<=geometry.ysz; ++j)
				lis::cuda::copy(device+j*pitch+1, global.data()+(j-1)*geometry.xsz,
					geometry.xsz*sizeof(NUMERIC_TYPE));
	};
	if (gather_depth) gather(flow.H);
	if (gather_momentum) {
		gather(flow.HU);
		gather(flow.HV);
	}
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
			states.adaptive_ts, *solve
#if defined(LISFLOOD_CUDA_MPI)
			, mpi_row_begin * pitch, (mpi_row_end - mpi_row_begin) * pitch
#endif
			);
	solver.MinTstep = solver.InitTstep;
	NUMERIC_TYPE step_dt = C(0.0);

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
	stats_collector.start(d_U.H);

	// Dead cells (cuda_solver.cuh): NoData cells whose four neighbours are
	// NoData cells inside the domain. Only with drain_nodata, which keeps their
	// depth at zero, and without point sources. From the fourth step on, the
	// first steps having removed any start water from them and written the
	// same values into both states, blocks of dead cells skip their work.
	// The rain, the ghost cells and the time step in two kernels (Solver::
	// prepare_fused_dt), when nothing else changes the depths between them:
	// no point sources, no uniform rain, and an adaptive time step.
	Solver* fv1_solver = static_cast<Solver*>(solve.get());
	// With several ranks the halo exchange sits between the rain and the time
	// step and must see the rain: keep the separate kernels there.
	const bool fused_dt = fv1_solver->fused_path() && states.adaptive_ts == ON
		&& boundCs.numPS <= 0 && states.rainfall != ON
#if defined(LISFLOOD_CUDA_MPI)
		&& mpi_size == 1
#endif
		;

	// On that path the next rain frame is read in the background while the
	// GPU works (DynamicRain::start_prefetch) and copied to a device buffer.
	const bool rain_prefetch = fused_dt && rain.enabled();
	// and the host waits for the time step only after the step's kernels are
	// queued (DynamicTimestep::start_dt), unless the losses change with time
	const bool async_dt = fused_dt && !losses.depends_on_time();
	NUMERIC_TYPE* d_rain = nullptr;
	size_t rain_bytes = 0;
	if (rain_prefetch)
	{
		rain_bytes = static_cast<size_t>(rain.geometry().xsz) * rain.geometry().ysz
			* sizeof(NUMERIC_TYPE);
		d_rain = static_cast<NUMERIC_TYPE*>(cuda::malloc_device(rain_bytes));
		rain.start_prefetch();
		checkCudaErrors(cudaMemcpy(d_rain, rain.frame(), rain_bytes, cudaMemcpyHostToDevice));
	}

	int* d_live_sat = nullptr;
	const bool skip_dead = pars.drain_nodata && boundCs.numPS <= 0;
	bool skipping = false;
	if (skip_dead)
	{
		const int nx = geometry.xsz + 2, ny = geometry.ysz + 2;
		auto nodata = [&](int i, int j) {
			return FABS(DEM[j*pitch + i] - pars.nodata_elevation) < C(1e-6);
		};
		auto ghost = [&](int i, int j) {
			return i == 0 || j == 0 || i == nx - 1 || j == ny - 1;
		};
		auto dead = [&](int i, int j) {
			if (ghost(i, j) || !nodata(i, j)) return false;
			const int di[4] = {-1, 1, 0, 0}, dj[4] = {0, 0, -1, 1};
			for (int k = 0; k < 4; k++)
				if (ghost(i + di[k], j + dj[k]) || !nodata(i + di[k], j + dj[k])) return false;
			return true;
		};
		const int w = nx + 1;
		std::vector<int> sat(static_cast<size_t>(ny + 1) * w, 0);
		long dead_cells = 0;
		for (int j = 0; j < ny; j++)
		{
			int row = 0;
			for (int i = 0; i < nx; i++)
			{
				const bool d = dead(i, j);
				dead_cells += d;
				row += d ? 0 : 1;
				sat[(j + 1)*w + i + 1] = sat[j*w + i + 1] + row;
			}
		}
		d_live_sat = static_cast<int*>(cuda::malloc_device(sat.size() * sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_live_sat, sat.data(), sat.size() * sizeof(int),
				cudaMemcpyHostToDevice));
		cuda::copy_to_symbol(cuda::fv1_live_sat, &d_live_sat, sizeof(int*));
		if (verbose) printf("FV1: %ld of %ld cells are dead NoData cells and are skipped from step 4\n",
				dead_cells, static_cast<long>(nx) * ny);
	}

	time_t loop_start;
	time(&loop_start);

	while (solver.t < solver.Sim_Time)
	{
		if (skip_dead && !skipping && solver.itCount >= 3)
		{
			const int one = 1;
			cuda::copy_to_symbol(cuda::fv1_skip_dead, &one, sizeof(int));
			skipping = true;
		}

		if (rain_prefetch)
		{
			if (rain.advance_prefetched(solver.t))
				checkCudaErrors(cudaMemcpy(d_rain, rain.frame(), rain_bytes,
						cudaMemcpyHostToDevice));
		}
		else if (rain.enabled())
		{
			rain.update_time(solver.t);
			if (!fused_dt)
				lis::cuda::DynamicRain::update<<<cell_grid, CUDA_BLOCK_SIZE>>>(
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
		stats_collector.accumulate_mass(step_dt);
		stats_collector.zero_instantaneous_mass();
		if (fused_dt)
			fv1_solver->prepare_fused_dt(rain_prefetch ? d_rain : nullptr, cell_grid);
		else
			solve->update_ghost_cells();
		if (async_dt)
		{
			// The step's kernels read dt on the device, so they are launched
			// before the host has it; the loss parameters do not depend on t.
			dynamic_dt.start_dt();
			fv1_solver->set_cell_updates(
					max_field.data(), pars.drain_nodata, losses.enabled(),
					losses.cell_params(solver.t), losses.device_volume());
			d_U = solve->update_flow_variables(
					stats_collector.instantaneous_mass());
			step_dt = dynamic_dt.finish_dt();
		}
		else
		{
			step_dt = dynamic_dt.update_dt();
		}
		solver.MinTstep = FMIN(solver.MinTstep, step_dt);

		if (
#if defined(LISFLOOD_CUDA_MPI)
			mpi_rank == 0 &&
#endif
			verbose) printf("t=%f\tdt=%f\n", solver.t, step_dt);
		solver.t += step_dt;
		solver.itCount++;

		// drain, losses and maximum depth: inside the y flux kernel, or in one
		// pass over the cells after the separate kernels
		bool cells_fused = async_dt;
		if (!async_dt)
		{
			cells_fused = fv1_solver->set_cell_updates(
					max_field.data(), pars.drain_nodata, losses.enabled(),
					losses.cell_params(solver.t), losses.device_volume());
			d_U = solve->update_flow_variables(
					stats_collector.instantaneous_mass());
		}
		if (!cells_fused)
			losses.apply_fv1_cells(d_U.H, d_U.HU, d_U.HV, d_DEM, max_field.data(),
					pars.drain_nodata, stats_collector.instantaneous_mass(),
					solver.t, cell_grid);
		stats_collector.accumulate_mass(step_dt);

		bool gathered_depth = false;
		bool gathered_full_state = false;
		if (stats.need_to_write(solver.t))
		{
#if defined(LISFLOOD_CUDA_MPI)
			if (states.save_stages == ON) {
				gather_fv1_rows(d_U, geometry, pitch, mpi_rank, mpi_size,
						true, true);
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
			// Snapshot writers may need H, HU and HV. Reuse H if the mass
			// write above already gathered it.
			gather_fv1_rows(d_U, geometry, pitch, mpi_rank, mpi_size,
					!gathered_depth, true);
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

	delete[] DEM;
	delete[] manning;
	cuda::free_device(d_DEM);
	cuda::free_device(d_manning);
	cuda::free_device(Zstar_x);
	cuda::free_device(Zstar_y);
	if (d_live_sat != nullptr) cuda::free_device(d_live_sat);
	if (d_rain != nullptr) cuda::free_device(d_rain);
	Flow::free_pinned(U);
	cuda::Boundary::free_device(boundaries);
#if defined(LISFLOOD_CUDA_MPI)
	if (initialized_here) MPI_Finalize();
#endif
}
