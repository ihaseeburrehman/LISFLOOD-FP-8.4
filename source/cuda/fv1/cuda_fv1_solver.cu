#include "cuda_fv1_solver.cuh"
#include "cuda_boundary.cuh"
#include "cuda_hll.cuh"
#include "cuda_device_functions.cuh"
#include "cuda_solver.cuh"
#include "cuda_rain.cuh"
#include <algorithm>
#include <cstdlib>
#if defined(LISFLOOD_CUDA_MPI)
#include <mpi.h>
#endif

namespace lis
{
namespace cuda
{
namespace fv1
{

#if defined(LISFLOOD_CUDA_MPI)
namespace {
int mpi_row_begin = 1, mpi_row_end = 1, mpi_rank = 0, mpi_size = 1;

__global__ void pack_mpi_halo_rows(Flow U, int pitch, int north_row,
		int south_row, NUMERIC_TYPE* packed)
{
	const int n = 3 * pitch;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < 2 * n; idx += blockDim.x * gridDim.x) {
		const int side = idx / n;
		const int field_idx = idx - side * n;
		const int field = field_idx / pitch;
		const int column = field_idx - field * pitch;
		const int row = side == 0 ? north_row : south_row;
		const NUMERIC_TYPE* src = field == 0 ? U.H : (field == 1 ? U.HU : U.HV);
		packed[idx] = src[row * pitch + column];
	}
}

__global__ void unpack_mpi_halo_rows(Flow U, int pitch, int north_halo,
		int south_halo, bool has_north, bool has_south,
		const NUMERIC_TYPE* packed)
{
	const int n = 3 * pitch;
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < 2 * n; idx += blockDim.x * gridDim.x) {
		const int side = idx / n;
		if ((side == 0 && !has_south) || (side == 1 && !has_north)) continue;
		const int field_idx = idx - side * n;
		const int field = field_idx / pitch;
		const int column = field_idx - field * pitch;
		const int row = side == 0 ? south_halo : north_halo;
		NUMERIC_TYPE* dst = field == 0 ? U.H : (field == 1 ? U.HU : U.HV);
		dst[row * pitch + column] = packed[idx];
	}
}
}

void configure_mpi_partition(int row_begin, int row_end, int rank, int size)
{
	mpi_row_begin = row_begin; mpi_row_end = row_end; mpi_rank = rank; mpi_size = size;
	checkCudaErrors(cudaMemcpyToSymbol(cuda::fv1_row_begin, &row_begin, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(cuda::fv1_row_end, &row_end, sizeof(int)));
	const int active = 1;
	checkCudaErrors(cudaMemcpyToSymbol(cuda::fv1_partition_active, &active, sizeof(int)));
	const int north = rank == 0 ? 1 : 0;
	const int south = rank == size-1 ? 1 : 0;
	checkCudaErrors(cudaMemcpyToSymbol(cuda::fv1_owns_north_ghost, &north, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(cuda::fv1_owns_south_ghost, &south, sizeof(int)));
	int active_check = 0, begin_check = 0, end_check = 0;
	checkCudaErrors(cudaMemcpyFromSymbol(&active_check, cuda::fv1_partition_active, sizeof(int)));
	checkCudaErrors(cudaMemcpyFromSymbol(&begin_check, cuda::fv1_row_begin, sizeof(int)));
	checkCudaErrors(cudaMemcpyFromSymbol(&end_check, cuda::fv1_row_end, sizeof(int)));
	printf("CUDA FV1 MPI rank %d/%d owns rows [%d,%d), device partition=%d\n",
		rank, size, begin_check, end_check, active_check);
	fflush(stdout);
}

void exchange_mpi_halos(Flow& U, int pitch)
{
	if (mpi_size <= 1) return;
	const int n = 3 * pitch;
	static NUMERIC_TYPE* d_send = nullptr;
	static NUMERIC_TYPE* d_recv = nullptr;
	static NUMERIC_TYPE* h_send = nullptr;
	static NUMERIC_TYPE* h_recv = nullptr;
	static int allocated_values = 0;
	if (allocated_values != 2 * n) {
		if (d_send) checkCudaErrors(cudaFree(d_send));
		if (d_recv) checkCudaErrors(cudaFree(d_recv));
		if (h_send) checkCudaErrors(cudaFreeHost(h_send));
		if (h_recv) checkCudaErrors(cudaFreeHost(h_recv));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_send),
				2 * n * sizeof(NUMERIC_TYPE)));
		checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_recv),
				2 * n * sizeof(NUMERIC_TYPE)));
		checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_send),
				2 * n * sizeof(NUMERIC_TYPE)));
		checkCudaErrors(cudaMallocHost(reinterpret_cast<void**>(&h_recv),
				2 * n * sizeof(NUMERIC_TYPE)));
		allocated_values = 2 * n;
	}
	const int threads = 256;
	const int blocks = std::min(256, (2 * n + threads - 1) / threads);
	pack_mpi_halo_rows<<<blocks, threads>>>(U, pitch, mpi_row_begin,
			mpi_row_end - 1, d_send);
	const int north = mpi_rank > 0 ? mpi_rank - 1 : MPI_PROC_NULL;
	const int south = mpi_rank + 1 < mpi_size ? mpi_rank + 1 : MPI_PROC_NULL;
	// One transfer synchronizes the pack kernel and stages both boundary rows.
	checkCudaErrors(cudaMemcpy(h_send, d_send, 2 * n * sizeof(NUMERIC_TYPE),
			cudaMemcpyDeviceToHost));
	const MPI_Datatype type = sizeof(NUMERIC_TYPE) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
	MPI_Request requests[4];
	MPI_Irecv(h_recv, n, type, south, 42, MPI_COMM_WORLD, &requests[0]);
	MPI_Irecv(h_recv + n, n, type, north, 43, MPI_COMM_WORLD, &requests[1]);
	MPI_Isend(h_send, n, type, north, 42, MPI_COMM_WORLD, &requests[2]);
	MPI_Isend(h_send + n, n, type, south, 43, MPI_COMM_WORLD, &requests[3]);
	MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
	checkCudaErrors(cudaMemcpyAsync(d_recv, h_recv,
			2 * n * sizeof(NUMERIC_TYPE), cudaMemcpyHostToDevice));
	unpack_mpi_halo_rows<<<blocks, threads>>>(U, pitch, mpi_row_begin - 1,
			mpi_row_end, mpi_rank > 0, mpi_rank + 1 < mpi_size, d_recv);
}
#endif

__global__ void zero_ghost_cells_north_south
(
	Flow U
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;

	for (int i=global_i; i<cuda::pitch; i+=blockDim.x*gridDim.x)
	{
		{
			int j=0;
			U.H[j*pitch + i] = C(0.0);
			U.HU[j*pitch + i] = C(0.0);
			U.HV[j*pitch + i] = C(0.0);
		}

		{
			int j=cuda::geometry.ysz+1;
			U.H[j*pitch + i] = C(0.0);
			U.HU[j*pitch + i] = C(0.0);
			U.HV[j*pitch + i] = C(0.0);
		}
	}
}

__global__ void zero_ghost_cells_east_west
(
	Flow U
)
{
	int global_j = blockIdx.x*blockDim.x + threadIdx.x;

	for (int j=global_j; j<cuda::geometry.ysz+2; j+=blockDim.x*gridDim.x)
	{
		{
			int i=0;
			U.H[j*pitch + i] = C(0.0);
			U.HU[j*pitch + i] = C(0.0);
			U.HV[j*pitch + i] = C(0.0);
		}

		{
			int i=cuda::geometry.xsz+1;
			U.H[j*pitch + i] = C(0.0);
			U.HU[j*pitch + i] = C(0.0);
			U.HV[j*pitch + i] = C(0.0);
		}
	}
}

__global__ void update_dt_per_element
(
	NUMERIC_TYPE* dt,
	Flow U
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int row_begin = 0;
	int row_end = cuda::geometry.ysz + 2;
#if defined(LISFLOOD_CUDA_MPI)
	if (cuda::fv1_partition_active) {
		row_begin = cuda::fv1_row_begin;
		row_end = cuda::fv1_row_end;
	}
#endif
	int global_j = row_begin + blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j; j<row_end; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<cuda::pitch; i+=blockDim.x*gridDim.x)
		{
			NUMERIC_TYPE H = U.H[j*cuda::pitch + i];

			if (H > cuda::solver_params.DepthThresh)
			{
				NUMERIC_TYPE HU = U.HU[j*cuda::pitch + i];
				NUMERIC_TYPE HV = U.HV[j*cuda::pitch + i];
				NUMERIC_TYPE U = HU/H;
				NUMERIC_TYPE V = HV/H;
				
				NUMERIC_TYPE dt_x = cuda::solver_params.cfl * cuda::geometry.dx
					/ (FABS(U)+SQRT(cuda::physical_params.g * H));
				NUMERIC_TYPE dt_y = cuda::solver_params.cfl * cuda::geometry.dy
					/ (FABS(V)+SQRT(cuda::physical_params.g * H));

				dt[j*cuda::pitch + i] = FMIN(dt_x, dt_y);
			}
			else
			{
				dt[j*cuda::pitch + i] = cuda::solver_params.max_dt;
			}
		}
	}
}

/**
 * update_dt_per_element and the minimum over elements [first, last) in one
 * kernel: each cell's time step is computed as there, and the minimum of the
 * same values is exact in any order
 */
__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
reduce_dt
(
	NUMERIC_TYPE* min_dt,
	Flow U,
	int first,
	int last
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int row_begin = 0;
	int row_end = cuda::geometry.ysz + 2;
#if defined(LISFLOOD_CUDA_MPI)
	if (cuda::fv1_partition_active) {
		row_begin = cuda::fv1_row_begin;
		row_end = cuda::fv1_row_end;
	}
#endif
	int global_j = row_begin + blockIdx.y*blockDim.y + threadIdx.y;
	NUMERIC_TYPE local = FLT_MAX;

	for (int j=global_j; j<row_end; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<cuda::pitch; i+=blockDim.x*gridDim.x)
		{
			int idx = j*cuda::pitch + i;
			if (idx < first || idx >= last) continue;
			const int ti = i - threadIdx.x, tj = j - threadIdx.y;
			if (cuda::fv1_dead(ti, tj, ti + blockDim.x - 1, tj + blockDim.y - 1))
			{
				local = FMIN(local, cuda::solver_params.max_dt);
				continue;
			}
			NUMERIC_TYPE H = U.H[idx];
			NUMERIC_TYPE cell_dt;

			if (H > cuda::solver_params.DepthThresh)
			{
				NUMERIC_TYPE HU = U.HU[idx];
				NUMERIC_TYPE HV = U.HV[idx];
				NUMERIC_TYPE U = HU/H;
				NUMERIC_TYPE V = HV/H;

				NUMERIC_TYPE dt_x = cuda::solver_params.cfl * cuda::geometry.dx
					/ (FABS(U)+SQRT(cuda::physical_params.g * H));
				NUMERIC_TYPE dt_y = cuda::solver_params.cfl * cuda::geometry.dy
					/ (FABS(V)+SQRT(cuda::physical_params.g * H));

				cell_dt = FMIN(dt_x, dt_y);
			}
			else
			{
				cell_dt = cuda::solver_params.max_dt;
			}
			local = FMIN(local, cell_dt);
		}
	}

	block_min_to(local, min_dt);
}

__global__
void update_uniform_rain_func
(
	Flow U,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE  rain_rate
)
{
	int global_i = blockIdx.x * blockDim.x + threadIdx.x;
	int global_j = cuda::fv1_owned_row_begin()
		+ blockIdx.y * blockDim.y + threadIdx.y;
	int cells = 0; // real cells given rain, for rain_volume

	for (int j = global_j; j < cuda::fv1_owned_row_end(); j += blockDim.y * gridDim.y)
	{
		for (int i = global_i; i < cuda::pitch; i += blockDim.x * gridDim.x)
		{
			const int ti = i - threadIdx.x, tj = j - threadIdx.y;
			if (cuda::fv1_dead(ti, tj, ti + blockDim.x - 1, tj + blockDim.y - 1)) continue;
			NUMERIC_TYPE cell_rain;

			cell_rain = rain_rate * cuda::dt;

			NUMERIC_TYPE Z = DEM[j * cuda::pitch + i];
			if (FABS(Z - cuda::solver_params.nodata_elevation) < C(1e-6) /* || Z < C(0.0) */) {

			}
			else {

				NUMERIC_TYPE& Hval = U.H[j * cuda::pitch + i];
				Hval += cell_rain;
				if (i >= 1 && i <= cuda::geometry.xsz) cells++;
			}
		}
	}
	cuda::detail::add_block_sum(cells*(double)rain_rate*cuda::dt*cuda::geometry.dx*cuda::geometry.dy,
		&cuda::rain_volume);
}

__global__ void update_ghost_cells
(
	Flow U,
	NUMERIC_TYPE* Zstar_x,
	NUMERIC_TYPE* Zstar_y
)
{
	for (int j=blockIdx.x*blockDim.x+threadIdx.x+1; j<cuda::geometry.ysz+1;
			j+=blockDim.x*gridDim.x)
	{
		// west
		{
			int i = 1;
			NUMERIC_TYPE Zstar = Zstar_x[j*cuda::pitch + i-1];
			FlowVector U_inside = U[j*cuda::pitch + i];

			FlowVector U_outside = cuda::detail::outside_x(U_inside, U_inside,
					cuda::detail::index_w(i, j), 1, Zstar);

			U.H[j*cuda::pitch + i-1] = U_outside.H;
			U.HU[j*cuda::pitch + i-1] = U_outside.HU;
			U.HV[j*cuda::pitch + i-1] = U_outside.HV;
		}

		// east
		{
			int i = cuda::geometry.xsz;
			NUMERIC_TYPE Zstar = Zstar_x[j*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];

			FlowVector U_outside = cuda::detail::outside_x(U_inside, U_inside,
					cuda::detail::index_e(i, j), -1, Zstar);

			U.H[j*cuda::pitch + i+1] = U_outside.H;
			U.HU[j*cuda::pitch + i+1] = U_outside.HU;
			U.HV[j*cuda::pitch + i+1] = U_outside.HV;
		}
	}

	for (int i=blockIdx.x*blockDim.x+threadIdx.x+1; i<cuda::geometry.xsz+1;
			i+=blockDim.x*gridDim.x)
	{
		// north
		{
			int j = 1;
			NUMERIC_TYPE Zstar = Zstar_y[(j-1)*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];

			FlowVector U_outside = cuda::detail::outside_y(U_inside, U_inside,
					cuda::detail::index_n(i, j), -1, Zstar);

			U.H[(j-1)*cuda::pitch + i] = U_outside.H;
			U.HU[(j-1)*cuda::pitch + i] = U_outside.HU;
			U.HV[(j-1)*cuda::pitch + i] = U_outside.HV;
		}

		// south
		{
			int j = cuda::geometry.ysz;
			NUMERIC_TYPE Zstar = Zstar_y[j*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];

			FlowVector U_outside = cuda::detail::outside_y(U_inside, U_inside,
					cuda::detail::index_s(i, j), 1, Zstar);

			U.H[(j+1)*cuda::pitch + i] = U_outside.H;
			U.HU[(j+1)*cuda::pitch + i] = U_outside.HU;
			U.HV[(j+1)*cuda::pitch + i] = U_outside.HV;
		}
	}
}

// blocks of CUDA_BLOCK_SIZE threads kept resident per SM: register limits that
// give apply_friction and update_flow_variables_y enough warps to hide the
// memory latency (measured on an A100; the x kernel and rain_reduce_dt are
// slower with a limit)
#ifndef FV1_MIN_BLOCKS_FRICTION
#define FV1_MIN_BLOCKS_FRICTION 8
#endif
#ifndef FV1_MIN_BLOCKS_Y
#define FV1_MIN_BLOCKS_Y 4
#endif

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE, FV1_MIN_BLOCKS_FRICTION)
apply_friction
(
	Flow U,
	NUMERIC_TYPE* manning
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int friction_row_begin = cuda::fv1_owned_row_begin();
	int friction_row_end = cuda::fv1_owned_row_end();
#if defined(LISFLOOD_CUDA_MPI)
	// The preceding halo exchange supplies neighbour rows. Friction is a
	// cell-local operation, so applying it to those rows avoids exchanging the
	// same state again immediately before the flux kernels.
	if (cuda::fv1_partition_active) {
		if (friction_row_begin > 1) --friction_row_begin;
		if (friction_row_end < cuda::geometry.ysz + 1) ++friction_row_end;
	}
#endif
	int global_j = friction_row_begin
		+ blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j; j<friction_row_end; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			const int ti = i - threadIdx.x, tj = j - threadIdx.y;
			if (cuda::fv1_dead(ti, tj, ti + blockDim.x - 1, tj + blockDim.y - 1)) continue;
			NUMERIC_TYPE H = U.H[j*cuda::pitch + i];
			NUMERIC_TYPE& HU = U.HU[j*cuda::pitch + i];
			NUMERIC_TYPE& HV = U.HV[j*cuda::pitch + i];

			if (H <= cuda::solver_params.DepthThresh) {
				HU = C(0.0);
				HV = C(0.0);
				continue;
			}

			NUMERIC_TYPE U = HU/H;
			NUMERIC_TYPE V = HV/H;
			if (FABS(U) <= cuda::solver_params.SpeedThresh
					&& FABS(V) <= cuda::solver_params.SpeedThresh)
			{
				HU = C(0.0);
				HV = C(0.0);
				continue;
			}

            NUMERIC_TYPE n = (manning == nullptr)
                ? cuda::physical_params.manning
				: manning[j*cuda::pitch + i];

			NUMERIC_TYPE Cf = cuda::physical_params.g * n * n /
				POW(H, C(1.0)/C(3.0));
			NUMERIC_TYPE speed = SQRT(U*U+V*V);

			NUMERIC_TYPE Sf_x = -Cf*U*speed;
			NUMERIC_TYPE Sf_y = -Cf*V*speed;
			NUMERIC_TYPE D_x = C(1.0) + cuda::dt*Cf / H
				* (C(2.0)*U*U+V*V)/speed;
			NUMERIC_TYPE D_y = C(1.0) + cuda::dt*Cf / H
				* (U*U+C(2.0)*V*V)/speed;

			HU += cuda::dt*Sf_x/D_x;
			HV += cuda::dt*Sf_y/D_y;
		}
	}
}

__device__ NUMERIC_TYPE bed_source_x
(
 	NUMERIC_TYPE Zstar_w,
	NUMERIC_TYPE Zstar_e,
	NUMERIC_TYPE Hstar_w,
	NUMERIC_TYPE Hstar_e,
	NUMERIC_TYPE ETA
)
{
	NUMERIC_TYPE Zdagger_w = Zstar_w - FMAX(C(0.0), -(ETA - Zstar_w));
	NUMERIC_TYPE Zdagger_e = Zstar_e - FMAX(C(0.0), -(ETA - Zstar_e));

	return -cuda::physical_params.g * C(0.5) * (Hstar_w + Hstar_e)
		* (Zdagger_e - Zdagger_w)/cuda::geometry.dx;
}

__device__ NUMERIC_TYPE bed_source_y
(
 	NUMERIC_TYPE Zstar_s,
	NUMERIC_TYPE Zstar_n,
	NUMERIC_TYPE Hstar_s,
	NUMERIC_TYPE Hstar_n,
	NUMERIC_TYPE ETA
)
{
	NUMERIC_TYPE Zdagger_s = Zstar_s - FMAX(C(0.0), -(ETA - Zstar_s));
	NUMERIC_TYPE Zdagger_n = Zstar_n - FMAX(C(0.0), -(ETA - Zstar_n));

	return -cuda::physical_params.g * C(0.5) * (Hstar_s + Hstar_n)
		* (Zdagger_n - Zdagger_s)/cuda::geometry.dy;
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
update_flow_variables_x
(
	Flow Uold,
	Flow U,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* Zstar_x,
	MassStats* mass_stats
)
{
	__shared__ FlowVector F[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
	__shared__ NUMERIC_TYPE Hstar[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
	// DEM of each thread's cell: Zstar_x is the maximum of the two DEM values
	// (initialise_Zstar_x), so it is formed here instead of read
	__shared__ NUMERIC_TYPE Zs[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];

	int global_i = blockIdx.x*(blockDim.x-1) + threadIdx.x;
	const int row_begin = cuda::fv1_owned_row_begin();
	const int row_end = cuda::fv1_owned_row_end();
	int global_j = row_begin - 1 + blockIdx.y*blockDim.y + threadIdx.y;

	int max_i = ((cuda::geometry.xsz+blockDim.x-2)/(blockDim.x-1))*(blockDim.x-1);
	int max_j = row_begin - 1
		+ ((row_end-row_begin+blockDim.y-1)/blockDim.y)*blockDim.y;

	for (int j=global_j+1; j<=max_j; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<=max_i; i+=(blockDim.x-1)*gridDim.x)
		{
			// blocks overlap in the x direction
			// so don't start a new block at the right-hand edge
			if (threadIdx.x == 0 && i == max_i) continue;
			{
				// the same for every thread of the block: skip it whole
				const int ti = i - threadIdx.x, tj = j - threadIdx.y;
				if (cuda::fv1_dead(ti, tj, ti + blockDim.x, tj + blockDim.y - 1)) continue;
			}

			FlowVector U_neg;
			FlowVector Ustar_neg;
			FlowVector F_e;
			NUMERIC_TYPE Z_neg;
			NUMERIC_TYPE Zstar_e;

			// only update flow variables if (i, j) is inside the domain
			if (i <= cuda::geometry.xsz && j < row_end)
			{
				Z_neg = DEM[j*cuda::pitch + i];
				NUMERIC_TYPE Z_pos = DEM[j*cuda::pitch + i+1];
				Zstar_e = FMAX(Z_neg, Z_pos);
				Zs[threadIdx.y][threadIdx.x] = Z_neg;

				U_neg = Uold[j*cuda::pitch + i];
				Ustar_neg = cuda::detail::star(U_neg, Z_neg, Zstar_e);

				FlowVector U_pos = Uold[j*cuda::pitch + i+1];
				FlowVector Ustar_pos = cuda::detail::star(U_pos, Z_pos, Zstar_e);

				if (i == 0)
				{
					Ustar_pos = cuda::detail::inside_x(Ustar_neg, Ustar_pos,
							Ustar_pos, cuda::detail::index_w(i, j));
				}
				else if (i == cuda::geometry.xsz)
				{
					Ustar_neg = cuda::detail::inside_x(Ustar_pos, Ustar_neg,
							Ustar_neg, cuda::detail::index_e(i, j));
				}

				Hstar[threadIdx.y][threadIdx.x] = Ustar_pos.H;

				F_e = cuda::detail::hll_x(Ustar_neg, Ustar_pos);
				F[threadIdx.y][threadIdx.x] = F_e;
				cuda::detail::update_mass_stats_x(mass_stats, F_e.H, i, j);

			}

			__syncthreads();

			if (threadIdx.x != 0 && i <= cuda::geometry.xsz && j < row_end)
			{
				FlowVector& F_w = F[threadIdx.y][threadIdx.x-1];
				FlowVector U0 = Uold[j*cuda::pitch + i];

				NUMERIC_TYPE& H = U.H[j*cuda::pitch + i];
				NUMERIC_TYPE& HU = U.HU[j*cuda::pitch + i];
				NUMERIC_TYPE& HV = U.HV[j*cuda::pitch + i];

				NUMERIC_TYPE Zstar_w = FMAX(Zs[threadIdx.y][threadIdx.x-1], Z_neg);
				NUMERIC_TYPE Hstar_w = Hstar[threadIdx.y][threadIdx.x-1];
				NUMERIC_TYPE ETA = U_neg.H + Z_neg;

				H = U0.H - cuda::dt * (F_e.H - F_w.H)/cuda::geometry.dx;
				HU = U0.HU - cuda::dt * ((F_e.HU - F_w.HU)/cuda::geometry.dx
					- bed_source_x(Zstar_w, Zstar_e, Hstar_w, Ustar_neg.H, ETA));
				HV = U0.HV - cuda::dt * (F_e.HV - F_w.HV)/cuda::geometry.dx;

			}
			// shared memory is rewritten in the next pass: wait for every reader
			__syncthreads();
		}
	}
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE, FV1_MIN_BLOCKS_Y)
update_flow_variables_y
(
	Flow Uold,
	Flow Uint,
	Flow U,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* Zstar_y,
	MassStats* mass_stats,
	NUMERIC_TYPE* maxH,
	bool drain,
	bool losses,
	LossParams p,
	double* volume
)
{
	double removed = 0.0;   // depth removed by the losses, as fv1_cell_updates
        __shared__ FlowVector F[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
        __shared__ NUMERIC_TYPE Hstar[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
	// DEM of each thread's cell, for Zstar_y as in update_flow_variables_x
	__shared__ NUMERIC_TYPE Zs[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];

        int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	const int row_begin = cuda::fv1_owned_row_begin();
	const int row_end = cuda::fv1_owned_row_end();
	const int interface_begin = row_begin - 1;
	int global_j = interface_begin
		+ blockIdx.y*(blockDim.y-1) + threadIdx.y;

	int max_i = ((cuda::geometry.xsz+blockDim.x-1)/blockDim.x)*blockDim.x;
	int max_j = interface_begin
		+ ((row_end-row_begin+blockDim.y-2)/(blockDim.y-1))*(blockDim.y-1);

	for (int j=global_j; j<=max_j; j+=(blockDim.y-1)*gridDim.y)
	{
		if (threadIdx.y == 0 && j == max_j) continue;

		for (int i=global_i+1; i<=max_i; i+=blockDim.x*gridDim.x)
		{
			{
				// the same for every thread of the block: skip it whole
				const int ti = i - threadIdx.x, tj = j - threadIdx.y;
				if (cuda::fv1_dead(ti, tj, ti + blockDim.x - 1, tj + blockDim.y)) continue;
			}
			FlowVector U_pos;
			FlowVector Ustar_pos;
			FlowVector F_s;
			NUMERIC_TYPE Z_pos;
			NUMERIC_TYPE Zstar_s;

			if (i <= cuda::geometry.xsz && j <= cuda::geometry.ysz &&
					(cuda::fv1_owns_row(j) || cuda::fv1_owns_row(j+1)))
			{
				NUMERIC_TYPE Z_neg = DEM[(j+1)*cuda::pitch + i];
				Z_pos = DEM[j*cuda::pitch + i];
				Zstar_s = FMAX(Z_neg, Z_pos);
				Zs[threadIdx.y][threadIdx.x] = Z_pos;

				FlowVector U_neg = Uold[(j+1)*cuda::pitch + i];
				FlowVector Ustar_neg = cuda::detail::star(U_neg, Z_neg, Zstar_s);

				U_pos = Uold[j*cuda::pitch + i];
				Ustar_pos = cuda::detail::star(U_pos, Z_pos, Zstar_s);

				// this hack appears necessary to prevent some unknown
				// compiler optimisation from causing a bug seen in
				// EA test 4
				//Ustar_pos.H = Ustar_pos.H*C(2.0)/C(2.0);

				if (j == 0)
				{       
					Ustar_neg = cuda::detail::inside_y(Ustar_pos, Ustar_neg,
							Ustar_neg, cuda::detail::index_n(i, j));
				}
				else if (j == cuda::geometry.ysz)
				{
					Ustar_pos = cuda::detail::inside_y(Ustar_neg, Ustar_pos,
							Ustar_pos, cuda::detail::index_s(i, j));
				}

				Hstar[threadIdx.y][threadIdx.x] = Ustar_neg.H;

				F_s = cuda::detail::hll_y(Ustar_neg, Ustar_pos);
				F[threadIdx.y][threadIdx.x] = F_s;
				if ((j == 0 && cuda::fv1_owns_row(1)) ||
						(j > 0 && cuda::fv1_owns_row(j)))
					cuda::detail::update_mass_stats_y(mass_stats, F_s.H, i, j);
			}

                        __syncthreads();

			if (threadIdx.y != 0 && i <= cuda::geometry.xsz && j <= cuda::geometry.ysz && cuda::fv1_owns_row(j))
			{
				FlowVector& F_n = F[threadIdx.y-1][threadIdx.x];
				FlowVector U0 = Uint[j*cuda::pitch + i];

				NUMERIC_TYPE Zstar_n = FMAX(Z_pos, Zs[threadIdx.y-1][threadIdx.x]);
				NUMERIC_TYPE Hstar_n = Hstar[threadIdx.y-1][threadIdx.x];
				NUMERIC_TYPE ETA = U_pos.H + Z_pos;

				NUMERIC_TYPE H = U0.H - cuda::dt * (F_n.H - F_s.H)/cuda::geometry.dy;
				NUMERIC_TYPE HU = U0.HU - cuda::dt * (F_n.HU - F_s.HU)/cuda::geometry.dy;
				NUMERIC_TYPE HV = U0.HV - cuda::dt * ((F_n.HV - F_s.HV)/cuda::geometry.dy
					- bed_source_y(Zstar_s, Zstar_n, Ustar_pos.H, Hstar_n, ETA));
				const int idx = j*cuda::pitch + i;
				if (maxH != nullptr)
				{
					// Losses::apply_fv1_cells: drainage or losses, then the maximum
					const bool nodata = FABS(Z_pos - cuda::solver_params.nodata_elevation) < C(1e-6);
					cuda::detail::fv1_cell_values(H, HU, HV, idx, nodata, drain, losses, mass_stats, p, removed);
					maxH[idx] = FMAX(maxH[idx], H);
				}
				U.H[idx] = H;
				U.HU[idx] = HU;
				U.HV[idx] = HV;
			}
			// shared memory is rewritten in the next pass: wait for every reader
			__syncthreads();
        }
    }

	if (maxH != nullptr && losses) cuda::detail::add_block_sum(removed*cuda::geometry.dx*cuda::geometry.dy, volume);
}

}
}
}

namespace lis
{
namespace cuda
{
namespace fv1
{

#ifndef FV1_MIN_BLOCKS_YM
#define FV1_MIN_BLOCKS_YM 4
#endif

/**
 * update_flow_variables_y with one thread per column, marching down a strip
 * of strip rows: each y face is computed once, and the face south of a
 * cell is the face north of the next, kept in registers. The arithmetic is
 * that of update_flow_variables_y. Launched on march_grid(CUDA_BLOCK_SIZE).
 */
__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE, FV1_MIN_BLOCKS_YM)
update_flow_variables_ym
(
	Flow Uold,
	Flow Uint,
	Flow U,
	NUMERIC_TYPE* DEM,
	MassStats* mass_stats,
	NUMERIC_TYPE* maxH,
	bool drain,
	bool losses,
	LossParams p,
	double* volume,
	int strip
)
{
	double removed = 0.0;   // depth removed by the losses, as fv1_cell_updates
	const int xsz = cuda::geometry.xsz, ysz = cuda::geometry.ysz;
	const int row_begin = cuda::fv1_owned_row_begin();
	const int row_end = cuda::fv1_owned_row_end();
	const int strips = (row_end - row_begin + strip - 1)/strip;

	for (int s = blockIdx.y; s < strips; s += gridDim.y)
	{
		const int jb = row_begin + s*strip;
		const int je = min(jb + strip, row_end);   // rows [jb, je)

		for (int i = blockIdx.x*blockDim.x + threadIdx.x + 1; i <= xsz;
				i += blockDim.x*gridDim.x)
		{
			if (cuda::fv1_dead(i, jb, i, je - 1)) continue;

			// the face south of row j: between rows j (pos) and j+1 (neg)
			NUMERIC_TYPE Z_pos, Z_neg, Zstar_s;
			FlowVector U_pos, U_neg, Ustar_pos, Ustar_neg, F_s;
			auto face = [&](int j)
			{
				Zstar_s = FMAX(Z_neg, Z_pos);
				Ustar_neg = cuda::detail::star(U_neg, Z_neg, Zstar_s);
				Ustar_pos = cuda::detail::star(U_pos, Z_pos, Zstar_s);

				if (j == 0)
				{
					Ustar_neg = cuda::detail::inside_y(Ustar_pos, Ustar_neg,
							Ustar_neg, cuda::detail::index_n(i, j));
				}
				else if (j == ysz)
				{
					Ustar_pos = cuda::detail::inside_y(Ustar_neg, Ustar_pos,
							Ustar_pos, cuda::detail::index_s(i, j));
				}

				F_s = cuda::detail::hll_y(Ustar_neg, Ustar_pos);
			};
			auto mass = [&](int j)
			{
				if ((j == 0 && cuda::fv1_owns_row(1)) ||
						(j > 0 && cuda::fv1_owns_row(j)))
					cuda::detail::update_mass_stats_y(mass_stats, F_s.H, i, j);
			};

			// the face north of row jb (the last face of the strip above)
			int j = jb - 1;
			Z_pos = DEM[j*cuda::pitch + i];
			U_pos = Uold[j*cuda::pitch + i];
			Z_neg = DEM[(j+1)*cuda::pitch + i];
			U_neg = Uold[(j+1)*cuda::pitch + i];
			face(j);
			if (jb == row_begin) mass(j);

			// the loads of the next row, issued a row ahead
			NUMERIC_TYPE Z_next = DEM[(jb+1)*cuda::pitch + i];
			FlowVector U_next = Uold[(jb+1)*cuda::pitch + i];
			FlowVector U0_next = Uint[jb*cuda::pitch + i];
			NUMERIC_TYPE maxH_next = maxH != nullptr ? maxH[jb*cuda::pitch + i] : C(0.0);

			for (j = jb; j < je; j++)
			{
				// the face north of row j
				const FlowVector F_n = F_s;
				const NUMERIC_TYPE Hstar_n = Ustar_neg.H;
				const NUMERIC_TYPE Zstar_n = Zstar_s;

				Z_pos = Z_neg;
				U_pos = U_neg;
				Z_neg = Z_next;
				U_neg = U_next;
				const FlowVector U0 = U0_next;
				const NUMERIC_TYPE maxH0 = maxH_next;
				if (j + 1 < je)
				{
					Z_next = DEM[(j+2)*cuda::pitch + i];
					U_next = Uold[(j+2)*cuda::pitch + i];
					U0_next = Uint[(j+1)*cuda::pitch + i];
					if (maxH != nullptr) maxH_next = maxH[(j+1)*cuda::pitch + i];
				}
				face(j);
				mass(j);

				const int idx = j*cuda::pitch + i;
				NUMERIC_TYPE ETA = U_pos.H + Z_pos;

				NUMERIC_TYPE H = U0.H - cuda::dt * (F_n.H - F_s.H)/cuda::geometry.dy;
				NUMERIC_TYPE HU = U0.HU - cuda::dt * (F_n.HU - F_s.HU)/cuda::geometry.dy;
				NUMERIC_TYPE HV = U0.HV - cuda::dt * ((F_n.HV - F_s.HV)/cuda::geometry.dy
					- bed_source_y(Zstar_s, Zstar_n, Ustar_pos.H, Hstar_n, ETA));
				if (maxH != nullptr)
				{
					// Losses::apply_fv1_cells: drainage or losses, then the maximum
					const bool nodata = FABS(Z_pos - cuda::solver_params.nodata_elevation) < C(1e-6);
					cuda::detail::fv1_cell_values(H, HU, HV, idx, nodata, drain, losses, mass_stats, p, removed);
					maxH[idx] = FMAX(maxH0, H);
				}
				U.H[idx] = H;
				U.HU[idx] = HU;
				U.HV[idx] = HV;
			}
		}
	}

	if (maxH != nullptr && losses) cuda::detail::add_block_sum(removed*cuda::geometry.dx*cuda::geometry.dy, volume);
}

#ifndef FV1_MIN_BLOCKS_XM
#define FV1_MIN_BLOCKS_XM 5
#endif

/**
 * update_flow_variables_x with each warp on a segment of 32 cells of a row,
 * marching down a strip of strip rows: the cell east of a lane and the
 * flux west of it come from the neighbouring lanes. Segments overlap by one
 * cell; lane 31 reads the cell east of the segment. The arithmetic is that of
 * update_flow_variables_x. Launched on march_grid_x(CUDA_BLOCK_SIZE).
 */
__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE, FV1_MIN_BLOCKS_XM)
update_flow_variables_xm
(
	Flow Uold,
	Flow U,
	NUMERIC_TYPE* DEM,
	MassStats* mass_stats,
	int strip
)
{
	const unsigned all = 0xffffffff;
	const int lane = threadIdx.x & 31;
	const int warps = blockDim.x/32;
	const int xsz = cuda::geometry.xsz;
	const int row_begin = cuda::fv1_owned_row_begin();
	const int row_end = cuda::fv1_owned_row_end();
	// segment w: cells 31w .. 31w+31, the faces east of them, the updates of
	// cells 31w+1 .. 31w+31; the last segment holds cell xsz below lane 31
	const int segments = xsz/31 + 1;
	const int strips = (row_end - row_begin + strip - 1)/strip;

	for (int s = blockIdx.y; s < strips; s += gridDim.y)
	{
		const int jb = row_begin + s*strip;
		const int je = min(jb + strip, row_end);   // rows [jb, je)

		for (int w = blockIdx.x*warps + threadIdx.x/32; w < segments; w += warps*gridDim.x)
		{
			// the same for the whole warp
			if (cuda::fv1_dead(31*w, jb, 31*w + 32, je - 1)) continue;

			const int c = 31*w + lane;
			const bool in = c <= xsz + 1;
			const bool east = lane == 31 && c + 1 <= xsz + 1;

			NUMERIC_TYPE Z_next = C(0.0), Ze_next = C(0.0);
			FlowVector U_next = { C(0.0), C(0.0), C(0.0) }, Ue_next = U_next;
			auto load = [&](int j)
			{
				if (in)
				{
					Z_next = DEM[j*cuda::pitch + c];
					U_next = Uold[j*cuda::pitch + c];
				}
				if (east)
				{
					Ze_next = DEM[j*cuda::pitch + c+1];
					Ue_next = Uold[j*cuda::pitch + c+1];
				}
			};
			load(jb);

			for (int j = jb; j < je; j++)
			{
				const NUMERIC_TYPE Z_neg = Z_next, Ze = Ze_next;
				const FlowVector U_neg = U_next, Ue = Ue_next;
				if (j + 1 < je) load(j + 1);

				// the cell east of this one
				NUMERIC_TYPE Z_pos = __shfl_down_sync(all, Z_neg, 1);
				FlowVector U_pos;
				U_pos.H = __shfl_down_sync(all, U_neg.H, 1);
				U_pos.HU = __shfl_down_sync(all, U_neg.HU, 1);
				U_pos.HV = __shfl_down_sync(all, U_neg.HV, 1);
				if (lane == 31)
				{
					Z_pos = Ze;
					U_pos = Ue;
				}

				// the face east of the cell
				const bool face = c <= xsz;
				NUMERIC_TYPE Zstar_e = C(0.0);
				FlowVector Ustar_neg = { C(0.0), C(0.0), C(0.0) };
				FlowVector Ustar_pos = Ustar_neg, F_e = Ustar_neg;
				if (face)
				{
					Zstar_e = FMAX(Z_neg, Z_pos);
					Ustar_neg = cuda::detail::star(U_neg, Z_neg, Zstar_e);
					Ustar_pos = cuda::detail::star(U_pos, Z_pos, Zstar_e);

					if (c == 0)
					{
						Ustar_pos = cuda::detail::inside_x(Ustar_neg, Ustar_pos,
								Ustar_pos, cuda::detail::index_w(c, j));
					}
					else if (c == xsz)
					{
						Ustar_neg = cuda::detail::inside_x(Ustar_pos, Ustar_neg,
								Ustar_neg, cuda::detail::index_e(c, j));
					}

					F_e = cuda::detail::hll_x(Ustar_neg, Ustar_pos);
					// lane 31's face is lane 0's of the next segment
					if (lane != 31) cuda::detail::update_mass_stats_x(mass_stats, F_e.H, c, j);
				}

				// the face west of the cell
				FlowVector F_w;
				F_w.H = __shfl_up_sync(all, F_e.H, 1);
				F_w.HU = __shfl_up_sync(all, F_e.HU, 1);
				F_w.HV = __shfl_up_sync(all, F_e.HV, 1);
				const NUMERIC_TYPE Hstar_w = __shfl_up_sync(all, Ustar_pos.H, 1);
				const NUMERIC_TYPE Z_w = __shfl_up_sync(all, Z_neg, 1);

				if (lane != 0 && face && c >= 1)
				{
					const FlowVector U0 = U_neg;
					NUMERIC_TYPE Zstar_w = FMAX(Z_w, Z_neg);
					NUMERIC_TYPE ETA = U_neg.H + Z_neg;

					const int idx = j*cuda::pitch + c;
					U.H[idx] = U0.H - cuda::dt * (F_e.H - F_w.H)/cuda::geometry.dx;
					U.HU[idx] = U0.HU - cuda::dt * ((F_e.HU - F_w.HU)/cuda::geometry.dx
						- bed_source_x(Zstar_w, Zstar_e, Hstar_w, Ustar_neg.H, ETA));
					U.HV[idx] = U0.HV - cuda::dt * (F_e.HV - F_w.HV)/cuda::geometry.dx;
				}
			}
		}
	}
}

/** apply_friction for one cell: the same arithmetic in the same order */
__device__ __forceinline__ void friction_cell
(
	NUMERIC_TYPE H,
	NUMERIC_TYPE& HU,
	NUMERIC_TYPE& HV,
	NUMERIC_TYPE n
)
{
	if (H <= cuda::solver_params.DepthThresh) {
		HU = C(0.0);
		HV = C(0.0);
		return;
	}

	NUMERIC_TYPE U = HU/H;
	NUMERIC_TYPE V = HV/H;
	if (FABS(U) <= cuda::solver_params.SpeedThresh
			&& FABS(V) <= cuda::solver_params.SpeedThresh)
	{
		HU = C(0.0);
		HV = C(0.0);
		return;
	}

	NUMERIC_TYPE Cf = cuda::physical_params.g * n * n /
		POW(H, C(1.0)/C(3.0));
	NUMERIC_TYPE speed = SQRT(U*U+V*V);

	NUMERIC_TYPE Sf_x = -Cf*U*speed;
	NUMERIC_TYPE Sf_y = -Cf*V*speed;
	NUMERIC_TYPE D_x = C(1.0) + cuda::dt*Cf / H
		* (C(2.0)*U*U+V*V)/speed;
	NUMERIC_TYPE D_y = C(1.0) + cuda::dt*Cf / H
		* (U*U+C(2.0)*V*V)/speed;

	HU += cuda::dt*Sf_x/D_x;
	HV += cuda::dt*Sf_y/D_y;
}

#ifndef FV1_BATCH
#define FV1_BATCH 2
#endif

/**
 * apply_friction with the loads of FV1_BATCH of each thread's cells (rows
 * blockDim.y*gridDim.y apart) issued together, so that their latencies
 * overlap; the arithmetic of apply_friction (friction_cell) for each cell
 */
__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE, FV1_MIN_BLOCKS_FRICTION)
apply_friction_batched
(
	Flow U,
	NUMERIC_TYPE* manning
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int friction_row_begin = cuda::fv1_owned_row_begin();
	int friction_row_end = cuda::fv1_owned_row_end();
#if defined(LISFLOOD_CUDA_MPI)
	if (cuda::fv1_partition_active) {
		if (friction_row_begin > 1) --friction_row_begin;
		if (friction_row_end < cuda::geometry.ysz + 1) ++friction_row_end;
	}
#endif
	int global_j = friction_row_begin
		+ blockIdx.y*blockDim.y + threadIdx.y;
	const int stride_j = blockDim.y*gridDim.y;

	for (int j0=global_j; j0<friction_row_end; j0+=FV1_BATCH*stride_j)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			const int ti = i - threadIdx.x;
			bool live[FV1_BATCH];
			NUMERIC_TYPE H[FV1_BATCH], HU[FV1_BATCH], HV[FV1_BATCH], n[FV1_BATCH];
#pragma unroll
			for (int k = 0; k < FV1_BATCH; k++)
			{
				const int j = j0 + k*stride_j, tj = j - threadIdx.y;
				live[k] = j < friction_row_end
					&& !cuda::fv1_dead(ti, tj, ti + blockDim.x - 1, tj + blockDim.y - 1);
				H[k] = live[k] ? U.H[j*cuda::pitch + i] : C(0.0);
			}
#pragma unroll
			for (int k = 0; k < FV1_BATCH; k++)
			{
				const int idx = (j0 + k*stride_j)*cuda::pitch + i;
				HU[k] = HV[k] = n[k] = C(0.0);
				if (live[k] && H[k] > cuda::solver_params.DepthThresh)
				{
					HU[k] = U.HU[idx];
					HV[k] = U.HV[idx];
					n[k] = (manning == nullptr) ? cuda::physical_params.manning : manning[idx];
				}
			}
#pragma unroll
			for (int k = 0; k < FV1_BATCH; k++)
			{
				if (!live[k]) continue;
				const int idx = (j0 + k*stride_j)*cuda::pitch + i;
				friction_cell(H[k], HU[k], HV[k], n[k]);
				U.HU[idx] = HU[k];
				U.HV[idx] = HV[k];
			}
		}
	}
}

/** the time step of one cell, as reduce_dt */
__device__ __forceinline__ NUMERIC_TYPE cell_dt
(
	NUMERIC_TYPE H,
	NUMERIC_TYPE HU,
	NUMERIC_TYPE HV
)
{
	if (H > cuda::solver_params.DepthThresh)
	{
		NUMERIC_TYPE U = HU/H;
		NUMERIC_TYPE V = HV/H;

		NUMERIC_TYPE dt_x = cuda::solver_params.cfl * cuda::geometry.dx
			/ (FABS(U)+SQRT(cuda::physical_params.g * H));
		NUMERIC_TYPE dt_y = cuda::solver_params.cfl * cuda::geometry.dy
			/ (FABS(V)+SQRT(cuda::physical_params.g * H));

		return FMIN(dt_x, dt_y);
	}
	return cuda::solver_params.max_dt;
}

/**
 * DynamicRain::update and reduce_dt over the inner cells this rank owns, in
 * one pass: each cell first gets its rain (with the previous step's dt), then
 * gives its time step from the new depth. The ghost cells follow in
 * update_ghost_cells_dt.
 */
__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
rain_reduce_dt
(
	NUMERIC_TYPE* min_dt,
	Flow U,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* rain_data,
	int first,
	int last
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = cuda::fv1_owned_row_begin()
		+ blockIdx.y*blockDim.y + threadIdx.y;
	NUMERIC_TYPE local = FLT_MAX;
	double rate_sum = 0.0; // m/s summed over the cells, for rain_volume

	for (int j=global_j; j<cuda::fv1_owned_row_end(); j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			const int idx = j*cuda::pitch + i;
			const bool in_range = idx >= first && idx < last;
			const int ti = i - threadIdx.x, tj = j - threadIdx.y;
			if (cuda::fv1_dead(ti, tj, ti + blockDim.x - 1, tj + blockDim.y - 1))
			{
				if (in_range) local = FMIN(local, cuda::solver_params.max_dt);
				continue;
			}
			NUMERIC_TYPE H = U.H[idx];
			if (rain_data != nullptr)
			{
				NUMERIC_TYPE Z = DEM[idx];
				if (!(FABS(Z - cuda::solver_params.nodata_elevation) < C(1e-6)))
				{
					NUMERIC_TYPE& Hval = U.H[idx];
					const NUMERIC_TYPE rate = cuda::detail::rate_at_cell(rain_data, i, j);
					Hval += rate * cuda::dt;
					rate_sum += rate;
					H = Hval;
				}
			}
			if (in_range) local = FMIN(local, cell_dt(H, U.HU[idx], U.HV[idx]));
		}
	}

	block_min_to(local, min_dt);
	if (rain_data != nullptr)
		cuda::detail::add_block_sum(rate_sum*cuda::dt*cuda::geometry.dx*cuda::geometry.dy, &rain_volume);
}

/**
 * update_ghost_cells, then the time steps of the ghost cells in elements
 * [first, last) (the corners, which update_ghost_cells never writes, as they
 * are), into min_dt as reduce_dt
 */
__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
update_ghost_cells_dt
(
	NUMERIC_TYPE* min_dt,
	Flow U,
	NUMERIC_TYPE* Zstar_x,
	NUMERIC_TYPE* Zstar_y,
	int first,
	int last
)
{
	NUMERIC_TYPE local = FLT_MAX;
	auto take = [&](int idx, const FlowVector& V)
	{
		U.H[idx] = V.H;
		U.HU[idx] = V.HU;
		U.HV[idx] = V.HV;
		if (idx >= first && idx < last) local = FMIN(local, cell_dt(V.H, V.HU, V.HV));
	};

	for (int j=blockIdx.x*blockDim.x+threadIdx.x+1; j<cuda::geometry.ysz+1;
			j+=blockDim.x*gridDim.x)
	{
		// west
		{
			int i = 1;
			NUMERIC_TYPE Zstar = Zstar_x[j*cuda::pitch + i-1];
			FlowVector U_inside = U[j*cuda::pitch + i];
			take(j*cuda::pitch + i-1, cuda::detail::outside_x(U_inside, U_inside,
					cuda::detail::index_w(i, j), 1, Zstar));
		}

		// east
		{
			int i = cuda::geometry.xsz;
			NUMERIC_TYPE Zstar = Zstar_x[j*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];
			take(j*cuda::pitch + i+1, cuda::detail::outside_x(U_inside, U_inside,
					cuda::detail::index_e(i, j), -1, Zstar));
		}
	}

	for (int i=blockIdx.x*blockDim.x+threadIdx.x+1; i<cuda::geometry.xsz+1;
			i+=blockDim.x*gridDim.x)
	{
		// north
		{
			int j = 1;
			NUMERIC_TYPE Zstar = Zstar_y[(j-1)*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];
			take((j-1)*cuda::pitch + i, cuda::detail::outside_y(U_inside, U_inside,
					cuda::detail::index_n(i, j), -1, Zstar));
		}

		// south
		{
			int j = cuda::geometry.ysz;
			NUMERIC_TYPE Zstar = Zstar_y[j*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];
			take((j+1)*cuda::pitch + i, cuda::detail::outside_y(U_inside, U_inside,
					cuda::detail::index_s(i, j), 1, Zstar));
		}
	}

	if (blockIdx.x == 0 && threadIdx.x < 4)
	{
		const int i = (threadIdx.x & 1) ? cuda::geometry.xsz + 1 : 0;
		const int j = (threadIdx.x & 2) ? cuda::geometry.ysz + 1 : 0;
		const int idx = j*cuda::pitch + i;
		if (idx >= first && idx < last)
			local = FMIN(local, cell_dt(U.H[idx], U.HU[idx], U.HV[idx]));
	}

	block_min_to(local, min_dt);
}

}
}
}

lis::cuda::fv1::Solver::Solver
(
	Flow& U,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* Zstar_x,
	NUMERIC_TYPE* Zstar_y,
	NUMERIC_TYPE* manning,
	Geometry& geometry,
	PhysicalParams& physical_params,
	dim3 grid_size
)
:
Uold(U1),
U(U2),
DEM(DEM),
Zstar_x(Zstar_x),
Zstar_y(Zstar_y),
manning(manning),
	grid_size(grid_size),
	raster_pitch(geometry.xsz + 2)
{
	Flow::allocate_device(U1, geometry);
	Flow::allocate_device(U2, geometry);
	Flow::allocate_device(Ux, geometry);
	const size_t bytes = lis::GhostRaster::elements(geometry) * sizeof(NUMERIC_TYPE);
	for (Flow* f : {&U2, &Ux})
	{
		checkCudaErrors(cudaMemset(f->H, 0, bytes));
		checkCudaErrors(cudaMemset(f->HU, 0, bytes));
		checkCudaErrors(cudaMemset(f->HV, 0, bytes));
	}

	Flow::copy(U1, U, geometry);

	friction = (physical_params.manning > C(0.0)) || (manning != nullptr);
	// LISFLOOD_FV1_UNFUSED=1: the kernels of the 8.4 release (separate rain,
	// time step, ghost cells and cell-wise updates)
	fused = std::getenv("LISFLOOD_FV1_UNFUSED") == nullptr;
	// LISFLOOD_FV1_TILES=1: the x and y fluxes in 16 x 16 tiles instead of
	// strips of rows
	march = fused && std::getenv("LISFLOOD_FV1_TILES") == nullptr;
	// strips of up to 16 rows, fewer on grids too small to fill the GPU with
	// a thread per column and strip (LISFLOOD_FV1_STRIP=n: n rows)
	{
		int device = 0, sms = 1, threads = 2048;
		checkCudaErrors(cudaGetDevice(&device));
		checkCudaErrors(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
		checkCudaErrors(cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, device));
		const double cells = static_cast<double>(geometry.xsz) * geometry.ysz;
		strip = static_cast<int>(cells / (2.0 * sms * threads));
		if (const char* e = std::getenv("LISFLOOD_FV1_STRIP")) strip = std::atoi(e);
		strip = std::min(16, std::max(1, strip));
	}
	march_grid = dim3(std::max(1, (geometry.xsz + CUDA_BLOCK_SIZE - 1)/CUDA_BLOCK_SIZE),
			std::min(65535, std::max(1, (geometry.ysz + strip - 1)/strip)));
	march_grid_x = dim3(std::max(1, (geometry.xsz/31 + 1 + CUDA_BLOCK_SIZE/32 - 1)/(CUDA_BLOCK_SIZE/32)),
			march_grid.y);
}

void lis::cuda::fv1::Solver::zero_ghost_cells()
{
	zero_ghost_cells_north_south<<<1, CUDA_BLOCK_SIZE>>>(U);
	zero_ghost_cells_east_west<<<1, CUDA_BLOCK_SIZE>>>(U);
}

void lis::cuda::fv1::Solver::update_ghost_cells()
{
	lis::cuda::fv1::update_ghost_cells<<<64, CUDA_BLOCK_SIZE>>>(
			Uold, Zstar_x, Zstar_y);
}

lis::cuda::fv1::Flow& lis::cuda::fv1::Solver::update_flow_variables
(
	MassStats* mass_stats
)
{
	if (friction && fused)
	{
		apply_friction_batched<<<grid_size, cuda::block_size>>>(Uold, manning);
		update_ghost_cells();
	}
	else if (friction)
	{
		apply_friction<<<grid_size, cuda::block_size>>>(Uold, manning);
		update_ghost_cells();
	}

	if (fused && march)
		update_flow_variables_xm<<<march_grid_x, CUDA_BLOCK_SIZE>>>
				(Uold, Ux, DEM, mass_stats, strip);
	else
		update_flow_variables_x<<<grid_size, cuda::block_size>>>
				(Uold, Ux, DEM, Zstar_x, mass_stats);

	if (fused && march)
		update_flow_variables_ym<<<march_grid, CUDA_BLOCK_SIZE>>>
				(Uold, Ux, U, DEM, mass_stats, cells.maxH,
				 cells.drain, cells.losses, cells.p, cells.volume, strip);
	else if (fused)
		update_flow_variables_y<<<grid_size, cuda::block_size>>>
				(Uold, Ux, U, DEM, Zstar_y, mass_stats, cells.maxH,
				 cells.drain, cells.losses, cells.p, cells.volume);
	else
		update_flow_variables_y<<<grid_size, cuda::block_size>>>
				(Uold, Ux, U, DEM, Zstar_y, mass_stats, nullptr,
				 false, false, cells.p, nullptr);

	std::swap(Uold, U);

	return Uold;
}

lis::cuda::fv1::Flow& lis::cuda::fv1::Solver::d_U()
{
	return Uold;
}

void lis::cuda::fv1::Solver::update_dt_per_element
(
	NUMERIC_TYPE* dt_field
) const
{
	lis::cuda::fv1::update_dt_per_element<<<grid_size, cuda::block_size>>>(
			dt_field, Uold);
}



bool lis::cuda::fv1::Solver::reduce_dt
(
	NUMERIC_TYPE* min_dt,
	int first,
	int count
) const
{
	// 0x7f bytes: a float or double far above any time step
	checkCudaErrors(cudaMemsetAsync(min_dt, 0x7f, sizeof(NUMERIC_TYPE)));
	if (fused_dt_pending)
	{
		// rain, ghost cells and time step (see prepare_fused_dt)
		lis::cuda::fv1::rain_reduce_dt<<<fused_cell_grid, CUDA_BLOCK_SIZE>>>(
				min_dt, Uold, DEM, fused_rain, first, first + count);
		lis::cuda::fv1::update_ghost_cells_dt<<<64, CUDA_BLOCK_SIZE>>>(
				min_dt, Uold, Zstar_x, Zstar_y, first, first + count);
		fused_dt_pending = false;
		return true;
	}
	lis::cuda::fv1::reduce_dt<<<grid_size, cuda::block_size>>>(
			min_dt, Uold, first, first + count);
	return true;
}

void lis::cuda::fv1::Solver::FloodplainQ()
{

  return;	
}

void lis::cuda::fv1::Solver::zero_thin_depth_slopes()
{

  return;	
}

void lis::cuda::fv1::Solver::updateMaxFieldACC
(
	NUMERIC_TYPE t
)
{
	return;
}

void lis::cuda::fv1::Solver::update_uniform_rain
(
	NUMERIC_TYPE rain_rate
)
{
	update_uniform_rain_func << <grid_size, cuda::block_size >> > (
		Uold, DEM, rain_rate);
}


lis::cuda::fv1::Solver::~Solver()
{
	Flow::free_device(U1);
	Flow::free_device(U2);
	Flow::free_device(Ux);
}
