#include "cuda_fv1_solver.cuh"
#include "cuda_boundary.cuh"
#include "cuda_hll.cuh"
#include "cuda_solver.cuh"
#include <algorithm>
#include <cfloat>
#include <helper_cuda.h>
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

/** *a = min(*a, v) for v >= 0: the bit patterns of non-negative floats order as the values */
__device__ inline void atomic_min_nonnegative(float* a, float v)
{
	atomicMin(reinterpret_cast<unsigned int*>(a), __float_as_uint(v));
}

__device__ inline void atomic_min_nonnegative(double* a, double v)
{
	atomicMin(reinterpret_cast<unsigned long long*>(a),
			static_cast<unsigned long long>(__double_as_longlong(v)));
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

	// minimum over the block: within each warp, then over the warps
	__shared__ NUMERIC_TYPE warp_min[CUDA_BLOCK_SIZE/32];
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	for (int s=16; s>0; s>>=1) local = FMIN(local, __shfl_down_sync(0xffffffff, local, s));
	if ((tid & 31) == 0) warp_min[tid/32] = local;
	__syncthreads();
	if (tid < 32)
	{
		local = (tid < (blockDim.x*blockDim.y)/32) ? warp_min[tid] : FLT_MAX;
		for (int s=16; s>0; s>>=1) local = FMIN(local, __shfl_down_sync(0xffffffff, local, s));
		if (tid == 0) atomic_min_nonnegative(min_dt, local);
	}
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

	for (int j = global_j; j < cuda::fv1_owned_row_end(); j += blockDim.y * gridDim.y)
	{
		for (int i = global_i; i < cuda::pitch; i += blockDim.x * gridDim.x)
		{
			NUMERIC_TYPE cell_rain;

			cell_rain = rain_rate * cuda::dt;

			NUMERIC_TYPE Z = DEM[j * cuda::pitch + i];
			if (FABS(Z - cuda::solver_params.nodata_elevation) < C(1e-6) /* || Z < C(0.0) */) {

			}
			else {

				NUMERIC_TYPE& Hval = U.H[j * cuda::pitch + i];
				Hval += cell_rain;
			}
		}
	}
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

			FlowVector U_outside = Boundary::outside_x(U_inside, U_inside,
					Boundary::index_w(i, j), 1, Zstar);

			U.H[j*cuda::pitch + i-1] = U_outside.H;
			U.HU[j*cuda::pitch + i-1] = U_outside.HU;
			U.HV[j*cuda::pitch + i-1] = U_outside.HV;
		}

		// east
		{
			int i = cuda::geometry.xsz;
			NUMERIC_TYPE Zstar = Zstar_x[j*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];

			FlowVector U_outside = Boundary::outside_x(U_inside, U_inside,
					Boundary::index_e(i, j), -1, Zstar);

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

			FlowVector U_outside = Boundary::outside_y(U_inside, U_inside,
					Boundary::index_n(i, j), -1, Zstar);

			U.H[(j-1)*cuda::pitch + i] = U_outside.H;
			U.HU[(j-1)*cuda::pitch + i] = U_outside.HU;
			U.HV[(j-1)*cuda::pitch + i] = U_outside.HV;
		}

		// south
		{
			int j = cuda::geometry.ysz;
			NUMERIC_TYPE Zstar = Zstar_y[j*cuda::pitch + i];
			FlowVector U_inside = U[j*cuda::pitch + i];

			FlowVector U_outside = Boundary::outside_y(U_inside, U_inside,
					Boundary::index_s(i, j), 1, Zstar);

			U.H[(j+1)*cuda::pitch + i] = U_outside.H;
			U.HU[(j+1)*cuda::pitch + i] = U_outside.HU;
			U.HV[(j+1)*cuda::pitch + i] = U_outside.HV;
		}
	}
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
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

			FlowVector U_neg;
			FlowVector Ustar_neg;
			FlowVector F_e;
			NUMERIC_TYPE Z_neg;
			NUMERIC_TYPE Zstar_e;

			// only update flow variables if (i, j) is inside the domain
			if (i <= cuda::geometry.xsz && j < row_end)
			{
				Zstar_e = Zstar_x[j*cuda::pitch + i];

				Z_neg = DEM[j*cuda::pitch + i];
				U_neg = Uold[j*cuda::pitch + i];
				Ustar_neg = U_neg.star(Z_neg, Zstar_e);

				NUMERIC_TYPE Z_pos = DEM[j*cuda::pitch + i+1];
				FlowVector U_pos = Uold[j*cuda::pitch + i+1];
				FlowVector Ustar_pos = U_pos.star(Z_pos, Zstar_e);

				if (i == 0)
				{
					Ustar_pos = Boundary::inside_x(Ustar_neg, Ustar_pos,
							Ustar_pos, Boundary::index_w(i, j));
				}
				else if (i == cuda::geometry.xsz)
				{
					Ustar_neg = Boundary::inside_x(Ustar_pos, Ustar_neg,
							Ustar_neg, Boundary::index_e(i, j));
				}

				Hstar[threadIdx.y][threadIdx.x] = Ustar_pos.H;

				F_e = HLL::x(Ustar_neg, Ustar_pos);
				F[threadIdx.y][threadIdx.x] = F_e;
				update_mass_stats_x(mass_stats, F_e.H, i, j);

			}

			__syncthreads();

			if (threadIdx.x != 0 && i <= cuda::geometry.xsz && j < row_end)
			{
				FlowVector& F_w = F[threadIdx.y][threadIdx.x-1];
				FlowVector U0 = Uold[j*cuda::pitch + i];

				NUMERIC_TYPE& H = U.H[j*cuda::pitch + i];
				NUMERIC_TYPE& HU = U.HU[j*cuda::pitch + i];
				NUMERIC_TYPE& HV = U.HV[j*cuda::pitch + i];

				NUMERIC_TYPE Zstar_w = Zstar_x[j*cuda::pitch + i-1];
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
__launch_bounds__(CUDA_BLOCK_SIZE)
update_flow_variables_y
(
	Flow Uold,
	Flow Uint,
	Flow U,
	NUMERIC_TYPE* DEM,
	NUMERIC_TYPE* Zstar_y,
	MassStats* mass_stats
)
{
        __shared__ FlowVector F[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
        __shared__ NUMERIC_TYPE Hstar[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];

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
			FlowVector U_pos;
			FlowVector Ustar_pos;
			FlowVector F_s;
			NUMERIC_TYPE Z_pos;
			NUMERIC_TYPE Zstar_s;

			if (i <= cuda::geometry.xsz && j <= cuda::geometry.ysz &&
					(cuda::fv1_owns_row(j) || cuda::fv1_owns_row(j+1)))
			{
				Zstar_s = Zstar_y[j*cuda::pitch + i];

				NUMERIC_TYPE Z_neg = DEM[(j+1)*cuda::pitch + i];
				FlowVector U_neg = Uold[(j+1)*cuda::pitch + i];
				FlowVector Ustar_neg = U_neg.star(Z_neg, Zstar_s);

				Z_pos = DEM[j*cuda::pitch + i];
				U_pos = Uold[j*cuda::pitch + i];
				Ustar_pos = U_pos.star(Z_pos, Zstar_s);

				// this hack appears necessary to prevent some unknown
				// compiler optimisation from causing a bug seen in
				// EA test 4
				//Ustar_pos.H = Ustar_pos.H*C(2.0)/C(2.0);

				if (j == 0)
				{       
					Ustar_neg = Boundary::inside_y(Ustar_pos, Ustar_neg,
							Ustar_neg, Boundary::index_n(i, j));
				}
				else if (j == cuda::geometry.ysz)
				{
					Ustar_pos = Boundary::inside_y(Ustar_neg, Ustar_pos,
							Ustar_pos, Boundary::index_s(i, j));
				}

				Hstar[threadIdx.y][threadIdx.x] = Ustar_neg.H;

				F_s = HLL::y(Ustar_neg, Ustar_pos);
				F[threadIdx.y][threadIdx.x] = F_s;
				if ((j == 0 && cuda::fv1_owns_row(1)) ||
						(j > 0 && cuda::fv1_owns_row(j)))
					update_mass_stats_y(mass_stats, F_s.H, i, j);
			}

                        __syncthreads();

			if (threadIdx.y != 0 && i <= cuda::geometry.xsz && j <= cuda::geometry.ysz && cuda::fv1_owns_row(j))
			{
				FlowVector& F_n = F[threadIdx.y-1][threadIdx.x];
				FlowVector U0 = Uint[j*cuda::pitch + i];

				NUMERIC_TYPE& H = U.H[j*cuda::pitch + i];
				NUMERIC_TYPE& HU = U.HU[j*cuda::pitch + i];
				NUMERIC_TYPE& HV = U.HV[j*cuda::pitch + i];

				NUMERIC_TYPE Zstar_n = Zstar_y[(j-1)*cuda::pitch + i];
				NUMERIC_TYPE Hstar_n = Hstar[threadIdx.y-1][threadIdx.x];
				NUMERIC_TYPE ETA = U_pos.H + Z_pos;

				H = U0.H - cuda::dt * (F_n.H - F_s.H)/cuda::geometry.dy;
				HU = U0.HU - cuda::dt * (F_n.HU - F_s.HU)/cuda::geometry.dy;
				HV = U0.HV - cuda::dt * ((F_n.HV - F_s.HV)/cuda::geometry.dy
					- bed_source_y(Zstar_s, Zstar_n, Ustar_pos.H, Hstar_n, ETA));
			
			}
			// shared memory is rewritten in the next pass: wait for every reader
			__syncthreads();
        }
    }
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

	Flow::copy(U1, U, geometry);

	friction = (physical_params.manning > C(0.0)) || (manning != nullptr);
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
	if (friction)
	{
		apply_friction<<<grid_size, cuda::block_size>>>(Uold, manning);
		update_ghost_cells();
	}

	update_flow_variables_x<<<grid_size, cuda::block_size>>>
			(Uold, Ux, DEM, Zstar_x, mass_stats);

	update_flow_variables_y<<<grid_size, cuda::block_size>>>
			(Uold, Ux, U, DEM, Zstar_y, mass_stats);

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
