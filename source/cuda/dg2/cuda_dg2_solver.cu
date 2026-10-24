#include "cuda_dg2_solver.cuh"
#include "cuda_hll.cuh"
#include "cuda_util.cuh"
#include <nvfunctional>
#include <algorithm>
#if defined(LISFLOOD_CUDA_MPI)
#include <mpi.h>
#include <vector>
#endif

__managed__ NUMERIC_TYPE lis::cuda::dg2::maxH;

#if defined(LISFLOOD_CUDA_MPI)
namespace {
static int stored_pitch_dg2 = 0;
static int mpi_row_begin_dg2 = 1, mpi_row_end_dg2 = 1;
static int mpi_rank_dg2 = 0, mpi_size_dg2 = 1;

__global__ void pack_mpi_halo_rows_dg2(
    NUMERIC_TYPE* H, NUMERIC_TYPE* H1x, NUMERIC_TYPE* H1y,
    NUMERIC_TYPE* HU, NUMERIC_TYPE* HU1x, NUMERIC_TYPE* HU1y,
    NUMERIC_TYPE* HV, NUMERIC_TYPE* HV1x, NUMERIC_TYPE* HV1y,
    int pitch, int north_row, int south_row, NUMERIC_TYPE* packed)
{
    const int n = 9 * pitch;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < 2 * n; idx += blockDim.x * gridDim.x) {
        const int side = idx / n;
        const int field_idx = idx - side * n;
        const int field = field_idx / pitch;
        const int column = field_idx - field * pitch;
        const int row = side == 0 ? north_row : south_row;
        const NUMERIC_TYPE* src =
            field == 0 ? H    : field == 1 ? H1x  : field == 2 ? H1y  :
            field == 3 ? HU   : field == 4 ? HU1x : field == 5 ? HU1y :
            field == 6 ? HV   : field == 7 ? HV1x : HV1y;
        packed[idx] = src[row * pitch + column];
    }
}

__global__ void unpack_mpi_halo_rows_dg2(
    NUMERIC_TYPE* H, NUMERIC_TYPE* H1x, NUMERIC_TYPE* H1y,
    NUMERIC_TYPE* HU, NUMERIC_TYPE* HU1x, NUMERIC_TYPE* HU1y,
    NUMERIC_TYPE* HV, NUMERIC_TYPE* HV1x, NUMERIC_TYPE* HV1y,
    int pitch, int north_halo, int south_halo,
    bool has_north, bool has_south, const NUMERIC_TYPE* packed)
{
    const int n = 9 * pitch;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < 2 * n; idx += blockDim.x * gridDim.x) {
        const int side = idx / n;
        if ((side == 0 && !has_south) || (side == 1 && !has_north)) continue;
        const int field_idx = idx - side * n;
        const int field = field_idx / pitch;
        const int column = field_idx - field * pitch;
        const int row = side == 0 ? south_halo : north_halo;
        NUMERIC_TYPE* dst =
            field == 0 ? H    : field == 1 ? H1x  : field == 2 ? H1y  :
            field == 3 ? HU   : field == 4 ? HU1x : field == 5 ? HU1y :
            field == 6 ? HV   : field == 7 ? HV1x : HV1y;
        dst[row * pitch + column] = packed[idx];
    }
}
}

void lis::cuda::dg2::configure_mpi_partition(
    int row_begin, int row_end, int rank, int size, int pitch)
{
    mpi_row_begin_dg2 = row_begin; mpi_row_end_dg2 = row_end;
    mpi_rank_dg2 = rank; mpi_size_dg2 = size; stored_pitch_dg2 = pitch;
    checkCudaErrors(cudaMemcpyToSymbol(cuda::fv1_row_begin, &row_begin, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(cuda::fv1_row_end, &row_end, sizeof(int)));
    const int active = 1;
    checkCudaErrors(cudaMemcpyToSymbol(cuda::fv1_partition_active, &active, sizeof(int)));
    const int north = rank == 0 ? 1 : 0;
    const int south = rank == size - 1 ? 1 : 0;
    checkCudaErrors(cudaMemcpyToSymbol(cuda::fv1_owns_north_ghost, &north, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(cuda::fv1_owns_south_ghost, &south, sizeof(int)));
    printf("CUDA DG2 MPI rank %d/%d owns rows [%d,%d)\n", rank, size, row_begin, row_end);
    fflush(stdout);
}

void lis::cuda::dg2::exchange_mpi_halos(Flow& U, int pitch)
{
    if (mpi_size_dg2 <= 1) return;
    const int n = 9 * pitch;
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
    pack_mpi_halo_rows_dg2<<<blocks, threads>>>(
        U.H, U.H1x, U.H1y, U.HU, U.HU1x, U.HU1y, U.HV, U.HV1x, U.HV1y,
        pitch, mpi_row_begin_dg2, mpi_row_end_dg2 - 1, d_send);
    const int north = mpi_rank_dg2 > 0 ? mpi_rank_dg2 - 1 : MPI_PROC_NULL;
    const int south = mpi_rank_dg2 + 1 < mpi_size_dg2
                    ? mpi_rank_dg2 + 1 : MPI_PROC_NULL;
    checkCudaErrors(cudaMemcpy(h_send, d_send, 2 * n * sizeof(NUMERIC_TYPE),
            cudaMemcpyDeviceToHost));
    const MPI_Datatype type =
            sizeof(NUMERIC_TYPE) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Request requests[4];
    MPI_Irecv(h_recv,     n, type, south, 46, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(h_recv + n, n, type, north, 47, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(h_send,     n, type, north, 46, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(h_send + n, n, type, south, 47, MPI_COMM_WORLD, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    checkCudaErrors(cudaMemcpyAsync(d_recv, h_recv,
            2 * n * sizeof(NUMERIC_TYPE), cudaMemcpyHostToDevice));
    unpack_mpi_halo_rows_dg2<<<blocks, threads>>>(
        U.H, U.H1x, U.H1y, U.HU, U.HU1x, U.HU1y, U.HV, U.HV1x, U.HV1y,
        pitch, mpi_row_begin_dg2 - 1, mpi_row_end_dg2,
        mpi_rank_dg2 > 0, mpi_rank_dg2 + 1 < mpi_size_dg2, d_recv);
}

void lis::cuda::dg2::gather_dg2_rows(Flow& flow, const Geometry& geometry,
    int pitch, int rank, int size)
{
    if (size <= 1) return;
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        int lo = geometry.ysz * r / size, hi = geometry.ysz * (r + 1) / size;
        counts[r] = (hi - lo) * geometry.xsz;
        displs[r] = lo * geometry.xsz;
    }
    const MPI_Datatype type =
        sizeof(NUMERIC_TYPE) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
    const int first = 1 + geometry.ysz * rank / size;
    const int last  = 1 + geometry.ysz * (rank + 1) / size;
    auto gather = [&](NUMERIC_TYPE* device) {
        std::vector<NUMERIC_TYPE> local((last - first) * geometry.xsz);
        std::vector<NUMERIC_TYPE> global;
        if (rank == 0) global.resize(geometry.ysz * geometry.xsz);
        for (int j = first; j < last; ++j)
            lis::cuda::copy(local.data() + (j - first) * geometry.xsz,
                device + j * pitch + 1, geometry.xsz * sizeof(NUMERIC_TYPE));
        MPI_Gatherv(local.data(), (int)local.size(), type,
            rank == 0 ? global.data() : nullptr,
            counts.data(), displs.data(), type, 0, MPI_COMM_WORLD);
        if (rank == 0)
            for (int j = 1; j <= geometry.ysz; ++j)
                lis::cuda::copy(device + j * pitch + 1,
                    global.data() + (j - 1) * geometry.xsz,
                    geometry.xsz * sizeof(NUMERIC_TYPE));
    };
    gather(flow.H);
    gather(flow.HU);
    gather(flow.HV);
}
#endif

namespace lis
{
namespace cuda
{
namespace dg2
{

__global__ void zero_ghost_cells_north_south
(
	Flow U
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x + 1;

	for (int i=global_i; i<cuda::pitch - 1; i+=blockDim.x*gridDim.x)
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
	int global_j = blockIdx.x*blockDim.x + threadIdx.x + 1;

	for (int j=global_j; j<cuda::geometry.ysz+1; j+=blockDim.x*gridDim.x)
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

__device__
NUMERIC_TYPE min_dt
(
	NUMERIC_TYPE dt,
	FlowVector U
)
{
    if (U.H > cuda::solver_params.DepthThresh)
    {
        NUMERIC_TYPE Uspeed = U.HU/U.H;
        NUMERIC_TYPE Vspeed = U.HV/U.H;
        NUMERIC_TYPE dt_local_x = cuda::solver_params.cfl*cuda::geometry.dx
			/ (FABS(Uspeed)+SQRT(cuda::physical_params.g*U.H));
        NUMERIC_TYPE dt_local_y = cuda::solver_params.cfl*cuda::geometry.dy
			/ (FABS(Vspeed)+SQRT(cuda::physical_params.g*U.H));
        return FMIN(dt, FMIN(dt_local_x, dt_local_y));
    }
    else
    {
        return dt;
    }
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
update_dt_per_element
(
	NUMERIC_TYPE* dt,
	Flow flow
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
#if defined(LISFLOOD_CUDA_MPI)
	const int j_begin = cuda::fv1_owned_row_begin();
	const int j_end   = cuda::fv1_owned_row_end();
#else
	const int j_begin = 0;
	const int j_end   = cuda::geometry.ysz + 2;
#endif

	for (int j=global_j; j<cuda::geometry.ysz+2; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<cuda::pitch; i+=blockDim.x*gridDim.x)
		{
#if defined(LISFLOOD_CUDA_MPI)
			if (j < j_begin || j >= j_end) continue;
#endif
			FlowCoeffs U = flow[j*cuda::pitch + i];
			NUMERIC_TYPE local_dt = cuda::solver_params.max_dt;
			local_dt = min_dt(local_dt, U.get_0());
			dt[j*cuda::pitch + i] = local_dt;
		}
	}
}

__global__
void update_uniform_rain_func
(
	Flow U,
	DeviceTopography DEM,
	NUMERIC_TYPE  rain_rate
)
{
	int global_i = blockIdx.x * blockDim.x + threadIdx.x;
	int global_j = blockIdx.y * blockDim.y + threadIdx.y;

	for (int j = global_j; j < cuda::geometry.ysz + 2; j += blockDim.y * gridDim.y)
	{
		for (int i = global_i; i < cuda::pitch; i += blockDim.x * gridDim.x)
		{
			NUMERIC_TYPE cell_rain;

			cell_rain = rain_rate * cuda::dt;

			NUMERIC_TYPE Z = DEM._0[j * cuda::pitch + i];
			if (FABS(Z - cuda::solver_params.nodata_elevation) < C(1e-6) /* || Z < C(0.0) */) {

			}
			else {

				NUMERIC_TYPE& Hval = U.H[j * cuda::pitch + i];
				Hval += cell_rain;
			}
		}
	}
}

__device__
FlowCoeffs bed_source_x
(
	NUMERIC_TYPE Hstar_0x,
	NUMERIC_TYPE Hstar_1x,
	DeviceTopography DEM,
	FlowCoeffs U,
	int i,
	int j
)
{
	NUMERIC_TYPE ETA_w = U.pos_x().H + DEM.pos_x(j*cuda::pitch + i);
	NUMERIC_TYPE ETA_e = U.neg_x().H + DEM.neg_x(j*cuda::pitch + i);
	NUMERIC_TYPE Zstar_w = DEM.Zstar_x[j*cuda::pitch + i-1];
	NUMERIC_TYPE Zstar_e = DEM.Zstar_x[j*cuda::pitch + i];
	NUMERIC_TYPE Zdagger_w = DEM.Zdagger(ETA_w, Zstar_w);
	NUMERIC_TYPE Zdagger_e = DEM.Zdagger(ETA_e, Zstar_e);

	NUMERIC_TYPE Zdagger_1x = (Zdagger_e - Zdagger_w) / (C(2.0)*SQRT(C(3.0)));

	NUMERIC_TYPE g = cuda::physical_params.g;

	FlowCoeffs Sbx = {};
	Sbx.HU = -C(2.0)*SQRT(C(3.0)) * g*Hstar_0x*Zdagger_1x / cuda::geometry.dx;
	Sbx.HU1x = -C(2.0)*SQRT(C(3.0)) * g*Hstar_1x*Zdagger_1x / cuda::geometry.dx;

	return Sbx;
}

__device__
FlowCoeffs bed_source_y
(
	NUMERIC_TYPE Hstar_0y,
	NUMERIC_TYPE Hstar_1y,
	DeviceTopography DEM,
	FlowCoeffs U,
	int i,
	int j
)
{
	NUMERIC_TYPE ETA_s = U.pos_y().H + DEM.pos_y(j*cuda::pitch + i);
	NUMERIC_TYPE ETA_n = U.neg_y().H + DEM.neg_y(j*cuda::pitch + i);
	NUMERIC_TYPE Zstar_s = DEM.Zstar_y[j*cuda::pitch + i];
	NUMERIC_TYPE Zstar_n = DEM.Zstar_y[(j-1)*cuda::pitch + i];
	NUMERIC_TYPE Zdagger_s = DEM.Zdagger(ETA_s, Zstar_s);
	NUMERIC_TYPE Zdagger_n = DEM.Zdagger(ETA_n, Zstar_n);

	NUMERIC_TYPE Zdagger_1y = (Zdagger_n - Zdagger_s) / (C(2.0)*SQRT(C(3.0)));

	NUMERIC_TYPE g = cuda::physical_params.g;

	FlowCoeffs Sby = {};
	Sby.HV = -C(2.0)*SQRT(C(3.0)) * g*Hstar_0y*Zdagger_1y / cuda::geometry.dy;
	Sby.HV1y = -C(2.0)*SQRT(C(3.0)) * g*Hstar_1y*Zdagger_1y / cuda::geometry.dy;

	return Sby;
}

__device__
void update_flow_variables_x
(
	DeviceTopography DEM,
	Flow U_in,
	Flow U_out,
	const nvstd::function<FlowCoeffs(int i, int j, FlowCoeffs)>& rk_op,
	MassStats* mass_stats
)
{
	__shared__ FlowVector F[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
	__shared__ FlowVector Ustar[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];

	int global_i = blockIdx.x*(blockDim.x-1) + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	int max_i = ((cuda::geometry.xsz+blockDim.x-2)/(blockDim.x-1))*(blockDim.x-1);
	int max_j = ((cuda::geometry.ysz+blockDim.y-1)/blockDim.y)*blockDim.y;

	for (int j=global_j+1; j<=max_j; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<=max_i; i+=(blockDim.x-1)*gridDim.x)
		{
			// blocks overlap in the x direction
			// so don't start a new block at the right-hand edge
			if (threadIdx.x == 0 && i == max_i) continue;

			FlowVector Ustar_neg;
			FlowVector Ustar_pos;
			FlowVector F_e;
			FlowCoeffs U_w;

			// only update flow variables if (i, j) is inside the domain
			if (i <= cuda::geometry.xsz && j <= cuda::geometry.ysz)
			{
				U_w = U_in[j*cuda::pitch + i];
				FlowCoeffs U_e = U_in[j*cuda::pitch + i+1];

				NUMERIC_TYPE Z_neg = DEM.neg_x(j*cuda::pitch + i);

				NUMERIC_TYPE Z_pos = DEM.pos_x(j*cuda::pitch + i+1);

				NUMERIC_TYPE Zstar = DEM.Zstar_x[j*cuda::pitch + i];

				Ustar_neg = U_w.neg_x().star(Z_neg, Zstar);

				Ustar_pos = U_e.pos_x().star(Z_pos, Zstar);

				Ustar[threadIdx.y][threadIdx.x] = Ustar_pos;

				FlowVector Ustar_neg_flux = Ustar_neg;
				if (i == 0)
				{
					Ustar_pos = Boundary::inside_x(Ustar_neg, U_e.get_0(),
							Ustar_pos, Boundary::index_w(i, j));
				}
				else if (i == cuda::geometry.xsz)
				{
					Ustar_neg_flux = Boundary::inside_x(Ustar_pos, U_w.get_0(),
							Ustar_neg, Boundary::index_e(i, j));
				}

				F_e = HLL::x(Ustar_neg_flux, Ustar_pos);
				F[threadIdx.y][threadIdx.x] = F_e;
				update_mass_stats_x(mass_stats, F_e.H, i, j, C(0.5));
			}

			__syncthreads();

			if (threadIdx.x != 0 && i <= cuda::geometry.xsz && j <= cuda::geometry.ysz)
			{
				FlowVector& F_w = F[threadIdx.y][threadIdx.x-1];
				Ustar_pos = Ustar[threadIdx.y][threadIdx.x-1];
				FlowVector Ustar_0x = C(0.5)*(Ustar_pos + Ustar_neg);
				FlowVector Ustar_1x = (Ustar_neg - Ustar_pos)/(C(2.0)*SQRT(C(3.0)));

				FlowCoeffs Lx = {};
				
					Lx.set_0(-(F_e - F_w)/cuda::geometry.dx);
					Lx.set_1x(-SQRT(C(3.0))/cuda::geometry.dx * (F_w + F_e
								- (Ustar_0x - Ustar_1x).physical_flux_x()
								- (Ustar_0x + Ustar_1x).physical_flux_x()));
					Lx += bed_source_x(Ustar_0x.H, Ustar_1x.H, DEM, U_w, i, j);
				U_out[j*cuda::pitch + i] = rk_op(i, j, Lx);
			}
			// shared memory is rewritten in the next pass: wait for every reader
			__syncthreads();
		}
	}
}

__device__
void update_flow_variables_y
(
	DeviceTopography DEM,
	Flow U_in,
	Flow U_out,
	const nvstd::function<FlowCoeffs(int i, int j, FlowCoeffs)>& rk_op,
	MassStats* mass_stats
)
{
	__shared__ FlowVector F[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];
	__shared__ FlowVector Ustar[CUDA_BLOCK_SIZE_Y][CUDA_BLOCK_SIZE_X];

	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*(blockDim.y-1) + threadIdx.y;

	int max_i = ((cuda::geometry.xsz+blockDim.x-1)/blockDim.x)*blockDim.x;
	int max_j = ((cuda::geometry.ysz+blockDim.y-2)/(blockDim.y-1))*(blockDim.y-1);

	for (int j=global_j; j<=max_j; j+=(blockDim.y-1)*gridDim.y)
	{
		if (threadIdx.y == 0 && j == max_j) continue;

		for (int i=global_i+1; i<=max_i; i+=blockDim.x*gridDim.x)
		{
			FlowVector Ustar_neg;
			FlowVector Ustar_pos;
			FlowVector F_s;
			FlowCoeffs U_n;

			if (i <= cuda::geometry.xsz && j <= cuda::geometry.ysz)
			{
				U_n = U_in[j*cuda::pitch + i];
				FlowCoeffs U_s = U_in[(j+1)*cuda::pitch + i];

				NUMERIC_TYPE Z_neg = DEM.neg_y((j + 1) * cuda::pitch + i);

				NUMERIC_TYPE Z_pos = DEM.pos_y(j * cuda::pitch + i);

				NUMERIC_TYPE Zstar = DEM.Zstar_y[j * cuda::pitch + i];

				Ustar_neg = U_s.neg_y().star(Z_neg, Zstar);

				Ustar_pos = U_n.pos_y().star(Z_pos, Zstar);

				Ustar[threadIdx.y][threadIdx.x] = Ustar_neg;

				FlowVector Ustar_pos_flux = Ustar_pos;
				if (j == 0)
				{
					Ustar_neg = Boundary::inside_y(Ustar_pos, U_s.get_0(),
							Ustar_neg, Boundary::index_n(i, j));
				}
				else if (j == cuda::geometry.ysz)
				{
					Ustar_pos_flux = Boundary::inside_y(Ustar_neg, U_n.get_0(),
							Ustar_pos, Boundary::index_s(i, j));
				}

				F_s = HLL::y(Ustar_neg, Ustar_pos_flux);

				F[threadIdx.y][threadIdx.x] = F_s;
				update_mass_stats_y(mass_stats, F_s.H, i, j, C(0.5));
			}

			__syncthreads();

			if (threadIdx.y != 0 && i <= cuda::geometry.xsz && j <= cuda::geometry.ysz)
			{
				FlowVector& F_n = F[threadIdx.y-1][threadIdx.x];
				Ustar_neg = Ustar[threadIdx.y-1][threadIdx.x];

				FlowVector Ustar_0y = C(0.5)*(Ustar_pos + Ustar_neg);
				FlowVector Ustar_1y = (Ustar_neg - Ustar_pos)/(C(2.0)*SQRT(C(3.0)));

				FlowCoeffs Ly = {};
				
					Ly.set_0(-(F_n - F_s) / cuda::geometry.dy);
					Ly.set_1y(-SQRT(C(3.0)) / cuda::geometry.dy * (F_s + F_n
						- (Ustar_0y - Ustar_1y).physical_flux_y()
						- (Ustar_0y + Ustar_1y).physical_flux_y()));
					Ly += bed_source_y(Ustar_0y.H, Ustar_1y.H, DEM, U_n, i, j);
				U_out[j*cuda::pitch + i] = rk_op(i, j, Ly);
			}
			// shared memory is rewritten in the next pass: wait for every reader
			__syncthreads();
		}
	}
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
rk_stage1_x
(
	DeviceTopography DEM,
	Flow Uold,
	Flow Ux,
	MassStats* mass_stats
)
{

	update_flow_variables_x(DEM, Uold, Ux, [&](int i, int j, FlowCoeffs Lx)
	{
#if defined(LISFLOOD_CUDA_MPI)
		if (j < cuda::fv1_owned_row_begin() || j >= cuda::fv1_owned_row_end()) return FlowCoeffs(Uold[j*cuda::pitch + i]);
#endif
		return FlowCoeffs(Uold[j*cuda::pitch + i] + cuda::dt*Lx);
	}, mass_stats);
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
rk_stage1_y
(
	DeviceTopography DEM,
	Flow Uold,
	Flow Ux,
	Flow Uint,
	MassStats* mass_stats
)
{
	update_flow_variables_y(DEM, Uold, Uint, [&](int i, int j, FlowCoeffs Ly)
	{
#if defined(LISFLOOD_CUDA_MPI)
		if (j < cuda::fv1_owned_row_begin() || j >= cuda::fv1_owned_row_end()) return FlowCoeffs(Uold[j*cuda::pitch + i]);
#endif
		return FlowCoeffs(Ux[j*cuda::pitch + i] + cuda::dt*Ly);
	}, mass_stats);
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
rk_stage2_x
(
	DeviceTopography DEM,
	Flow Uold,
	Flow Uint,
	Flow Ux,
	MassStats* mass_stats
)
{
	update_flow_variables_x(DEM, Uint, Ux, [&](int i, int j, FlowCoeffs Lx)
	{
#if defined(LISFLOOD_CUDA_MPI)
		if (j < cuda::fv1_owned_row_begin() || j >= cuda::fv1_owned_row_end()) return FlowCoeffs(Uint[j*cuda::pitch + i]);
#endif
		return FlowCoeffs(C(0.5)*(Uold[j*cuda::pitch + i] +
				Uint[j*cuda::pitch + i] + cuda::dt*Lx));
	}, mass_stats);
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
rk_stage2_y
(
	DeviceTopography DEM,
	Flow Uint,
	Flow Ux,
	Flow U,
	MassStats* mass_stats
)
{
	update_flow_variables_y(DEM, Uint, U, [&](int i, int j, FlowCoeffs Ly)
	{
#if defined(LISFLOOD_CUDA_MPI)
		if (j < cuda::fv1_owned_row_begin() || j >= cuda::fv1_owned_row_end()) return FlowCoeffs(Ux[j*cuda::pitch + i]);
#endif
		return FlowCoeffs(Ux[j*cuda::pitch + i] + C(0.5)*cuda::dt*Ly);
	}, mass_stats);
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
zero_discharge
(
	Flow flow
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
#if defined(LISFLOOD_CUDA_MPI)
	const int j_begin = cuda::fv1_owned_row_begin();
	const int j_end   = cuda::fv1_owned_row_end();
#endif

	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
#if defined(LISFLOOD_CUDA_MPI)
			if (j < j_begin || j >= j_end) continue;
#endif
			FlowCoeffsRef U = flow[j*cuda::pitch + i];
			if (U.H < cuda::solver_params.DepthThresh)
			{
				U.HU = C(0.0);
				U.HU1x = C(0.0);
				U.HU1y = C(0.0);
				U.HV = C(0.0);
				U.HV1x = C(0.0);
				U.HV1y = C(0.0);
			}
		}
	}
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
zero_thin_depth_slopes
(
	DeviceTopography DEM,
	Flow flow
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			FlowCoeffsRef U = flow[j*cuda::pitch + i];
			if (U.H <= C(10.0)*cuda::solver_params.DepthThresh)
			{
				flow.H1x[j * cuda::pitch + i] = C(0.0);
				flow.H1y[j * cuda::pitch + i] = C(0.0);
				flow.HU1x[j * cuda::pitch + i] = C(0.0);
				flow.HU1y[j * cuda::pitch + i] = C(0.0);
				flow.HV1x[j * cuda::pitch + i] = C(0.0);
				flow.HV1y[j * cuda::pitch + i] = C(0.0);
			}
		}
	}
}

__global__ void update_ghost_cells
(
	DeviceTopography DEM,
	Flow flow
)
{
	for (int j=blockIdx.x*blockDim.x+threadIdx.x+1; j<cuda::geometry.ysz+1;
			j+=blockDim.x*gridDim.x)
	{
		// west
		{
			int i = 1;
			NUMERIC_TYPE Z_pos = DEM.pos_x(j*cuda::pitch + i);
			NUMERIC_TYPE Zstar = DEM.Zstar_x[j*cuda::pitch + i-1];
			FlowCoeffs U = flow[j*cuda::pitch + i];
			FlowVector U_inside = U.pos_x().star(Z_pos, Zstar);

			FlowVector U_outside = Boundary::outside_x(U.get_0(), U_inside,
					Boundary::index_w(i, j), 1, Zstar);

			if (cuda::boundaries.BC_type[Boundary::index_w(i, j)] == FREE1 && flow[j * cuda::pitch + i].H <= cuda::solver_params.DepthThresh) {
				flow[j * cuda::pitch + i].H = C(0.0);
				flow[j * cuda::pitch + i].H1x = C(0.0);
				flow[j * cuda::pitch + i].H1y = C(0.0);
				U_outside.H = C(0.0);
			}

			flow[j*cuda::pitch + i-1].set_0(U_outside);
		}

		// east
		{
			int i = cuda::geometry.xsz;
			NUMERIC_TYPE Z_neg = DEM.neg_x(j*cuda::pitch + i);
			NUMERIC_TYPE Zstar = DEM.Zstar_x[j*cuda::pitch + i];
			FlowCoeffs U = flow[j*cuda::pitch + i];
			FlowVector U_inside = U.neg_x().star(Z_neg, Zstar);

			FlowVector U_outside = Boundary::outside_x(U.get_0(), U_inside,
					Boundary::index_e(i, j), -1, Zstar);

			if (cuda::boundaries.BC_type[Boundary::index_e(i, j)] == FREE1 && flow[j * cuda::pitch + i].H <= cuda::solver_params.DepthThresh) {
				flow[j * cuda::pitch + i].H = C(0.0);
				flow[j * cuda::pitch + i].H1x = C(0.0);
				flow[j * cuda::pitch + i].H1y = C(0.0);
				U_outside.H = C(0.0);
			}

			flow[j*cuda::pitch + i+1].set_0(U_outside);
		}
	}

	for (int i=blockIdx.x*blockDim.x+threadIdx.x+1; i<cuda::geometry.xsz+1;
			i+=blockDim.x*gridDim.x)
	{
		// north
		{
			int j = 1;
			NUMERIC_TYPE Z_neg = DEM.neg_y(j*cuda::pitch + i);
			NUMERIC_TYPE Zstar = DEM.Zstar_y[(j-1)*cuda::pitch + i];
			FlowCoeffs U = flow[j*cuda::pitch + i];
			FlowVector U_inside = U.neg_y().star(Z_neg, Zstar);

			FlowVector U_outside = Boundary::outside_y(U.get_0(), U_inside,
					Boundary::index_n(i, j), -1, Zstar);

			if (cuda::boundaries.BC_type[Boundary::index_n(i, j)] == FREE1 && flow[j * cuda::pitch + i].H <= cuda::solver_params.DepthThresh) {
				flow[j * cuda::pitch + i].H = C(0.0);
				flow[j * cuda::pitch + i].H1x = C(0.0);
				flow[j * cuda::pitch + i].H1y = C(0.0);
				U_outside.H = C(0.0);
			}

			flow[(j-1)*cuda::pitch + i].set_0(U_outside);
		}

		// south
		{
			int j = cuda::geometry.ysz;
			NUMERIC_TYPE Z_pos = DEM.pos_y(j*cuda::pitch + i);
			NUMERIC_TYPE Zstar = DEM.Zstar_y[j*cuda::pitch + i];
			FlowCoeffs U = flow[j*cuda::pitch + i];
			FlowVector U_inside = U.pos_y().star(Z_pos, Zstar);

			FlowVector U_outside = Boundary::outside_y(U.get_0(), U_inside,
					Boundary::index_s(i, j), 1, Zstar);

			if (cuda::boundaries.BC_type[Boundary::index_s(i, j)] == FREE1 && flow[j * cuda::pitch + i].H <= cuda::solver_params.DepthThresh) {
				flow[j * cuda::pitch + i].H = C(0.0);
				flow[j * cuda::pitch + i].H1x = C(0.0);
				flow[j * cuda::pitch + i].H1y = C(0.0);
				U_outside.H = C(0.0);
			}

			flow[(j+1)*cuda::pitch + i].set_0(U_outside);
		}
	}
}

__device__
void friction
(
    NUMERIC_TYPE n,
    NUMERIC_TYPE H,
    NUMERIC_TYPE& HU,
    NUMERIC_TYPE& HV
)
{
    if (H <= cuda::solver_params.DepthThresh)
    {
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

    NUMERIC_TYPE speed = SQRT(U*U+V*V);
    NUMERIC_TYPE Cf = cuda::physical_params.g * n * n
        / POW(H, C(1.0)/C(3.0));
    NUMERIC_TYPE Sfx = -Cf*U*speed;
    NUMERIC_TYPE Sfy = -Cf*V*speed;
    NUMERIC_TYPE Dx = C(1.0) + cuda::dt*Cf/H * (C(2.0)*U*U + V*V)/speed;
    NUMERIC_TYPE Dy = C(1.0) + cuda::dt*Cf/H * (U*U + C(2.0)*V*V)/speed;
    HU += cuda::dt * Sfx / Dx;
    HV += cuda::dt * Sfy / Dy;
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
apply_friction
(
	Flow flow,
	NUMERIC_TYPE* manning
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;
#if defined(LISFLOOD_CUDA_MPI)
	const int j_begin = cuda::fv1_owned_row_begin();
	const int j_end   = cuda::fv1_owned_row_end();
#endif

	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
#if defined(LISFLOOD_CUDA_MPI)
			if (j < j_begin || j >= j_end) continue;
#endif
			FlowCoeffsRef Uref = flow[j*cuda::pitch + i];
			FlowCoeffs U(Uref);

			FlowVector U_lower_x = U.gauss_lower_x();
			NUMERIC_TYPE HU_lower_x = U_lower_x.HU;
			NUMERIC_TYPE HV_lower_x = U_lower_x.HV;

			FlowVector U_upper_x = U.gauss_upper_x();
			NUMERIC_TYPE HU_upper_x = U_upper_x.HU;
			NUMERIC_TYPE HV_upper_x = U_upper_x.HV;

			FlowVector U_lower_y = U.gauss_lower_y();
			NUMERIC_TYPE HU_lower_y = U_lower_y.HU;
			NUMERIC_TYPE HV_lower_y = U_lower_y.HV;

			FlowVector U_upper_y = U.gauss_upper_y();
			NUMERIC_TYPE HU_upper_y = U_upper_y.HU;
			NUMERIC_TYPE HV_upper_y = U_upper_y.HV;

			NUMERIC_TYPE n = (manning == nullptr)
				? cuda::physical_params.manning
				: manning[j*cuda::pitch + i];

			friction(n, U.H, Uref.HU, Uref.HV);

				friction(n, U_lower_x.H, HU_lower_x, HV_lower_x);
				friction(n, U_upper_x.H, HU_upper_x, HV_upper_x);
				friction(n, U_lower_y.H, HU_lower_y, HV_lower_y);
				friction(n, U_upper_y.H, HU_upper_y, HV_upper_y);

				Uref.HU1x = C(0.5) * (HU_upper_x - HU_lower_x);
				Uref.HU1y = C(0.5) * (HU_upper_y - HU_lower_y);
				Uref.HV1x = C(0.5) * (HV_upper_x - HV_lower_x);
				Uref.HV1y = C(0.5) * (HV_upper_y - HV_lower_y);

		}
	}
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
copy_slopes
(
	DeviceTopography DEM,
	Flow flow,
	Slopes slopes
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j; j<cuda::geometry.ysz+2; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i; i<cuda::pitch; i+=blockDim.x*gridDim.x)
		{
			NUMERIC_TYPE Z1x = DEM._1x[j*cuda::pitch + i];
			NUMERIC_TYPE Z1y = DEM._1y[j*cuda::pitch + i];

			FlowCoeffs U = flow[j*cuda::pitch + i];

			slopes.ETA1x[j*cuda::pitch + i] = Z1x + U.H1x;
			slopes.ETA1y[j*cuda::pitch + i] = Z1y + U.H1y;
			slopes.HU1x[j*cuda::pitch + i] = U.HU1x;
			slopes.HU1y[j*cuda::pitch + i] = U.HU1y;
			slopes.HV1x[j*cuda::pitch + i] = U.HV1x;
			slopes.HV1y[j*cuda::pitch + i] = U.HV1y;
		}
	}
}

__global__ void
zero_perimeter_slopes_north_south
(
	DeviceTopography DEM,
	Flow flow
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x + 1;

	for (int i=global_i; i<cuda::pitch - 1; i+=blockDim.x*gridDim.x)
	{
		{
			int j=1;
			FlowCoeffsRef U = flow[j*cuda::pitch + i];
			U.H1y = C(0.0);
			U.HU1y = C(0.0);
			U.HV1y = C(0.0);
			U.H1x = C(0.0);
			U.HU1x = C(0.0);
			U.HV1x = C(0.0);
		}

		{
			int j=cuda::geometry.ysz;
			FlowCoeffsRef U = flow[j*cuda::pitch + i];
			U.H1y = C(0.0);
			U.HU1y = C(0.0);
			U.HV1y = C(0.0);
			U.H1x = C(0.0);
			U.HU1x = C(0.0);
			U.HV1x = C(0.0);
		}
	}
}

__global__ void
zero_perimeter_slopes_east_west
(
	DeviceTopography DEM,
	Flow flow
)
{
	int global_j = blockIdx.x*blockDim.x + threadIdx.x + 1;

	for (int j=global_j; j<cuda::geometry.ysz+1; j+=blockDim.x*gridDim.x)
	{
		{
			int i=1;
			FlowCoeffsRef U = flow[j*cuda::pitch + i];
			U.H1x = C(0.0);
			U.HU1x = C(0.0);
			U.HV1x = C(0.0);
			U.H1y = C(0.0);
			U.HU1y = C(0.0);
			U.HV1y = C(0.0);
		}

		{
			int i=cuda::geometry.xsz;
			FlowCoeffsRef U = flow[j*cuda::pitch + i];
			U.H1x = C(0.0);
			U.HU1x = C(0.0);
			U.HV1x = C(0.0);
			U.H1y = C(0.0);
			U.HU1y = C(0.0);
			U.HV1y = C(0.0);
		}
	}
}

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
limit_slopes
(
	DeviceTopography DEM,
	Flow flow,
	Slopes slopes
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			NUMERIC_TYPE Z1x = DEM._1x[j*cuda::pitch + i];
			NUMERIC_TYPE Z1y = DEM._1y[j*cuda::pitch + i];
			FlowCoeffsRef U = flow[j*cuda::pitch + i];

			NUMERIC_TYPE minH_x = Stencil::minH_x(flow, i, j);
			Stencil ETA_stencil_x = slopes.ETA_stencil_x(DEM._0, flow.H, i, j);
			U.H1x = ETA_stencil_x.limit(cuda::geometry.dx, minH_x) - Z1x;
			Stencil HU_stencil_x = slopes.HU_stencil_x(flow.HU, i, j);
			U.HU1x = HU_stencil_x.limit(cuda::geometry.dx, minH_x);
			Stencil HV_stencil_x = slopes.HV_stencil_x(flow.HV, i, j);
			U.HV1x = HV_stencil_x.limit(cuda::geometry.dx, minH_x);

			NUMERIC_TYPE minH_y = Stencil::minH_y(flow, i, j);
			Stencil ETA_stencil_y = slopes.ETA_stencil_y(DEM._0, flow.H, i, j);
			U.H1y = ETA_stencil_y.limit(cuda::geometry.dy, minH_y) - Z1y;
			Stencil HU_stencil_y = slopes.HU_stencil_y(flow.HU, i, j);
			U.HU1y = HU_stencil_y.limit(cuda::geometry.dy, minH_y);
			Stencil HV_stencil_y = slopes.HV_stencil_y(flow.HV, i, j);
			U.HV1y = HV_stencil_y.limit(cuda::geometry.dy, minH_y);
		}
	}
}

}
}
}

lis::cuda::dg2::Solver::Solver
(
	Flow& U,
	DeviceTopography& DEM,
	NUMERIC_TYPE* manning,
	Geometry& geometry,
	PhysicalParams& physical_params,
	bool limit_slopes,
	dim3 grid_size
)
:
Uold(U1),
DEM(DEM),
manning(manning),
limit_slopes(limit_slopes),
grid_size(grid_size),
maxH(geometry)
{
	Flow::allocate_device(U1, geometry);
	Flow::allocate_device(Ux, geometry);
	Flow::allocate_device(Uint, geometry);

	Flow::copy(U1, U, geometry);

	Slopes::allocate_device(slopes, geometry);

	friction = (physical_params.manning > C(0.0)) || (manning != nullptr);
}

void lis::cuda::dg2::Solver::zero_ghost_cells()
{
	zero_ghost_cells_north_south<<<1, CUDA_BLOCK_SIZE>>>(Uold);
	zero_ghost_cells_east_west<<<1, CUDA_BLOCK_SIZE>>>(Uold);
}

void lis::cuda::dg2::Solver::update_ghost_cells()
{
	update_ghost_cells(Uold);
}

void lis::cuda::dg2::Solver::update_ghost_cells
(
	Flow& U
)
{
	lis::cuda::dg2::update_ghost_cells<<<64, CUDA_BLOCK_SIZE>>>(DEM, U);
}


void lis::cuda::dg2::Solver::zero_thin_depth_slopes()
{
	zero_thin_depth_slopes(Uold);
}

void lis::cuda::dg2::Solver::zero_thin_depth_slopes
(
    Flow& U
)
{
	lis::cuda::dg2::zero_thin_depth_slopes<<<grid_size, cuda::block_size>>>(DEM, U);
}

void lis::cuda::dg2::Solver::zero_perimeter_slopes
(
	DeviceTopography& DEM,
 	Flow& flow
)
{
	zero_perimeter_slopes_north_south<<<1, CUDA_BLOCK_SIZE>>>(DEM, flow);
	zero_perimeter_slopes_east_west<<<1, CUDA_BLOCK_SIZE>>>(DEM, flow);
}

lis::cuda::dg2::Flow& lis::cuda::dg2::Solver::update_flow_variables
(
	MassStats* mass_stats
)
{
	if (friction)
	{
		apply_friction<<<grid_size, cuda::block_size>>>(Uold, manning);
	}

	if (limit_slopes)
	{
		maxH.update(Uold);
	}

	update_ghost_cells(Uold);
	if (limit_slopes)
	{
		copy_slopes<<<grid_size, cuda::block_size>>>(DEM, Uold, slopes);
		dg2::limit_slopes<<<grid_size, cuda::block_size>>>(DEM, Uold, slopes);
	}
	zero_perimeter_slopes(DEM, Uold);
	rk_stage1_x<<<grid_size, cuda::block_size>>>(DEM, Uold, Ux, mass_stats);
	rk_stage1_y<<<grid_size, cuda::block_size>>>(DEM, Uold, Ux, Uint, mass_stats);
	zero_discharge<<<grid_size, cuda::block_size>>>(Uint);
#if defined(LISFLOOD_CUDA_MPI)
	lis::cuda::dg2::exchange_mpi_halos(Uint, stored_pitch_dg2);
#endif
	update_ghost_cells(Uint);
	if (limit_slopes)
	{
		copy_slopes<<<grid_size, cuda::block_size>>>(DEM, Uint, slopes);
		dg2::limit_slopes<<<grid_size, cuda::block_size>>>(DEM, Uint, slopes);
	}
	zero_perimeter_slopes(DEM, Uint);
	rk_stage2_x<<<grid_size, cuda::block_size>>>(DEM, Uold, Uint, Ux, mass_stats);
	rk_stage2_y<<<grid_size, cuda::block_size>>>(DEM, Uint, Ux, Uold, mass_stats);
	zero_discharge<<<grid_size, cuda::block_size>>>(Uold);

	return Uold;
}

lis::cuda::dg2::Flow& lis::cuda::dg2::Solver::d_U()
{
	return Uold;
}

lis::cuda::dg2::Flow& lis::cuda::dg2::Solver::d_Uint()
{
	return Uint;
}

void lis::cuda::dg2::Solver::update_dt_per_element
(
	NUMERIC_TYPE* dt_field
) const
{
	lis::cuda::dg2::update_dt_per_element<<<grid_size, cuda::block_size>>>(
			dt_field, Uold);
}


void lis::cuda::dg2::Solver::FloodplainQ()
{

  return;	
}

void lis::cuda::dg2::Solver::updateMaxFieldACC
(
	NUMERIC_TYPE t
)
{
	return;
}

void lis::cuda::dg2::Solver::update_uniform_rain
(
	NUMERIC_TYPE rain_rate
)
{
	update_uniform_rain_func << <grid_size, cuda::block_size >> > (
		Uold, DEM, rain_rate);
}

lis::cuda::dg2::Solver::~Solver()
{
	Flow::free_device(U1);
	Flow::free_device(Ux);
	Flow::free_device(Uint);
	Slopes::free_device(slopes);
}
