#pragma once
#include "cuda_solver.cuh"
#include "cuda_hll.cuh"
#include "cuda_losses.cuh"

namespace lis
{
namespace cuda
{

/**
 * The bodies of the device functions that the solvers share: the HLL
 * fluxes, the star states, the boundary conditions, the mass statistics,
 * the losses and the rain rate of a cell. Their out-of-line versions (HLL::x, FlowVector::star,
 * Boundary::outside_x, update_mass_stats_x, depth_after_losses, ...) call
 * these, so that every solver compiles to what it did before; the FV1
 * kernels call these directly, to have them inlined without device link-time
 * optimisation (which would change the rounding of the other solvers).
 */
namespace detail
{

__device__ __forceinline__ FlowVector hll_x
(
	const FlowVector& u_neg,
	const FlowVector& u_pos
)
{
	if (u_neg.H <= cuda::solver_params.DepthThresh &&
			u_pos.H <= cuda::solver_params.DepthThresh)
	{
		return FlowVector();
	}

	NUMERIC_TYPE U_neg = C(0.0);
	NUMERIC_TYPE V_neg = C(0.0);
	if (u_neg.H > cuda::solver_params.DepthThresh)
	{
		U_neg = u_neg.HU / u_neg.H;
		V_neg = u_neg.HV / u_neg.H;
	}

	NUMERIC_TYPE U_pos = C(0.0); 
	NUMERIC_TYPE V_pos = C(0.0);
	if (u_pos.H > cuda::solver_params.DepthThresh)
	{
		U_pos = u_pos.HU / u_pos.H;
		V_pos = u_pos.HV / u_pos.H;
	}

	NUMERIC_TYPE A_neg = SQRT(cuda::physical_params.g * u_neg.H);
	NUMERIC_TYPE A_pos = SQRT(cuda::physical_params.g * u_pos.H);

	NUMERIC_TYPE H_star = POW(C(0.5)*(A_neg+A_pos) + C(0.25)*(U_neg - U_pos),
			C(2.0))	/ cuda::physical_params.g;
	NUMERIC_TYPE U_star = C(0.5)*(U_neg + U_pos) + A_neg - A_pos;
	NUMERIC_TYPE A_star = SQRT(cuda::physical_params.g*H_star);

	NUMERIC_TYPE S_neg;
	if (u_neg.H <= cuda::solver_params.DepthThresh)
	{
		S_neg = U_pos - C(2.0)*A_pos;
	}
	else
	{
		S_neg = FMIN(U_neg - A_neg, U_star - A_star);
	}

	NUMERIC_TYPE S_pos;
	if (u_pos.H <= cuda::solver_params.DepthThresh)
	{
		S_pos = U_neg + C(2.0)*A_neg;
	}
	else
	{
		S_pos = FMAX(U_pos + A_pos, U_star + A_star);
	}

	FlowVector F_neg = {
		u_neg.HU,
		U_neg * u_neg.HU + C(0.5) * cuda::physical_params.g * u_neg.H * u_neg.H,
		u_neg.HV*U_neg
	};

	FlowVector F_pos = {
		u_pos.HU,
		U_pos * u_pos.HU + C(0.5) * cuda::physical_params.g * u_pos.H * u_pos.H,
		u_pos.HV*U_pos
	};

	if (S_neg >= C(0.0))
	{
		return F_neg;
	}
	else if (S_neg < C(0.0) && S_pos >= C(0.0))
	{
		FlowVector F;

		F.H =
			(S_pos*F_neg.H - S_neg*F_pos.H + S_neg*S_pos*(u_pos.H-u_neg.H))
			/
			(S_pos - S_neg);
		F.HU =
			(S_pos*F_neg.HU - S_neg*F_pos.HU + S_neg*S_pos*(u_pos.HU-u_neg.HU))
			/
			(S_pos - S_neg);

		NUMERIC_TYPE S_mid =
			(S_neg*u_pos.H*(U_pos - S_pos) - S_pos*u_neg.H*(U_neg - S_neg))
			/
			(u_pos.H*(U_pos-S_pos) - u_neg.H*(U_neg - S_neg));

		if (S_neg < C(0.0) && S_mid >= C(0.0))
		{
			F.HV = F.H * V_neg;
		}
		else
		{
			F.HV = F.H * V_pos;
		}

		return F;
	}
	else
	{
		return F_pos;
	}
}

__device__ __forceinline__ FlowVector hll_y
(
	const FlowVector& u_neg,
	const FlowVector& u_pos
)
{
	FlowVector u_neg_rotated = { u_neg.H, u_neg.HV, -u_neg.HU };
	FlowVector u_pos_rotated = { u_pos.H, u_pos.HV, -u_pos.HU };

	FlowVector F_rotated = hll_x(u_neg_rotated, u_pos_rotated);

	return { F_rotated.H, -F_rotated.HV, F_rotated.HU };
}

__device__ __forceinline__ FlowVector star
(
	const FlowVector& U,
	NUMERIC_TYPE Z,
	NUMERIC_TYPE Zstar
)
{
	// FlowVector::calculate_Hstar
	NUMERIC_TYPE ETA = U.H + Z;
	NUMERIC_TYPE Hstar = FMAX(C(0.0), ETA - Zstar);

	// FlowVector::speed
	auto speed = [&](NUMERIC_TYPE discharge)
	{
		if (U.H >= cuda::solver_params.DepthThresh)
		{
			return discharge/U.H;
		}
		else
		{
			return C(0.0);
		}
	};

	return { Hstar, Hstar*speed(U.HU), Hstar*speed(U.HV) };
}

__device__ __forceinline__ FlowVector outside_x
(
	FlowVector U_const,
	FlowVector U_inside,
	int bc_i,
	int HU_sign,
	NUMERIC_TYPE Zstar
)
{
	FlowVector U_outside = { U_const.H, U_const.HU, U_const.HV }; 

	switch (cuda::boundaries.BC_type[bc_i])
	{
	case FREE1:
		// outflow only (swe/fv1.cpp): HU_sign is the inward normal, and
		// inward-pointing momentum is reflected, as at a wall
		if (HU_sign*U_const.HU > C(0.0)) U_outside.HU = -U_const.HU;
		break;
	case HFIX2:
	case HVAR3:
		U_outside.H = FMAX(C(0.0), cuda::boundaries.BC_value[bc_i] - Zstar);
		break;
	case QFIX4:
	case QVAR5:
		U_outside.HU = HU_sign * cuda::boundaries.BC_value[bc_i];
		U_outside.HV = C(0.0);
		if (FABS(U_outside.HU) > C(1e-6))
		{
			U_outside.H = FMAX(U_const.H,
					C(1.1)*cuda::solver_params.DepthThresh);
		}
		break;
	case NONE0:
	default:
		U_outside.HU = -U_const.HU;
		break;
	}

	return U_outside;
}

__device__ __forceinline__ FlowVector outside_y
(
	FlowVector U_const,
	FlowVector U_inside,
	int bc_i,
	int HU_sign,
	NUMERIC_TYPE Zstar
)
{
	FlowVector U_const_rotated = { U_const.H, U_const.HV, U_const.HU };
	FlowVector U_inside_rotated = { U_inside.H, U_inside.HV, U_inside.HU };
	FlowVector U_outside = outside_x(U_const_rotated,
			U_inside_rotated, bc_i, HU_sign, Zstar);

	return { U_outside.H, U_outside.HV, U_outside.HU };
}

__device__ __forceinline__ FlowVector inside_x
(
	FlowVector U_outside,
	FlowVector U_const,
	FlowVector U_inside,
	int bc_i
)
{
	switch (cuda::boundaries.BC_type[bc_i])
	{
	case FREE1:
		return U_inside;
	case HFIX2:
	case HVAR3:
		U_const.H = U_outside.H;
		break;
	case QFIX4:
	case QVAR5:
		return { U_outside.H, U_outside.HU, U_const.HV };
	}

	return U_const;
}

__device__ __forceinline__ FlowVector inside_y
(
	FlowVector U_outside,
	FlowVector U_const,
	FlowVector U_inside,
	int bc_i
)
{
	FlowVector U_outside_rotated = { U_outside.H, U_outside.HV, U_outside.HU };
	FlowVector U_const_rotated = { U_const.H, U_const.HV, U_const.HU };
	FlowVector U_inside_rotated = { U_inside.H, U_inside.HV, U_inside.HU };
	U_inside = inside_x(U_outside_rotated, U_const_rotated, U_inside_rotated, bc_i);
	return { U_inside.H, U_inside.HV, U_inside.HU };
}

__device__ __forceinline__ int index_w
(
	const int i,
	const int j
)
{
	return 2*cuda::geometry.xsz + 2*cuda::geometry.ysz - j;
}

__device__ __forceinline__ int index_e
(
	const int i,
	const int j
)
{
	return cuda::geometry.xsz + j-1;
}

__device__ __forceinline__ int index_n
(
	const int i,
	const int j
)
{
	return i-1;
}

__device__ __forceinline__ int index_s
(
	const int i,
	const int j
)
{
	return 2*cuda::geometry.xsz + cuda::geometry.ysz - i;
}

__device__ __forceinline__ void update_mass_stats_x
(
	MassStats* stats,
	NUMERIC_TYPE FH,
	int i,
	int j,
	NUMERIC_TYPE time_stage_fraction = C(1.0)
)
{
	if (i == 0)
	{
		if (FH > C(0.0))
		{
			atomicAdd(&(stats->in), time_stage_fraction*FH*cuda::geometry.dy);
		}
		else
		{
			atomicAdd(&(stats->out), -time_stage_fraction*FH*cuda::geometry.dy);
		}
	}
	else if (i == cuda::geometry.xsz)
	{
		if (FH < C(0.0))
		{
			atomicAdd(&(stats->in), -time_stage_fraction*FH*cuda::geometry.dy);
		}
		else
		{
			atomicAdd(&(stats->out), time_stage_fraction*FH*cuda::geometry.dy);
		}
	}
}

__device__ __forceinline__ void update_mass_stats_y
(
	MassStats* stats,
	NUMERIC_TYPE FH,
	int i,
	int j,
	NUMERIC_TYPE time_stage_fraction = C(1.0)
)
{
	if (j == 0)
	{
		if (FH < C(0.0))
		{
			atomicAdd(&(stats->in), -time_stage_fraction*FH*cuda::geometry.dx);
		}
		else
		{
			atomicAdd(&(stats->out), time_stage_fraction*FH*cuda::geometry.dx);
		}
	}
	else if (j == cuda::geometry.ysz)
	{
		if (FH > C(0.0))
		{
			atomicAdd(&(stats->in), time_stage_fraction*FH*cuda::geometry.dx);
		}
		else
		{
			atomicAdd(&(stats->out), -time_stage_fraction*FH*cuda::geometry.dx);
		}
	}
}

__device__ __forceinline__ void add_block_sum(double v, double* total)
{
	__shared__ double partial[CUDA_BLOCK_SIZE];
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	partial[tid] = v;
	__syncthreads();
	for (int s = (blockDim.x*blockDim.y)/2; s > 0; s >>= 1)
	{
		if (tid < s) partial[tid] += partial[tid + s];
		__syncthreads();
	}
	if (tid == 0 && partial[0] > 0.0) atomicAdd(total, partial[0]);
}

__device__ __forceinline__ NUMERIC_TYPE depth_after_loss
(
	NUMERIC_TYPE h,
	NUMERIC_TYPE rate,
	NUMERIC_TYPE& residual,
	NUMERIC_TYPE step_dt, NUMERIC_TYPE threshold
)
{
	if (h <= threshold || rate <= C(0.0))
	{
		residual = C(0.0);
		return h;
	}
	NUMERIC_TYPE wanted = rate*step_dt + residual;
	NUMERIC_TYPE h_new = h - wanted;
	if (h_new <= C(0.0))
	{
		residual = C(0.0);
		return C(0.0);
	}
	residual = wanted - (h - h_new); // h - h_new is exact (Sterbenz)
	return h_new;
}

__device__ __forceinline__ double horton_capacity(double F, double f0, double fc, double k, double dt)
{
	if (k <= 0.0 || f0 <= fc) return fmax(f0, fc)*dt;
	const double A = (f0 - fc)/k;
	if (fc <= 0.0 && F >= A) return 0.0;
	double te = 0.0;
	if (F > 0.0)
	{
		te = F/f0;
		for (int n=0; n<50; n++)
		{
			double e = exp(-k*te);
			double step = (F - (fc*te + A*(1.0 - e)))/(fc + (f0 - fc)*e);
			te += step;
			if (fabs(step) <= 1e-9*(te + 1.0)) break;
		}
	}
	return fc*dt - A*exp(-k*te)*expm1(-k*dt);
}

__device__ __forceinline__ NUMERIC_TYPE depth_after_greenampt
(
	const LossParams& p,
	int idx,
	NUMERIC_TYPE h, NUMERIC_TYPE step_dt
)
{
	NUMERIC_TYPE& residual = p.residual[idx];
	if (h <= C(0.0))
	{
		residual = C(0.0);
		return h;
	}

	double dt = step_dt;
	double F = p.F[idx];
	double ks = p.ks[idx];
	double dF = 0.0;
	if (p.horton)
	{
		// ks = fc, psi = f0, dtheta = k (see LossParams)
		dF = horton_capacity(F, (double)p.psi[idx], ks, (double)p.dtheta[idx], dt);
	}
	else if (ks > 0.0)
	{
		// implicit Euler: F_new = F + dt Ks (1 + (psi + h) dtheta / F_new)
		double M = ((double)p.psi[idx] + (double)h) * (double)p.dtheta[idx];
		double a = F + ks*dt;
		dF = 0.5*(a + sqrt(a*a + 4.0*ks*dt*M)) - F;
		if (p.fmax != nullptr)
		{
			dF = fmin(dF, fmax((double)p.fmax[idx] - F, 0.0));
		}
	}
	double e = (double)p.evap_rate * dt;
	if (dF + e <= 0.0)
	{
		residual = C(0.0);
		return h;
	}

	NUMERIC_TYPE wanted = (NUMERIC_TYPE)(dF + e) + residual;
	NUMERIC_TYPE h_new = h - wanted;
	if (h_new <= C(0.0))
	{
		// not enough water: infiltration takes it first, evaporation the rest
		p.F[idx] = F + fmin(dF, (double)h);
		residual = C(0.0);
		return C(0.0);
	}
	residual = wanted - (h - h_new); // h - h_new is exact (Sterbenz)
	p.F[idx] = F + dF;
	return h_new;
}

__device__ __forceinline__ NUMERIC_TYPE depth_after_losses
(
	const LossParams& p,
	int idx,
	NUMERIC_TYPE h, NUMERIC_TYPE step_dt, NUMERIC_TYPE threshold
)
{
	if (p.greenampt || p.horton) return depth_after_greenampt(p, idx, h, step_dt);
	NUMERIC_TYPE rate = ((p.rate_grid != nullptr) ? p.rate_grid[idx] : p.uniform_rate)
		+ p.evap_rate;
	return depth_after_loss(h, rate, p.residual[idx], step_dt, threshold);
}

__device__ __forceinline__ void fv1_cell_values
(
	NUMERIC_TYPE& h,
	NUMERIC_TYPE& hu,
	NUMERIC_TYPE& hv,
	int idx,
	bool nodata,
	bool drain,
	bool losses,
	MassStats* mass_stats,
	const LossParams& p,
	double& removed
)
{
	if (nodata)
	{
		if (drain)
		{
			// as drain_cell
			NUMERIC_TYPE h0 = h;
			h = C(0.0);
			if (h0 == C(0.0)) return;
			NUMERIC_TYPE Q = -h0 * cuda::geometry.dx * cuda::geometry.dy / cuda::dt;
			if (Q > C(0.0)) atomicAdd(&(mass_stats->in), Q);
			else atomicAdd(&(mass_stats->out), -Q);
		}
	}
	else if (losses)
	{
		// as apply_losses_fv1
		NUMERIC_TYPE h_new = depth_after_losses(p, idx, h, cuda::dt, cuda::solver_params.DepthThresh);
		if (h_new != h)
		{
			if (h_new > cuda::solver_params.DepthThresh)
			{
				NUMERIC_TYPE scale = h_new/h;
				hu *= scale;
				hv *= scale;
			}
			else
			{
				hu = C(0.0);
				hv = C(0.0);
			}
			removed += (double)(h - h_new);
			h = h_new;
		}
	}
}

__device__ __forceinline__ NUMERIC_TYPE rate_at_cell
(
	NUMERIC_TYPE* rain_data,
	int i,
	int j
)
{
    int tile_i = (i-1) * cuda::geometry.dx / cuda::rain_geometry.dx;

    NUMERIC_TYPE top_gap = cuda::geometry.tly - cuda::rain_geometry.tly;
    int tile_j = (j-1 - top_gap/cuda::geometry.dy) *
		cuda::geometry.dy / cuda::rain_geometry.dy;

    if (tile_i < cuda::rain_geometry.xsz && tile_j >= 0)
    {
        return rain_data[tile_j*cuda::rain_geometry.xsz + tile_i];
    }
    else
    {
        return C(0.0);
    }
}

}

}
}
