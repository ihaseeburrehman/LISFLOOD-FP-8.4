//
// LISFLOOD-FP Metal DG2 Solver Kernels
// The FV1 and FV2 kernels this back end shares (HLL fluxes, boundary states,
// rain, max field), plus the DG2 kernels, which follow cuda/dg2 line by line
// (friction at the Gauss points, slope limiter, ghost cells, the two
// Runge-Kutta stages with the slope equations and the bed source)
//

#include <metal_stdlib>
using namespace metal;

// Constants matching LISFLOOD-FP definitions
#define DEPTH_THRESH 1e-3f
#define MIN_SAFE_H 1e-6f  // Minimum depth for safe division (1 micron)

// Portable atomic add for floats (works on any Metal version that supports atomic_uint)
inline void atomic_add_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float f_expected = as_type<float>(expected);
        float f_new = f_expected + val;
        uint desired = as_type<uint>(f_new);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline void account_boundary(device atomic_uint* totals, float outward_volume) {
    if (outward_volume > 0.0f) atomic_add_float(&totals[0], outward_volume);
    else if (outward_volume < 0.0f) atomic_add_float(&totals[1], -outward_volume);
}

// Boundary types matching ESourceType in lisflood.h
enum ENUM_BC_TYPE {
    BC_NONE = 0,   // closed / reflective
    BC_FREE = 1,   // transmissive
    BC_HFIX = 2,   // fixed water surface elevation
    BC_HVAR = 3,   // time-varying water surface elevation
    BC_QFIX = 4,   // fixed unit discharge (m2/s)
    BC_QVAR = 5,   // time-varying unit discharge (m2/s)
    BC_FREE6 = 6   // transmissive (alternate code)
};

// A boundary state split into the component normal to the edge and the one
// tangential to it, so the same routine serves both sweeps (the CPU does the
// same thing by swapping HU/HV in the argument list for the y boundaries).
struct EdgeState {
    float H;
    float Qn;  // momentum normal to the boundary
    float Qt;  // momentum tangential to the boundary
};

// Mirrors fv1::set_boundary_values() in swe/fv1.cpp.
//
// `in` is the interior cell's reconstructed state at the edge and `out` the
// state outside it; both may be modified, because the prescribed-discharge
// types also constrain the inside (the CPU does exactly this). `qn_sign` is +1
// where a positive prescribed discharge drives flow along this axis' positive
// direction and -1 where it opposes it, so that a positive BC value always
// means inflow on every edge.
inline void set_boundary_values(int type, float value, float Z,
                                float depth_thresh, float qn_sign,
                                thread EdgeState& in, thread EdgeState& out)
{
    if (type == BC_HFIX || type == BC_HVAR) {
        out.H = max(0.0f, value - Z);
        out.Qn = in.Qn;
        out.Qt = in.Qt;
    } else if (type == BC_QFIX || type == BC_QVAR) {
        out.Qn = qn_sign * value;
        if (abs(out.Qn) > 1e-12f) {
            in.H = max(in.H, 1.1f * depth_thresh);
            out.H = in.H;
        } else {
            out.H = in.H;
        }
        in.Qn = out.Qn;
        out.Qt = in.Qt;
    } else if (type == BC_FREE || type == BC_FREE6) {
        // outflow only (swe/fv1.cpp): qn_sign is the inward normal, and
        // inward-pointing momentum is reflected, as at a wall
        out.H = in.H;
        out.Qn = (qn_sign * in.Qn > 0.0f) ? -in.Qn : in.Qn;
        out.Qt = in.Qt;
    } else { // BC_NONE: closed wall
        out.H = in.H;
        out.Qn = -in.Qn;
        out.Qt = in.Qt;
    }
}

// Geometry parameters (set from host)
struct Geometry {
    int xsz;
    int ysz;
    int pitch;
    float dx;
    float dy;
    float tly;
    float blx;
    float bly;
    int row_begin;        // first owned row (1 for single-GPU)
    int row_end;          // one past last owned row (ysz+1 for single-GPU)
    int owns_north_ghost; // 1 if rank 0
    int owns_south_ghost; // 1 if last rank
};

// Physical parameters
struct PhysicalParams {
    float g;           // gravity
    float manning;     // Manning's n
    float speed_thresh;
};

// Solver parameters
struct SolverParams {
    float dt;
    float depth_thresh;
};

// Flow vector structure
struct FlowVector {
    float H;   // water height
    float HU;  // x-momentum
    float HV;  // y-momentum
};

// HLL Riemann solver in x-direction
inline FlowVector hll_x(FlowVector U_neg, FlowVector U_pos, float g, float depth_thresh) {
    float H_neg = max(U_neg.H, 0.0f);
    float H_pos = max(U_pos.H, 0.0f);
    
    if (H_neg <= depth_thresh && H_pos <= depth_thresh) {
        return FlowVector{0.0f, 0.0f, 0.0f};
    }

    float u_neg = (H_neg > depth_thresh) ? U_neg.HU / max(H_neg, 1e-6f) : 0.0f;
    float v_neg = (H_neg > depth_thresh) ? U_neg.HV / max(H_neg, 1e-6f) : 0.0f;
    float u_pos = (H_pos > depth_thresh) ? U_pos.HU / max(H_pos, 1e-6f) : 0.0f;
    float v_pos = (H_pos > depth_thresh) ? U_pos.HV / max(H_pos, 1e-6f) : 0.0f;
    
    float a_neg = sqrt(g * H_neg);
    float a_pos = sqrt(g * H_pos);
    
    // Toro estimate for H_star, U_star (matching hll_include.h)
    float h_star = pow(0.5f*(a_neg + a_pos) + 0.25f*(u_neg - u_pos), 2.0f) / g;
    float u_star = 0.5f*(u_neg + u_pos) + a_neg - a_pos;
    float a_star = sqrt(g * h_star);

    float s_neg, s_pos;
    if (H_neg <= depth_thresh) {
        s_neg = u_pos - 2.0f*a_pos;
    } else {
        s_neg = min(u_neg - a_neg, u_star - a_star);
    }

    if (H_pos <= depth_thresh) {
        s_pos = u_neg + 2.0f*a_neg;
    } else {
        s_pos = max(u_pos + a_pos, u_star + a_star);
    }
    
    s_neg = min(s_neg, 0.0f);
    s_pos = max(s_pos, 0.0f);
    
    FlowVector F_neg = {
        U_neg.HU,
        u_neg*U_neg.HU + 0.5f * g * H_neg * H_neg,
        U_neg.HV * u_neg
    };
    
    FlowVector F_pos = {
        U_pos.HU,
        u_pos*U_pos.HU + 0.5f * g * H_pos * H_pos,
        U_pos.HV * u_pos
    };
    
    float denom = s_pos - s_neg;
    if (abs(denom) < 1e-10f) return FlowVector{0.0f, 0.0f, 0.0f};
    
    FlowVector F;
    if (s_neg >= 0.0f) {
        F = F_neg;
    } else if (s_pos <= 0.0f) {
        F = F_pos;
    } else {
        F.H = (s_pos * F_neg.H - s_neg * F_pos.H + s_pos * s_neg * (U_pos.H - U_neg.H)) / denom;
        F.HU = (s_pos * F_neg.HU - s_neg * F_pos.HU + s_pos * s_neg * (U_pos.HU - U_neg.HU)) / denom;
        
        // Transverse momentum flux (HV) using S_mid logic from hll_include.h
        float s_mid = (s_neg * H_pos * (u_pos - s_pos) - s_pos * H_neg * (u_neg - s_neg)) 
                    / (H_pos * (u_pos - s_pos) - H_neg * (u_neg - s_neg));
        
        if (s_mid >= 0.0f) {
            F.HV = F.H * v_neg;
        } else {
            F.HV = F.H * v_pos;
        }
    }
    
    return F;
}

// HLL Riemann solver in y-direction
inline FlowVector hll_y(FlowVector U_neg, FlowVector U_pos, float g, float depth_thresh) {
    float H_neg = max(U_neg.H, 0.0f);
    float H_pos = max(U_pos.H, 0.0f);
    
    if (H_neg <= depth_thresh && H_pos <= depth_thresh) {
        return FlowVector{0.0f, 0.0f, 0.0f};
    }

    float u_neg = (H_neg > depth_thresh) ? U_neg.HU / max(H_neg, 1e-6f) : 0.0f;
    float v_neg = (H_neg > depth_thresh) ? U_neg.HV / max(H_neg, 1e-6f) : 0.0f;
    float u_pos = (H_pos > depth_thresh) ? U_pos.HU / max(H_pos, 1e-6f) : 0.0f;
    float v_pos = (H_pos > depth_thresh) ? U_pos.HV / max(H_pos, 1e-6f) : 0.0f;
    
    float a_neg = sqrt(g * H_neg);
    float a_pos = sqrt(g * H_pos);
    
    // Toro estimate for H_star, V_star (matching hll_include.h with V as primary)
    float h_star = pow(0.5f*(a_neg + a_pos) + 0.25f*(v_neg - v_pos), 2.0f) / g;
    float v_star = 0.5f*(v_neg + v_pos) + a_neg - a_pos;
    float a_star = sqrt(g * h_star);

    float s_neg, s_pos;
    if (H_neg <= depth_thresh) {
        s_neg = v_pos - 2.0f*a_pos;
    } else {
        s_neg = min(v_neg - a_neg, v_star - a_star);
    }

    if (H_pos <= depth_thresh) {
        s_pos = v_neg + 2.0f*a_neg;
    } else {
        s_pos = max(v_pos + a_pos, v_star + a_star);
    }
    
    s_neg = min(s_neg, 0.0f);
    s_pos = max(s_pos, 0.0f);
    
    FlowVector F_neg = {
        U_neg.HV,
        U_neg.HU * v_neg,
        v_neg * U_neg.HV + 0.5f * g * H_neg * H_neg
    };
    
    FlowVector F_pos = {
        U_pos.HV,
        U_pos.HU * v_pos,
        v_pos * U_pos.HV + 0.5f * g * H_pos * H_pos
    };
    
    float denom = s_pos - s_neg;
    if (abs(denom) < 1e-10f) return FlowVector{0.0f, 0.0f, 0.0f};
    
    FlowVector F;
    if (s_neg >= 0.0f) {
        F = F_neg;
    } else if (s_pos <= 0.0f) {
        F = F_pos;
    } else {
        F.H = (s_pos * F_neg.H - s_neg * F_pos.H + s_pos * s_neg * (U_pos.H - U_neg.H)) / denom;
        F.HV = (s_pos * F_neg.HV - s_neg * F_pos.HV + s_pos * s_neg * (U_pos.HV - U_neg.HV)) / denom;
        
        // Transverse momentum flux (HU in y-direction) using S_mid logic
        float s_mid = (s_neg * H_pos * (v_pos - s_pos) - s_pos * H_neg * (v_neg - s_neg)) 
                    / (H_pos * (v_pos - s_pos) - H_neg * (v_neg - s_neg));
        
        if (s_mid >= 0.0f) {
            F.HU = F.H * u_neg;
        } else {
            F.HU = F.H * u_pos;
        }
    }
    
    return F;
}

// Bed source term for x-direction
inline float bed_source_x(float Zstar_w, float Zstar_e, float Hstar_w, float Hstar_e, float ETA, float g, float dx) {
    float Zdagger_w = Zstar_w - max(0.0f, -(ETA - Zstar_w));
    float Zdagger_e = Zstar_e - max(0.0f, -(ETA - Zstar_e));
    return -g * 0.5f * (Hstar_w + Hstar_e) * (Zdagger_e - Zdagger_w) / dx;
}

// Bed source term for y-direction
inline float bed_source_y(float Zstar_n, float Zstar_s, float Hstar_n, float Hstar_s, float ETA, float g, float dy) {
    float Zdagger_n = Zstar_n - max(0.0f, -(ETA - Zstar_n));
    float Zdagger_s = Zstar_s - max(0.0f, -(ETA - Zstar_s));
    return -g * 0.5f * (Hstar_n + Hstar_s) * (Zdagger_s - Zdagger_n) / dy;
}

// Apply friction (Manning's equation)
kernel void apply_friction(
    device float* H [[buffer(0)]],
    device float* HU [[buffer(1)]],
    device float* HV [[buffer(2)]],
    constant Geometry& geom [[buffer(3)]],
    constant PhysicalParams& params [[buffer(4)]],
    constant SolverParams& solver [[buffer(5)]],
    device const float* manning [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x + 1;
    int j = gid.y + 1;
    
    if (i > geom.xsz || j > geom.ysz) return;
    
    int idx = j * geom.pitch + i;
    float h = H[idx];
    
    if (h > solver.depth_thresh) {
        float hu = HU[idx];
        float hv = HV[idx];
        
        // Use safe depth for division to prevent numerical instability
        float h_safe = max(h, MIN_SAFE_H);
        
        float n = (manning != nullptr) ? manning[idx] : params.manning;
        float u = hu / h_safe;
        float v = hv / h_safe;
        float speed = sqrt(u * u + v * v);
        float Cf = params.g * n * n / pow(h_safe, 0.33333333f);

        // Speed threshold - zero out momentum for very slow flow (matches fv1.cpp)
        if (abs(u) <= params.speed_thresh && abs(v) <= params.speed_thresh) {
            HU[idx] = 0.0f;
            HV[idx] = 0.0f;
            return;
        }
        
        // CPU semi-implicit friction (fv1.cpp lines 117-128)
        // The CPU uses: HU_new = HU_old + dt * Sf / D
        // where Sf = -Cf * u * speed and D = 1 + dt*Cf/h * directional_term
        float speed_safe = speed + 1e-12f; // Very small epsilon
        float denom_x = 1.0f + solver.dt * Cf / h_safe * (2.0f * u * u + v * v) / speed_safe;
        float denom_y = 1.0f + solver.dt * Cf / h_safe * (u * u + 2.0f * v * v) / speed_safe;
        
        float Sf_x = -Cf * u * speed;
        float Sf_y = -Cf * v * speed;
        
        HU[idx] = hu + solver.dt * Sf_x / denom_x;
        HV[idx] = hv + solver.dt * Sf_y / denom_y;
        
    } else {
        HU[idx] = 0.0f;
        HV[idx] = 0.0f;
    }
}

// Compute Z* at cell interfaces in x-direction
kernel void compute_zstar_x(
    device const float* DEM [[buffer(0)]],
    device float* Zstar_x [[buffer(1)]],
    constant Geometry& geom [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x;
    int j = gid.y + 1;
    
    if (i > geom.xsz || j > geom.ysz) return;
    
    int idx = j * geom.pitch + i;
    float Z_neg = DEM[idx];
    float Z_pos = DEM[idx + 1];
    
    Zstar_x[idx] = max(Z_neg, Z_pos);
}

// Compute Z* at cell interfaces in y-direction
kernel void compute_zstar_y(
    device const float* DEM [[buffer(0)]],
    device float* Zstar_y [[buffer(1)]],
    constant Geometry& geom [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x + 1;
    int j = gid.y;
    
    if (i > geom.xsz || j > geom.ysz) return;
    
    int idx = j * geom.pitch + i;
    float Z_neg = DEM[idx];         // North cell (row j)
    float Z_pos = DEM[(j+1) * geom.pitch + i]; // South cell (row j+1)
    
    Zstar_y[idx] = max(Z_neg, Z_pos);
}

// Threadgroup tile for the flux-sharing sweeps. Must match the host dispatch.
#define TG_X 16
#define TG_Y 16

// Reconstruct a cell's state at one interface (the "star" state).
inline FlowVector star_state(FlowVector U, float Z, float Zstar, float depth_thresh) {
    float u = (U.H > depth_thresh) ? U.HU / max(U.H, 1e-6f) : 0.0f;
    float v = (U.H > depth_thresh) ? U.HV / max(U.H, 1e-6f) : 0.0f;
    FlowVector S;
    S.H = max(0.0f, U.H + Z - Zstar);
    S.HU = S.H * u;
    S.HV = S.H * v;
    return S;
}

// Update flow variables in x-direction.
//
// Each thread evaluates exactly ONE interface flux (the interface between cell
// i and cell i+1) and publishes it to threadgroup memory; cell i then reads its
// west flux back as the *same bits* its neighbour computed. This mirrors
// cuda/fv1/cuda_fv1_solver.cu, and it is what makes the scheme conservative:
// evaluating the same interface independently in both adjacent cells is only
// algebraically equal, not bitwise equal, so the water leaving one cell did not
// match the water entering the next and the domain leaked mass.
// Threadgroups overlap by one column in x so every interface is covered;
// threads with tid.x == 0 contribute a flux but update no cell.
kernel void update_flow_x(
    device const float* H_old [[buffer(0)]],
    device const float* HU_old [[buffer(1)]],
    device const float* HV_old [[buffer(2)]],
    device float* H_new [[buffer(3)]],
    device float* HU_new [[buffer(4)]],
    device float* HV_new [[buffer(5)]],
    device const float* DEM [[buffer(6)]],
    device const float* Zstar_x [[buffer(7)]],
    constant Geometry& geom [[buffer(8)]],
    constant PhysicalParams& params [[buffer(9)]],
    constant SolverParams& solver [[buffer(10)]],
    device const int* boundary_types [[buffer(11)]],
    device const float* boundary_values [[buffer(12)]],
    device atomic_uint* mass_totals [[buffer(13)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup FlowVector F[TG_Y][TG_X];
    threadgroup float Hstar_sh[TG_Y][TG_X];

    int i = (int)(tgid.x * (TG_X - 1) + tid.x);   // interface / cell index 0..xsz
    int j = (int)(tgid.y * TG_Y + tid.y) + 1;     // row 1..ysz

    bool active = (i <= geom.xsz && j <= geom.ysz);

    FlowVector U_c, Ustar_c_e;
    float Z_c = 0.0f, Zstar_e = 0.0f;
    FlowVector F_e = FlowVector{0.0f, 0.0f, 0.0f};

    if (active) {
        int idx = j * geom.pitch + i;
        Zstar_e = Zstar_x[idx];

        Z_c = DEM[idx];
        U_c = FlowVector{H_old[idx], HU_old[idx], HV_old[idx]};
        Ustar_c_e = star_state(U_c, Z_c, Zstar_e, solver.depth_thresh);

        float Z_e = DEM[idx + 1];
        FlowVector U_e = FlowVector{H_old[idx + 1], HU_old[idx + 1], HV_old[idx + 1]};
        FlowVector Ustar_e = star_state(U_e, Z_e, Zstar_e, solver.depth_thresh);

        // Domain edges: replace the ghost-cell state with the prescribed
        // boundary condition. At i == 0 the outside state is Ustar_c_e (the
        // west ghost) and the interior is Ustar_e; at i == xsz it is the other
        // way round.
        if (i == 0 || i == geom.xsz) {
            bool west = (i == 0);
            int bc_idx = west ? (2 * geom.xsz + 2 * geom.ysz - (j - 1) - 1)
                              : (geom.xsz + (j - 1));
            int type = boundary_types[bc_idx];
            float value = boundary_values[bc_idx];
            float Z_in = west ? Z_e : Z_c;   // interior cell's bed elevation

            EdgeState in_s, out_s;
            if (west) {
                in_s = EdgeState{Ustar_e.H, Ustar_e.HU, Ustar_e.HV};
                out_s = EdgeState{Ustar_c_e.H, Ustar_c_e.HU, Ustar_c_e.HV};
            } else {
                in_s = EdgeState{Ustar_c_e.H, Ustar_c_e.HU, Ustar_c_e.HV};
                out_s = EdgeState{Ustar_e.H, Ustar_e.HU, Ustar_e.HV};
            }
            set_boundary_values(type, value, Z_in, solver.depth_thresh,
                                west ? 1.0f : -1.0f, in_s, out_s);
            if (west) {
                Ustar_e = FlowVector{in_s.H, in_s.Qn, in_s.Qt};
                Ustar_c_e = FlowVector{out_s.H, out_s.Qn, out_s.Qt};
            } else {
                Ustar_c_e = FlowVector{in_s.H, in_s.Qn, in_s.Qt};
                Ustar_e = FlowVector{out_s.H, out_s.Qn, out_s.Qt};
            }
        }

        F_e = hll_x(Ustar_c_e, Ustar_e, params.g, solver.depth_thresh);

        F[tid.y][tid.x] = F_e;
        Hstar_sh[tid.y][tid.x] = Ustar_e.H;

        // Domain-edge bookkeeping: i == 0 is the west face, i == xsz the east.
        if (i == 0) account_boundary(mass_totals, -F_e.H * solver.dt * geom.dy);
        if (i == geom.xsz) account_boundary(mass_totals, F_e.H * solver.dt * geom.dy);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.x == 0 || !active || i < 1) return;

    int idx = j * geom.pitch + i;
    FlowVector F_w = F[tid.y][tid.x - 1];
    float Hstar_w = Hstar_sh[tid.y][tid.x - 1];
    float Zstar_w = Zstar_x[idx - 1];

    float ETA = U_c.H + Z_c;
    float bed_src = bed_source_x(Zstar_w, Zstar_e, Hstar_w, Ustar_c_e.H, ETA,
                                 params.g, geom.dx);

    // Partial (x-only) contribution of the unsplit FV1 update. No clamping and
    // no momentum zeroing here: update_flow_y still has to add the y-direction
    // divergence to this intermediate state.
    H_new[idx] = U_c.H - solver.dt * (F_e.H - F_w.H) / geom.dx;
    HU_new[idx] = U_c.HU - solver.dt * ((F_e.HU - F_w.HU) / geom.dx - bed_src);
    HV_new[idx] = U_c.HV - solver.dt * (F_e.HV - F_w.HV) / geom.dx;
}

// Update flow variables in y-direction (same flux-sharing scheme, overlapping
// in y). Metal's y axis points south (+j), so a thread at row j evaluates the
// interface between rows j and j+1 -- its own south face -- and cell j reads its
// north flux back from the thread above it.
kernel void update_flow_y(
    device const float* H_int [[buffer(0)]],
    device const float* HU_int [[buffer(1)]],
    device const float* HV_int [[buffer(2)]],
    device const float* H_old [[buffer(3)]],
    device const float* HU_old [[buffer(4)]],
    device const float* HV_old [[buffer(5)]],
    device float* H_new [[buffer(6)]],
    device float* HU_new [[buffer(7)]],
    device float* HV_new [[buffer(8)]],
    device const float* DEM [[buffer(9)]],
    device const float* Zstar_y [[buffer(10)]],
    constant Geometry& geom [[buffer(11)]],
    constant PhysicalParams& params [[buffer(12)]],
    constant SolverParams& solver [[buffer(13)]],
    device const int* boundary_types [[buffer(14)]],
    device const float* boundary_values [[buffer(15)]],
    device atomic_uint* mass_totals [[buffer(16)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup FlowVector F[TG_Y][TG_X];
    threadgroup float Hstar_sh[TG_Y][TG_X];

    int i = (int)(tgid.x * TG_X + tid.x) + 1;     // column 1..xsz
    int j = (int)(tgid.y * (TG_Y - 1) + tid.y);   // interface / cell row 0..ysz

    bool active = (i <= geom.xsz && j <= geom.ysz);

    FlowVector U_c, Ustar_c_s;
    float Z_c = 0.0f, Zstar_s = 0.0f;
    FlowVector F_s = FlowVector{0.0f, 0.0f, 0.0f};

    if (active) {
        int idx = j * geom.pitch + i;
        Zstar_s = Zstar_y[idx];

        Z_c = DEM[idx];
        U_c = FlowVector{H_old[idx], HU_old[idx], HV_old[idx]};
        Ustar_c_s = star_state(U_c, Z_c, Zstar_s, solver.depth_thresh);

        int idx_s = (j + 1) * geom.pitch + i;
        float Z_s = DEM[idx_s];
        FlowVector U_s = FlowVector{H_old[idx_s], HU_old[idx_s], HV_old[idx_s]};
        FlowVector Ustar_s = star_state(U_s, Z_s, Zstar_s, solver.depth_thresh);

        // Domain edges. Metal's y axis points south, so a positive prescribed
        // discharge is +Qn on the north edge and -Qn on the south edge; that
        // keeps "positive value == inflow" on every side, as on the CPU.
        if (j == 0 || j == geom.ysz) {
            bool north = (j == 0);
            int bc_idx = north ? (i - 1)
                               : (2 * geom.xsz + geom.ysz - (i - 1) - 1);
            int type = boundary_types[bc_idx];
            float value = boundary_values[bc_idx];
            float Z_in = north ? Z_s : Z_c;

            EdgeState in_s, out_s;
            if (north) {
                in_s = EdgeState{Ustar_s.H, Ustar_s.HV, Ustar_s.HU};
                out_s = EdgeState{Ustar_c_s.H, Ustar_c_s.HV, Ustar_c_s.HU};
            } else {
                in_s = EdgeState{Ustar_c_s.H, Ustar_c_s.HV, Ustar_c_s.HU};
                out_s = EdgeState{Ustar_s.H, Ustar_s.HV, Ustar_s.HU};
            }
            set_boundary_values(type, value, Z_in, solver.depth_thresh,
                                north ? 1.0f : -1.0f, in_s, out_s);
            if (north) {
                Ustar_s = FlowVector{in_s.H, in_s.Qt, in_s.Qn};
                Ustar_c_s = FlowVector{out_s.H, out_s.Qt, out_s.Qn};
            } else {
                Ustar_c_s = FlowVector{in_s.H, in_s.Qt, in_s.Qn};
                Ustar_s = FlowVector{out_s.H, out_s.Qt, out_s.Qn};
            }
        }

        F_s = hll_y(Ustar_c_s, Ustar_s, params.g, solver.depth_thresh);

        F[tid.y][tid.x] = F_s;
        Hstar_sh[tid.y][tid.x] = Ustar_s.H;

        if (j == 0) account_boundary(mass_totals, -F_s.H * solver.dt * geom.dx);
        if (j == geom.ysz) account_boundary(mass_totals, F_s.H * solver.dt * geom.dx);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.y == 0 || !active || j < 1) return;

    int idx = j * geom.pitch + i;
    FlowVector F_n = F[tid.y - 1][tid.x];
    float Hstar_n = Hstar_sh[tid.y - 1][tid.x];
    float Zstar_n = Zstar_y[(j - 1) * geom.pitch + i];

    float ETA = U_c.H + Z_c;
    float bed_src = bed_source_y(Zstar_n, Zstar_s, Hstar_n, Ustar_c_s.H, ETA,
                                 params.g, geom.dy);

    float h_next = max(0.0f, H_int[idx] - solver.dt * (F_s.H - F_n.H) / geom.dy);
    float hu_next = HU_int[idx] - solver.dt * (F_s.HU - F_n.HU) / geom.dy;
    float hv_next = HV_int[idx] - solver.dt * ((F_s.HV - F_n.HV) / geom.dy - bed_src);

    // No "h <= depth_thresh => zero momentum" here: the CPU reference drops the
    // momentum of shallow cells in apply_friction, which runs after rainfall and
    // the point sources have topped the cell up.
    H_new[idx] = h_next;
    HU_new[idx] = hu_next;
    HV_new[idx] = hv_next;
}

// Compute CFL timestep per cell (for adaptive timestepping)
kernel void compute_dt_per_cell(
    device const float* H [[buffer(0)]],
    device const float* HU [[buffer(1)]],
    device const float* HV [[buffer(2)]],
    device float* dt_field [[buffer(3)]],
    constant Geometry& geom [[buffer(4)]],
    constant PhysicalParams& params [[buffer(5)]],
    constant float& cfl [[buffer(6)]],
    constant float& max_dt [[buffer(7)]],
    volatile device atomic_uint* min_dt_atomic [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x + 1;
    int j = gid.y + 1;
    
    if (i > geom.xsz || j > geom.ysz) return;
    
    int idx = j * geom.pitch + i;
    float h = H[idx];
    
    if (h > DEPTH_THRESH) {
        float hu = HU[idx];
        float hv = HV[idx];
        
        // Use safe depth for division to prevent numerical instability
        float h_safe = max(h, MIN_SAFE_H);
        float u = hu / h_safe;
        float v = hv / h_safe;
        float c = sqrt(params.g * h);
        
        float dt_x = geom.dx / (abs(u) + c);
        float dt_y = geom.dy / (abs(v) + c);
        
        dt_field[idx] = cfl * min(dt_x, dt_y);
    } else {
        dt_field[idx] = max_dt;
    }
    
    // Global reduction using SIMD and Atomics
    float local_dt = dt_field[idx];
    
    // SIMD reduction within warp/subgroup to reduce contention
    // Make sure to include metal_simdgroup if needed, but it's standard
    local_dt = simd_min(local_dt);
    
    if (simd_is_first()) {
        uint dt_uint = as_type<uint>(local_dt);
        atomic_fetch_min_explicit(min_dt_atomic, dt_uint, memory_order_relaxed);
    }
}

// Zero ghost cells (boundary)
kernel void zero_ghost_cells(
    device float* H [[buffer(0)]],
    device float* HU [[buffer(1)]],
    device float* HV [[buffer(2)]],
    constant Geometry& geom [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // North and South boundaries
    if (gid < (uint)geom.pitch) {
        int i = gid;
        // North (j=0)
        H[i] = 0.0f;
        HU[i] = 0.0f;
        HV[i] = 0.0f;
        // South (j=ysz+1)
        int idx_s = (geom.ysz + 1) * geom.pitch + i;
        H[idx_s] = 0.0f;
        HU[idx_s] = 0.0f;
        HV[idx_s] = 0.0f;
    }
    
    // East and West boundaries
    if (gid < (uint)(geom.ysz + 2)) {
        int j = gid;
        // West (i=0)
        int idx_w = j * geom.pitch;
        H[idx_w] = 0.0f;
        HU[idx_w] = 0.0f;
        HV[idx_w] = 0.0f;
        // East (i=xsz+1)
        int idx_e = j * geom.pitch + geom.xsz + 1;
        H[idx_e] = 0.0f;
        HU[idx_e] = 0.0f;
        HV[idx_e] = 0.0f;
    }
}

// Apply uniform rainfall
kernel void apply_uniform_rain(
    device float* H [[buffer(0)]],
    device const float* DEM [[buffer(1)]],
    constant Geometry& geom [[buffer(2)]],
    constant float& rain_rate [[buffer(3)]],
    constant float& dt [[buffer(4)]],
    constant float& nodata [[buffer(5)]],
    device atomic_uint* mass_totals [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x + 1;
    int j = gid.y + 1;
    
    if (i > geom.xsz || j > geom.ysz) return;
    
    int idx = j * geom.pitch + i;
    float z = DEM[idx];
    
    // Don't add rain to nodata cells
    if (abs(z - nodata) > 1e-6f) {
        float increment = max(-H[idx], rain_rate * dt);
        H[idx] += increment;
        // Slot 3 is signed uniform-rain volume; negative rainfall is limited
        // by the water present, so accounting must use the applied increment.
        if (increment != 0.0f)
            atomic_add_float(&mass_totals[3], increment * geom.dx * geom.dy);
    }
}

// Helper for apply_rain (Dynamic)
float rate_at_cell(device const float* rain_data, 
                   int i, int j, 
                   constant Geometry& dem_geom, 
                   constant Geometry& rain_geom) 
{
    // i, j are 0-based DEM cell indices (inner domain)
    
    float x_offset = dem_geom.blx - rain_geom.blx;
    float tile_i_f = (x_offset + (float)i * dem_geom.dx) / rain_geom.dx;
    int tile_i = (int)floor(tile_i_f);
    
    float top_gap = dem_geom.tly - rain_geom.tly;
    float tile_j_f = ((float)j - top_gap/dem_geom.dy) * dem_geom.dy / rain_geom.dy;
    int tile_j = (int)floor(tile_j_f);
    
    if (tile_i >= 0 && tile_i < rain_geom.xsz && 
        tile_j >= 0 && tile_j < rain_geom.ysz) {
         // Rain data is row-major (y * xsz + x)
         return rain_data[tile_j * rain_geom.xsz + tile_i];
    }
    return 0.0f;
}

kernel void apply_rain(
    device float* H [[buffer(0)]],
    device const float* DEM [[buffer(1)]],
    device const float* rain_data [[buffer(2)]],
    constant Geometry& dem_geom [[buffer(3)]],
    constant Geometry& rain_geom [[buffer(4)]],
    constant float& dt [[buffer(5)]],
    constant float& nodata [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x; // 0 to xsz-1
    int j = gid.y; // 0 to ysz-1
    
    if (i >= dem_geom.xsz || j >= dem_geom.ysz) return;
    
    // Calculate 1-based index for H/DEM which includes ghost cells
    int idx = (j + 1) * dem_geom.pitch + (i + 1);
    
    float z = DEM[idx];
    
    if (abs(z - nodata) < 1e-6f) return;
    
    float rate = rate_at_cell(rain_data, i, j, dem_geom, rain_geom);
    if (rate > 0.0f) {
        H[idx] += rate * dt;
    }
}

// Update ghost cells with reflective (wall) boundary conditions
// This copies interior values to ghost cells with velocity reflection
kernel void update_ghost_cells(
    device float* H [[buffer(0)]],
    device float* HU [[buffer(1)]],
    device float* HV [[buffer(2)]],
    constant Geometry& geom [[buffer(3)]],
    device const int* boundary_types [[buffer(4)]], // Packed array: N, E, S, W
    uint gid [[thread_position_in_grid]])
{
    int pitch = geom.pitch;
    int xsz = geom.xsz;
    int ysz = geom.ysz;
    
    // Offset for boundary_types packed array
    // North: 0
    // East: xsz
    // South: xsz + ysz
    // West: 2*xsz + ysz
    
    // West and East boundaries (i=0 and i=xsz+1)
    if (gid < (uint)(ysz + 2)) {
        int j = gid;
        
        // West boundary (i=0)
        if (j >= 1 && j <= ysz) {
            int bc_idx = 2 * xsz + ysz + (ysz - j); // LISFLOOD West index
            int type = boundary_types[bc_idx];
            
            int idx_ghost_w = j * pitch + 0;
            int idx_interior_w = j * pitch + 1;
            H[idx_ghost_w] = H[idx_interior_w];
            HV[idx_ghost_w] = HV[idx_interior_w];
            if (type == BC_FREE) {
                HU[idx_ghost_w] = HU[idx_interior_w]; // Simple transmissive (matches CPU fv1)
            } else {
                HU[idx_ghost_w] = -HU[idx_interior_w]; // Reflective
            }
        }
        
        // East boundary (i=xsz+1)
        if (j >= 1 && j <= ysz) {
            int bc_idx = xsz + (j - 1); // LISFLOOD East index
            int type = boundary_types[bc_idx];
            
            int idx_ghost_e = j * pitch + xsz + 1;
            int idx_interior_e = j * pitch + xsz;
            H[idx_ghost_e] = H[idx_interior_e];
            HV[idx_ghost_e] = HV[idx_interior_e];
            if (type == BC_FREE) {
                HU[idx_ghost_e] = HU[idx_interior_e]; // Simple transmissive (matches CPU fv1)
            } else {
                HU[idx_ghost_e] = -HU[idx_interior_e]; // Reflective
            }
        }
    }
    
    // North and South boundaries (j=0 and j=ysz+1)
    if (gid >= 1 && gid <= (uint)xsz) {
        int i = gid;
        
        // North boundary (j=0)
        int bc_idx_n = i - 1; // LISFLOOD North index
        int type_n = boundary_types[bc_idx_n];
        
        int idx_ghost_n = 0 * pitch + i;
        int idx_interior_n = 1 * pitch + i;
        H[idx_ghost_n] = H[idx_interior_n];
        HU[idx_ghost_n] = HU[idx_interior_n];
        if (type_n == BC_FREE) {
            HV[idx_ghost_n] = HV[idx_interior_n]; // Simple transmissive (matches CPU fv1)
        } else {
            HV[idx_ghost_n] = -HV[idx_interior_n]; // Reflective
        }
        
        // South boundary (j=ysz+1)
        int bc_idx_s = 2 * xsz + ysz - i; // LISFLOOD South index
        int type_s = boundary_types[bc_idx_s];
        
        int idx_ghost_s = (ysz + 1) * pitch + i;
        int idx_interior_s = ysz * pitch + i;
        H[idx_ghost_s] = H[idx_interior_s];
        HU[idx_ghost_s] = HU[idx_interior_s];
        if (type_s == BC_FREE) {
            HV[idx_ghost_s] = HV[idx_interior_s]; // Simple transmissive (matches CPU fv1)
        } else {
            HV[idx_ghost_s] = -HV[idx_interior_s]; // Reflective
        }
    }
}

// Drain water from NoData cells (implements drain_nodata functionality)
// Water entering NoData cells is eliminated (removed from simulation)
kernel void drain_nodata_cells(
    device float* H [[buffer(0)]],
    device float* HU [[buffer(1)]],
    device float* HV [[buffer(2)]],
    device const float* DEM [[buffer(3)]],
    constant Geometry& geom [[buffer(4)]],
    constant float& nodata [[buffer(5)]],
    device atomic_uint* mass_totals [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x + 1;
    int j = gid.y + 1;
    
    if (i > geom.xsz || j > geom.ysz) return;
    
    int idx = j * geom.pitch + i;
    float z = DEM[idx];
    
    // If this is a NoData cell, eliminate any water
    if (abs(z - nodata) < 1.0f) {
        if (H[idx] > 0.0f) atomic_add_float(&mass_totals[2], H[idx] * geom.dx * geom.dy);
        H[idx] = 0.0f;
        HU[idx] = 0.0f;
        HV[idx] = 0.0f;
    }
}

// Accumulate total outflow volume from FREE boundaries
// This kernel runs on 4 boundary sides and uses atomics to sum volume
kernel void accumulate_outflow(
    device const float* H [[buffer(0)]],
    device const float* HU [[buffer(1)]],
    device const float* HV [[buffer(2)]],
    device const int* boundary_types [[buffer(3)]],
    device atomic_uint* vol_out [[buffer(4)]], // Array of 4 totals: N, E, S, W
    constant Geometry& geom [[buffer(5)]],
    constant float& dt [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int pitch = geom.pitch;
    int xsz = geom.xsz;
    int ysz = geom.ysz;
    
    // West and East boundaries
    if (gid < (uint)ysz) {
        int j = gid + 1;
        
        // West boundary (i=1) - outflow is NEGATIVE HU
        int bc_idx_w = 2 * xsz + ysz + (ysz - j);
        if (boundary_types[bc_idx_w] == BC_FREE) {
            float flux = -HU[j * pitch + 1] * geom.dy * dt;
            if (flux > 0) atomic_add_float(&vol_out[3], flux);
        }
        
        // East boundary (i=xsz) - outflow is POSITIVE HU
        int bc_idx_e = xsz + (j - 1);
        if (boundary_types[bc_idx_e] == BC_FREE) {
            float flux = HU[j * pitch + xsz] * geom.dy * dt;
            if (flux > 0) atomic_add_float(&vol_out[1], flux);
        }
    }
    
    // North and South boundaries
    if (gid < (uint)xsz) {
        int i = gid + 1;
        
        // North boundary (j=1) - outflow is NEGATIVE HV
        int bc_idx_n = i - 1;
        if (boundary_types[bc_idx_n] == BC_FREE) {
            float flux = -HV[1 * pitch + i] * geom.dx * dt;
            if (flux > 0) atomic_add_float(&vol_out[0], flux);
        }
        
        // South boundary (j=ysz) - outflow is POSITIVE HV
        int bc_idx_s = xsz + ysz + (xsz - i);
        if (boundary_types[bc_idx_s] == BC_FREE) {
            float flux = HV[ysz * pitch + i] * geom.dx * dt;
            if (flux > 0) atomic_add_float(&vol_out[2], flux);
        }
    }
}

// Track the running maximum water depth per cell.
// Mirrors update_max_field() in swe/stats.cpp, which the CPU solver calls once
// per timestep. Keeping this on the GPU avoids having to copy H back to the
// host every step just to maintain the .max output.
kernel void update_max_field(
    device const float* H [[buffer(0)]],
    device float* maxH [[buffer(1)]],
    constant Geometry& geom [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = (int)gid.x + 1;
    int j = geom.row_begin + (int)gid.y;

    if (i > geom.xsz || j >= geom.row_end) return;

    int idx = j * geom.pitch + i;
    maxH[idx] = max(maxH[idx], H[idx]);
}

// ============================================================================
// FV2: slope limiting, reconstruction and the two Runge-Kutta stages.
// Follows cuda/fv2/cuda_fv2_solver.cu and cuda_fv2_slope_limit.cu.
//
// Conventions of this back end (as for FV1): the y axis points south (j + 1 is
// the cell to the south) and HV is stored with the opposite sign to the CPU and
// CUDA (positive southward). The slopes are computed with the same stencils as
// CUDA (x: backward = i - 1; y: backward = j + 1, the cell to the south), so a
// cell's faces are
//   east  = mean + 0.5 * slope_x     west  = mean - 0.5 * slope_x
//   north = mean + 0.5 * slope_y     south = mean - 0.5 * slope_y
// and HV's slope carries HV's sign, which keeps the face values consistent.
// ============================================================================

// cuda/fv2 Stencil::limit, as released (note: r = (local - forward) /
// (local - backward), so the limiter returns zero on monotone data and non-zero
// slopes only at local extrema; ported unchanged so that Metal FV2 reproduces
// CUDA FV2)
inline float fv2_limit(float backward, float local, float forward, float min_h)
{
    if (min_h <= 0.09f) return 0.0f;
    float b_jump = local - backward;
    float f_jump = local - forward;
    float r = (abs(b_jump) <= 0.0f) ? 0.0f : f_jump / b_jump;
    float beta = 1.25f;
    float phi = max(0.0f, max(min(beta * r, 1.0f), min(r, beta)));
    return phi * b_jump;
}

// Slopes of the depth (via the free surface), HU and HV for every interior cell.
kernel void fv2_limit_slopes(
    device const float* H [[buffer(0)]],
    device const float* HU [[buffer(1)]],
    device const float* HV [[buffer(2)]],
    device const float* DEM [[buffer(3)]],
    device float* H1x [[buffer(4)]],
    device float* HU1x [[buffer(5)]],
    device float* HV1x [[buffer(6)]],
    device float* H1y [[buffer(7)]],
    device float* HU1y [[buffer(8)]],
    device float* HV1y [[buffer(9)]],
    constant Geometry& geom [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x + 1;
    int j = gid.y + 1;
    if (i > geom.xsz || j > geom.ysz) return;
    int p = geom.pitch;
    int c = j * p + i;
    int w = c - 1, e = c + 1, s = c + p, n = c - p;

    float min_x = min(H[w], min(H[c], H[e]));
    H1x[c] = fv2_limit(DEM[w] + H[w], DEM[c] + H[c], DEM[e] + H[e], min_x);
    HU1x[c] = fv2_limit(HU[w], HU[c], HU[e], min_x);
    HV1x[c] = fv2_limit(HV[w], HV[c], HV[e], min_x);

    float min_y = min(H[s], min(H[c], H[n]));
    H1y[c] = fv2_limit(DEM[s] + H[s], DEM[c] + H[c], DEM[n] + H[n], min_y);
    HU1y[c] = fv2_limit(HU[s], HU[c], HU[n], min_y);
    HV1y[c] = fv2_limit(HV[s], HV[c], HV[n], min_y);
}

// cuda Boundary::outside_x/_y: the ghost-cell state of a boundary face, from
// the mean of the cell inside (rotated: Qn normal, Qt tangential)
inline EdgeState fv2_outside(int type, float value, float Zstar, float depth_thresh,
                             float qn_sign, EdgeState Uc)
{
    EdgeState o = Uc;
    if (type == BC_FREE || type == BC_FREE6) {
        // outflow only: inward-pointing momentum is reflected
        if (qn_sign * Uc.Qn > 0.0f) o.Qn = -Uc.Qn;
    } else if (type == BC_HFIX || type == BC_HVAR) {
        o.H = max(0.0f, value - Zstar);
    } else if (type == BC_QFIX || type == BC_QVAR) {
        o.Qn = qn_sign * value;
        o.Qt = 0.0f;
        if (abs(o.Qn) > 1e-6f) o.H = max(Uc.H, 1.1f * depth_thresh);
    } else {
        o.Qn = -Uc.Qn;
    }
    return o;
}

// cuda Boundary::inside_x/_y: the state inside a boundary face used in the
// flux; Uout is the star state of the ghost cell, Uc the mean of the cell
// inside and Uin its star state at the face
inline EdgeState fv2_inside(int type, EdgeState Uout, EdgeState Uc, EdgeState Uin)
{
    if (type == BC_FREE || type == BC_FREE6) return Uin;
    if (type == BC_HFIX || type == BC_HVAR) { Uc.H = Uout.H; return Uc; }
    if (type == BC_QFIX || type == BC_QVAR) return EdgeState{Uout.H, Uout.Qn, Uc.Qt};
    return Uc;
}

// cuda fv2 update_ghost_cells: every ghost cell holds the outside state of
// its boundary face (HV positive southward: north inflow is +q)
kernel void fv2_update_ghost_cells(
    device float* H [[buffer(0)]],
    device float* HU [[buffer(1)]],
    device float* HV [[buffer(2)]],
    device const float* Zstar_x [[buffer(3)]],
    device const float* Zstar_y [[buffer(4)]],
    constant Geometry& geom [[buffer(5)]],
    constant SolverParams& solver [[buffer(6)]],
    device const int* boundary_types [[buffer(7)]],
    device const float* boundary_values [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int p = geom.pitch, xsz = geom.xsz, ysz = geom.ysz;
    int j = (int)gid;
    if (j >= 1 && j <= ysz) {
        int c = j * p + 1;                               // west
        int b = 2 * xsz + 2 * ysz - j;
        EdgeState o = fv2_outside(boundary_types[b], boundary_values[b], Zstar_x[j * p],
                                  solver.depth_thresh, 1.0f, EdgeState{H[c], HU[c], HV[c]});
        H[c - 1] = o.H; HU[c - 1] = o.Qn; HV[c - 1] = o.Qt;
        c = j * p + xsz;                                 // east
        b = xsz + (j - 1);
        o = fv2_outside(boundary_types[b], boundary_values[b], Zstar_x[c],
                        solver.depth_thresh, -1.0f, EdgeState{H[c], HU[c], HV[c]});
        H[c + 1] = o.H; HU[c + 1] = o.Qn; HV[c + 1] = o.Qt;
    }
    int i = (int)gid;
    if (i >= 1 && i <= xsz) {
        int c = p + i;                                   // north
        int b = i - 1;
        EdgeState o = fv2_outside(boundary_types[b], boundary_values[b], Zstar_y[i],
                                  solver.depth_thresh, 1.0f, EdgeState{H[c], HV[c], HU[c]});
        H[c - p] = o.H; HV[c - p] = o.Qn; HU[c - p] = o.Qt;
        c = ysz * p + i;                                 // south
        b = 2 * xsz + ysz - i;
        o = fv2_outside(boundary_types[b], boundary_values[b], Zstar_y[c],
                        solver.depth_thresh, -1.0f, EdgeState{H[c], HV[c], HU[c]});
        H[c + p] = o.H; HV[c + p] = o.Qn; HU[c + p] = o.Qt;
    }
}

// cuda fv2 update_dt_per_element: every cell, ghost cells included (they hold
// the boundary states, e.g. the inflow of a QFIX boundary)
kernel void fv2_compute_dt(
    device const float* H [[buffer(0)]],
    device const float* HU [[buffer(1)]],
    device const float* HV [[buffer(2)]],
    constant Geometry& geom [[buffer(3)]],
    constant PhysicalParams& params [[buffer(4)]],
    constant SolverParams& solver [[buffer(5)]],
    constant float& cfl [[buffer(6)]],
    constant float& max_dt [[buffer(7)]],
    volatile device atomic_uint* min_dt_atomic [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = (int)gid.x, j = (int)gid.y;
    float local_dt = max_dt;
    if (i < geom.pitch && j < geom.ysz + 2) {
        int c = j * geom.pitch + i;
        float h = H[c];
        if (h > solver.depth_thresh) {
            float c0 = sqrt(params.g * h);
            float dt_x = cfl * geom.dx / (abs(HU[c] / h) + c0);
            float dt_y = cfl * geom.dy / (abs(HV[c] / h) + c0);
            local_dt = min(local_dt, min(dt_x, dt_y));
        }
    }
    local_dt = simd_min(local_dt);
    if (simd_is_first())
        atomic_fetch_min_explicit(min_dt_atomic, as_type<uint>(local_dt), memory_order_relaxed);
}

// Momentum of cells shallower than DepthThresh is zeroed after each stage.
kernel void fv2_zero_discharge(
    device const float* H [[buffer(0)]],
    device float* HU [[buffer(1)]],
    device float* HV [[buffer(2)]],
    constant Geometry& geom [[buffer(3)]],
    constant SolverParams& solver [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x + 1;
    int j = gid.y + 1;
    if (i > geom.xsz || j > geom.ysz) return;
    int c = j * geom.pitch + i;
    if (H[c] < solver.depth_thresh) {
        HU[c] = 0.0f;
        HV[c] = 0.0f;
    }
}

// cuda FlowVector::star: depth over the higher bed of the face, and the
// velocity of the face state (zero below DepthThresh)
inline FlowVector fv2_star(FlowVector U, float Z, float Zstar, float depth_thresh)
{
    float h = max(0.0f, U.H + Z - Zstar);
    float u = (U.H >= depth_thresh) ? U.HU / U.H : 0.0f;
    float v = (U.H >= depth_thresh) ? U.HV / U.H : 0.0f;
    return FlowVector{h, h * u, h * v};
}

inline float fv2_zdagger(float eta, float zstar)
{
    return zstar - max(0.0f, -(eta - zstar));
}

// One Runge-Kutta stage in x or y. L is the flux divergence plus bed source of
// the state In (with its limited slopes); the output is
//   mode 0: A + dt L      mode 1: 0.5 (A + B + dt L)      mode 2: A + 0.5 dt L
// (cuda rk_stage1_x/_y use mode 0, rk_stage2_x mode 1, rk_stage2_y mode 2).
// Each face flux is computed once and shared through threadgroup memory, with
// the same overlapping tiles as the FV1 sweeps.
struct Fv2Stage {
    int mode;
    float mass_weight;  // share of this stage in the boundary volumes (0.5)
};

inline FlowVector fv2_combine(int mode, FlowVector A, FlowVector B, FlowVector L, float dt)
{
    if (mode == 0) return FlowVector{A.H + dt * L.H, A.HU + dt * L.HU, A.HV + dt * L.HV};
    if (mode == 1) return FlowVector{0.5f * (A.H + B.H + dt * L.H),
                                     0.5f * (A.HU + B.HU + dt * L.HU),
                                     0.5f * (A.HV + B.HV + dt * L.HV)};
    return FlowVector{A.H + 0.5f * dt * L.H, A.HU + 0.5f * dt * L.HU, A.HV + 0.5f * dt * L.HV};
}

kernel void fv2_flow_x(
    device const float* H [[buffer(0)]],
    device const float* HU [[buffer(1)]],
    device const float* HV [[buffer(2)]],
    device const float* H1x [[buffer(3)]],
    device const float* HU1x [[buffer(4)]],
    device const float* HV1x [[buffer(5)]],
    device const float* HA [[buffer(6)]],
    device const float* HUA [[buffer(7)]],
    device const float* HVA [[buffer(8)]],
    device const float* HB [[buffer(9)]],
    device const float* HUB [[buffer(10)]],
    device const float* HVB [[buffer(11)]],
    device float* HO [[buffer(12)]],
    device float* HUO [[buffer(13)]],
    device float* HVO [[buffer(14)]],
    device const float* DEM [[buffer(15)]],
    device const float* Zstar_x [[buffer(16)]],
    constant Geometry& geom [[buffer(17)]],
    constant PhysicalParams& params [[buffer(18)]],
    constant SolverParams& solver [[buffer(19)]],
    device const int* boundary_types [[buffer(20)]],
    device const float* boundary_values [[buffer(21)]],
    device atomic_uint* mass_totals [[buffer(22)]],
    constant Fv2Stage& stage [[buffer(23)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup FlowVector F[TG_Y][TG_X];
    threadgroup FlowVector Ustar_sh[TG_Y][TG_X];

    int i = (int)(tgid.x * (TG_X - 1) + tid.x);   // face between i and i+1; cell i
    int j = (int)(tgid.y * TG_Y + tid.y) + 1;
    bool active = (i <= geom.xsz && j <= geom.ysz);

    int idx = j * geom.pitch + i;
    FlowVector U_w = FlowVector{0.0f, 0.0f, 0.0f};
    float s_H = 0.0f, Z_w = 0.0f;
    FlowVector Ustar_neg = FlowVector{0.0f, 0.0f, 0.0f};
    FlowVector F_e = FlowVector{0.0f, 0.0f, 0.0f};

    if (active) {
        float Zstar = Zstar_x[idx];
        U_w = FlowVector{H[idx], HU[idx], HV[idx]};
        s_H = H1x[idx];
        FlowVector U_w_east = FlowVector{U_w.H + 0.5f * s_H, U_w.HU + 0.5f * HU1x[idx],
                                         U_w.HV + 0.5f * HV1x[idx]};
        int e = idx + 1;
        FlowVector U_e_west = FlowVector{H[e] - 0.5f * H1x[e], HU[e] - 0.5f * HU1x[e],
                                         HV[e] - 0.5f * HV1x[e]};
        Z_w = DEM[idx];
        float Z_e = DEM[e];
        Ustar_neg = fv2_star(U_w_east, Z_w, Zstar, solver.depth_thresh);
        FlowVector Ustar_pos = fv2_star(U_e_west, Z_e, Zstar, solver.depth_thresh);
        Ustar_sh[tid.y][tid.x] = Ustar_pos;   // cell i+1's west face, before the BC

        FlowVector flux_neg = Ustar_neg;
        FlowVector flux_pos = Ustar_pos;
        if (i == 0) {
            int b = 2 * geom.xsz + 2 * geom.ysz - j;
            EdgeState in_s = fv2_inside(boundary_types[b],
                EdgeState{Ustar_neg.H, Ustar_neg.HU, Ustar_neg.HV},
                EdgeState{H[e], HU[e], HV[e]},
                EdgeState{Ustar_pos.H, Ustar_pos.HU, Ustar_pos.HV});
            flux_pos = FlowVector{in_s.H, in_s.Qn, in_s.Qt};
        } else if (i == geom.xsz) {
            int b = geom.xsz + (j - 1);
            EdgeState in_s = fv2_inside(boundary_types[b],
                EdgeState{Ustar_pos.H, Ustar_pos.HU, Ustar_pos.HV},
                EdgeState{U_w.H, U_w.HU, U_w.HV},
                EdgeState{Ustar_neg.H, Ustar_neg.HU, Ustar_neg.HV});
            flux_neg = FlowVector{in_s.H, in_s.Qn, in_s.Qt};
        }
        F_e = hll_x(flux_neg, flux_pos, params.g, solver.depth_thresh);
        F[tid.y][tid.x] = F_e;

        float vol = stage.mass_weight * F_e.H * solver.dt * geom.dy;
        if (i == 0) account_boundary(mass_totals, -vol);
        if (i == geom.xsz) account_boundary(mass_totals, vol);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.x == 0 || !active || i < 1) return;

    FlowVector F_w = F[tid.y][tid.x - 1];
    FlowVector Ustar_west = Ustar_sh[tid.y][tid.x - 1];   // this cell's west face

    // bed source (cuda bed_source_x)
    float Hstar_0x = 0.5f * (Ustar_west.H + Ustar_neg.H);
    float eta_w = (U_w.H - 0.5f * s_H) + Z_w;
    float eta_e = (U_w.H + 0.5f * s_H) + Z_w;
    float zd_w = fv2_zdagger(eta_w, Zstar_x[idx - 1]);
    float zd_e = fv2_zdagger(eta_e, Zstar_x[idx]);
    float zd_1x = (zd_e - zd_w) / 2.0f;
    float S = -2.0f * params.g * Hstar_0x * zd_1x / geom.dx;

    FlowVector L = FlowVector{-(F_e.H - F_w.H) / geom.dx,
                              -(F_e.HU - F_w.HU) / geom.dx + S,
                              -(F_e.HV - F_w.HV) / geom.dx};
    FlowVector A = FlowVector{HA[idx], HUA[idx], HVA[idx]};
    FlowVector B = (stage.mode == 1) ? FlowVector{HB[idx], HUB[idx], HVB[idx]}
                                     : FlowVector{0.0f, 0.0f, 0.0f};
    FlowVector O = fv2_combine(stage.mode, A, B, L, solver.dt);
    HO[idx] = O.H;
    HUO[idx] = O.HU;
    HVO[idx] = O.HV;
}

kernel void fv2_flow_y(
    device const float* H [[buffer(0)]],
    device const float* HU [[buffer(1)]],
    device const float* HV [[buffer(2)]],
    device const float* H1y [[buffer(3)]],
    device const float* HU1y [[buffer(4)]],
    device const float* HV1y [[buffer(5)]],
    device const float* HA [[buffer(6)]],
    device const float* HUA [[buffer(7)]],
    device const float* HVA [[buffer(8)]],
    device const float* HB [[buffer(9)]],
    device const float* HUB [[buffer(10)]],
    device const float* HVB [[buffer(11)]],
    device float* HO [[buffer(12)]],
    device float* HUO [[buffer(13)]],
    device float* HVO [[buffer(14)]],
    device const float* DEM [[buffer(15)]],
    device const float* Zstar_y [[buffer(16)]],
    constant Geometry& geom [[buffer(17)]],
    constant PhysicalParams& params [[buffer(18)]],
    constant SolverParams& solver [[buffer(19)]],
    device const int* boundary_types [[buffer(20)]],
    device const float* boundary_values [[buffer(21)]],
    device atomic_uint* mass_totals [[buffer(22)]],
    constant Fv2Stage& stage [[buffer(23)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup FlowVector F[TG_Y][TG_X];
    threadgroup FlowVector Ustar_sh[TG_Y][TG_X];

    int i = (int)(tgid.x * TG_X + tid.x) + 1;
    int j = (int)(tgid.y * (TG_Y - 1) + tid.y);   // face between j and j+1 (south); cell j
    bool active = (i <= geom.xsz && j <= geom.ysz);

    int idx = j * geom.pitch + i;
    FlowVector U_n = FlowVector{0.0f, 0.0f, 0.0f};   // this cell (row j)
    float s_H = 0.0f, Z_n = 0.0f;
    FlowVector Ustar_south_face = FlowVector{0.0f, 0.0f, 0.0f};
    FlowVector F_s = FlowVector{0.0f, 0.0f, 0.0f};

    if (active) {
        float Zstar = Zstar_y[idx];
        U_n = FlowVector{H[idx], HU[idx], HV[idx]};
        s_H = H1y[idx];
        FlowVector U_n_south = FlowVector{U_n.H - 0.5f * s_H, U_n.HU - 0.5f * HU1y[idx],
                                          U_n.HV - 0.5f * HV1y[idx]};
        int s = idx + geom.pitch;
        FlowVector U_s_north = FlowVector{H[s] + 0.5f * H1y[s], HU[s] + 0.5f * HU1y[s],
                                          HV[s] + 0.5f * HV1y[s]};
        Z_n = DEM[idx];
        float Z_s = DEM[s];
        Ustar_south_face = fv2_star(U_n_south, Z_n, Zstar, solver.depth_thresh);
        FlowVector Ustar_north_of_s = fv2_star(U_s_north, Z_s, Zstar, solver.depth_thresh);
        Ustar_sh[tid.y][tid.x] = Ustar_north_of_s;   // cell j+1's north face, before the BC

        FlowVector flux_neg = Ustar_south_face;    // north side of the face
        FlowVector flux_pos = Ustar_north_of_s;    // south side
        if (j == 0) {
            int b = i - 1;
            EdgeState in_s = fv2_inside(boundary_types[b],
                EdgeState{Ustar_south_face.H, Ustar_south_face.HV, Ustar_south_face.HU},
                EdgeState{H[s], HV[s], HU[s]},
                EdgeState{Ustar_north_of_s.H, Ustar_north_of_s.HV, Ustar_north_of_s.HU});
            flux_pos = FlowVector{in_s.H, in_s.Qt, in_s.Qn};
        } else if (j == geom.ysz) {
            int b = 2 * geom.xsz + geom.ysz - i;
            EdgeState in_s = fv2_inside(boundary_types[b],
                EdgeState{Ustar_north_of_s.H, Ustar_north_of_s.HV, Ustar_north_of_s.HU},
                EdgeState{U_n.H, U_n.HV, U_n.HU},
                EdgeState{Ustar_south_face.H, Ustar_south_face.HV, Ustar_south_face.HU});
            flux_neg = FlowVector{in_s.H, in_s.Qt, in_s.Qn};
        }
        F_s = hll_y(flux_neg, flux_pos, params.g, solver.depth_thresh);
        F[tid.y][tid.x] = F_s;

        float vol = stage.mass_weight * F_s.H * solver.dt * geom.dx;
        if (j == 0) account_boundary(mass_totals, -vol);
        if (j == geom.ysz) account_boundary(mass_totals, vol);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.y == 0 || !active || j < 1) return;

    FlowVector F_n = F[tid.y - 1][tid.x];
    FlowVector Ustar_north = Ustar_sh[tid.y - 1][tid.x];   // this cell's north face

    // bed source (cuda bed_source_y, with HV positive southward here)
    float Hstar_0y = 0.5f * (Ustar_north.H + Ustar_south_face.H);
    float eta_s = (U_n.H - 0.5f * s_H) + Z_n;
    float eta_n = (U_n.H + 0.5f * s_H) + Z_n;
    float zd_s = fv2_zdagger(eta_s, Zstar_y[idx]);
    float zd_n = fv2_zdagger(eta_n, Zstar_y[idx - geom.pitch]);
    float zd_1y = (zd_n - zd_s) / 2.0f;
    float S = 2.0f * params.g * Hstar_0y * zd_1y / geom.dy;

    FlowVector L = FlowVector{-(F_s.H - F_n.H) / geom.dy,
                              -(F_s.HU - F_n.HU) / geom.dy,
                              -(F_s.HV - F_n.HV) / geom.dy + S};
    FlowVector A = FlowVector{HA[idx], HUA[idx], HVA[idx]};
    FlowVector B = (stage.mode == 1) ? FlowVector{HB[idx], HUB[idx], HVB[idx]}
                                     : FlowVector{0.0f, 0.0f, 0.0f};
    FlowVector O = fv2_combine(stage.mode, A, B, L, solver.dt);
    HO[idx] = O.H;
    HUO[idx] = O.HU;
    HVO[idx] = O.HV;
}

// ============================================================================
// DG2: a line-by-line port of cuda/dg2 (cuda_dg2_solver.cu, _slope_limit.cu,
// _dem.cu, _flow.cu), so that Metal DG2 reproduces CUDA DG2.
//
// Unlike the FV1 and FV2 kernels above, DG2 keeps the CUDA conventions: rows
// run north to south (j - 1 is north), HV and every y slope are positive
// northward, and the host uploads HV unchanged.
//
// Each set of flow coefficients is one buffer of 9 planes of pitch x (ysz + 2)
// floats: H, H1x, H1y, HU, HU1x, HU1y, HV, HV1x, HV1y (Metal allows only 31
// buffer arguments per kernel, and a stage reads three sets and writes one).
// The DEM is one buffer of 3 planes: Z, Z1x, Z1y.
// ============================================================================

#define SQ3 1.7320508075688772f

enum { P_H = 0, P_H1x, P_H1y, P_HU, P_HU1x, P_HU1y, P_HV, P_HV1x, P_HV1y };

struct Dg2Params {
    float krivodonova_thresh;
    float max_h;          // filled by dg2_max_h (cuda dg2::maxH)
};

inline int dg2_plane(constant Geometry& geom) { return geom.pitch * (geom.ysz + 2); }

struct Coeffs { float c[9]; };

inline Coeffs dg2_load(device const float* U, int idx, int N)
{
    Coeffs r;
    for (int k = 0; k < 9; k++) r.c[k] = U[k * N + idx];
    return r;
}

inline FlowVector dg2_get0(Coeffs U) { return FlowVector{U.c[P_H], U.c[P_HU], U.c[P_HV]}; }
inline FlowVector dg2_neg_x(Coeffs U) { return FlowVector{U.c[P_H] + SQ3 * U.c[P_H1x], U.c[P_HU] + SQ3 * U.c[P_HU1x], U.c[P_HV] + SQ3 * U.c[P_HV1x]}; }
inline FlowVector dg2_pos_x(Coeffs U) { return FlowVector{U.c[P_H] - SQ3 * U.c[P_H1x], U.c[P_HU] - SQ3 * U.c[P_HU1x], U.c[P_HV] - SQ3 * U.c[P_HV1x]}; }
inline FlowVector dg2_neg_y(Coeffs U) { return FlowVector{U.c[P_H] + SQ3 * U.c[P_H1y], U.c[P_HU] + SQ3 * U.c[P_HU1y], U.c[P_HV] + SQ3 * U.c[P_HV1y]}; }
inline FlowVector dg2_pos_y(Coeffs U) { return FlowVector{U.c[P_H] - SQ3 * U.c[P_H1y], U.c[P_HU] - SQ3 * U.c[P_HU1y], U.c[P_HV] - SQ3 * U.c[P_HV1y]}; }

inline float dem_neg_x(device const float* Z, int idx, int N) { return Z[idx] + SQ3 * Z[N + idx]; }
inline float dem_pos_x(device const float* Z, int idx, int N) { return Z[idx] - SQ3 * Z[N + idx]; }
inline float dem_neg_y(device const float* Z, int idx, int N) { return Z[idx] + SQ3 * Z[2 * N + idx]; }
inline float dem_pos_y(device const float* Z, int idx, int N) { return Z[idx] - SQ3 * Z[2 * N + idx]; }

// cuda FlowVector::star
inline FlowVector dg2_star(FlowVector U, float Z, float Zstar, float depth_thresh)
{
    float Hstar = max(0.0f, U.H + Z - Zstar);
    float u = (U.H >= depth_thresh) ? U.HU / U.H : 0.0f;
    float v = (U.H >= depth_thresh) ? U.HV / U.H : 0.0f;
    return FlowVector{Hstar, Hstar * u, Hstar * v};
}

inline FlowVector physical_flux_x(FlowVector U, float g, float depth_thresh)
{
    if (U.H <= depth_thresh) return FlowVector{0.0f, 0.0f, 0.0f};
    return FlowVector{U.HU, U.HU * U.HU / U.H + 0.5f * g * U.H * U.H, U.HU * U.HV / U.H};
}

inline FlowVector physical_flux_y(FlowVector U, float g, float depth_thresh)
{
    if (U.H <= depth_thresh) return FlowVector{0.0f, 0.0f, 0.0f};
    return FlowVector{U.HV, U.HU * U.HV / U.H, U.HV * U.HV / U.H + 0.5f * g * U.H * U.H};
}

inline FlowVector fv_add(FlowVector a, FlowVector b) { return FlowVector{a.H + b.H, a.HU + b.HU, a.HV + b.HV}; }
inline FlowVector fv_sub(FlowVector a, FlowVector b) { return FlowVector{a.H - b.H, a.HU - b.HU, a.HV - b.HV}; }
inline FlowVector fv_scale(float s, FlowVector a) { return FlowVector{s * a.H, s * a.HU, s * a.HV}; }

inline float dg2_zdagger(float eta, float zstar) { return zstar - max(0.0f, -(eta - zstar)); }

// cuda dg2 initialise_Zstar_x / _y
kernel void dg2_zstar(
    device const float* Z [[buffer(0)]],
    device float* Zstar_x [[buffer(1)]],
    device float* Zstar_y [[buffer(2)]],
    constant Geometry& geom [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int N = dg2_plane(geom);
    int p = geom.pitch;
    int i = (int)gid.x, j = (int)gid.y;
    if (j >= 1 && j <= geom.ysz && i <= geom.xsz) {
        int c = j * p + i;
        Zstar_x[c] = max(dem_neg_x(Z, c, N), dem_pos_x(Z, c + 1, N));
    }
    if (i >= 1 && i <= geom.xsz && j <= geom.ysz) {
        int c = j * p + i;
        Zstar_y[c] = max(dem_neg_y(Z, c + p, N), dem_pos_y(Z, c, N));
    }
}

// cuda dg2 friction (one point)
inline void dg2_friction_point(float n, float H, thread float& HU, thread float& HV,
                               float g, float dt, float depth_thresh, float speed_thresh)
{
    if (H <= depth_thresh) { HU = 0.0f; HV = 0.0f; return; }
    float U = HU / H;
    float V = HV / H;
    if (abs(U) <= speed_thresh && abs(V) <= speed_thresh) { HU = 0.0f; HV = 0.0f; return; }
    float speed = sqrt(U * U + V * V);
    float Cf = g * n * n / pow(H, 1.0f / 3.0f);
    float Sfx = -Cf * U * speed;
    float Sfy = -Cf * V * speed;
    float Dx = 1.0f + dt * Cf / H * (2.0f * U * U + V * V) / speed;
    float Dy = 1.0f + dt * Cf / H * (U * U + 2.0f * V * V) / speed;
    HU += dt * Sfx / Dx;
    HV += dt * Sfy / Dy;
}

// cuda dg2 apply_friction: the mean, and the slopes from the four Gauss points
kernel void dg2_friction(
    device float* U [[buffer(0)]],
    device const float* manning [[buffer(1)]],
    constant Geometry& geom [[buffer(2)]],
    constant PhysicalParams& params [[buffer(3)]],
    constant SolverParams& solver [[buffer(4)]],
    constant int& has_manning_map [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = (int)gid.x + 1, j = geom.row_begin + (int)gid.y;
    if (i > geom.xsz || j >= geom.row_end) return;
    int N = dg2_plane(geom);
    int c = j * geom.pitch + i;
    Coeffs C = dg2_load(U, c, N);
    float n = has_manning_map ? manning[c] : params.manning;
    float g = params.g, dt = solver.dt, th = solver.depth_thresh, st = params.speed_thresh;

    float HU0 = C.c[P_HU], HV0 = C.c[P_HV];
    dg2_friction_point(n, C.c[P_H], HU0, HV0, g, dt, th, st);

    float Hlx = C.c[P_H] - C.c[P_H1x], HUlx = C.c[P_HU] - C.c[P_HU1x], HVlx = C.c[P_HV] - C.c[P_HV1x];
    float Hux = C.c[P_H] + C.c[P_H1x], HUux = C.c[P_HU] + C.c[P_HU1x], HVux = C.c[P_HV] + C.c[P_HV1x];
    float Hly = C.c[P_H] - C.c[P_H1y], HUly = C.c[P_HU] - C.c[P_HU1y], HVly = C.c[P_HV] - C.c[P_HV1y];
    float Huy = C.c[P_H] + C.c[P_H1y], HUuy = C.c[P_HU] + C.c[P_HU1y], HVuy = C.c[P_HV] + C.c[P_HV1y];
    dg2_friction_point(n, Hlx, HUlx, HVlx, g, dt, th, st);
    dg2_friction_point(n, Hux, HUux, HVux, g, dt, th, st);
    dg2_friction_point(n, Hly, HUly, HVly, g, dt, th, st);
    dg2_friction_point(n, Huy, HUuy, HVuy, g, dt, th, st);

    U[P_HU * N + c] = HU0;
    U[P_HV * N + c] = HV0;
    U[P_HU1x * N + c] = 0.5f * (HUux - HUlx);
    U[P_HU1y * N + c] = 0.5f * (HUuy - HUly);
    U[P_HV1x * N + c] = 0.5f * (HVux - HVlx);
    U[P_HV1y * N + c] = 0.5f * (HVuy - HVly);
}

// cuda dg2 MaxH::update: the largest depth over every element (ghost cells
// included); depths are compared as unsigned bit patterns, valid for h >= 0
kernel void dg2_max_h(
    device const float* U [[buffer(0)]],
    constant Geometry& geom [[buffer(1)]],
    device atomic_uint* max_bits [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int N = dg2_plane(geom);
    float h = ((int)gid < N) ? max(U[gid], 0.0f) : 0.0f;
    h = simd_max(h);
    if (simd_is_first()) atomic_fetch_max_explicit(max_bits, as_type<uint>(h), memory_order_relaxed);
}

// cuda dg2 copy_slopes: the slopes of the free surface and the discharges
kernel void dg2_copy_slopes(
    device const float* U [[buffer(0)]],
    device const float* Z [[buffer(1)]],
    device float* S [[buffer(2)]],   // 6 planes: ETA1x, ETA1y, HU1x, HU1y, HV1x, HV1y
    constant Geometry& geom [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int N = dg2_plane(geom);
    int c = (int)gid;
    if (c >= N) return;
    S[c] = Z[N + c] + U[P_H1x * N + c];
    S[N + c] = Z[2 * N + c] + U[P_H1y * N + c];
    S[2 * N + c] = U[P_HU1x * N + c];
    S[3 * N + c] = U[P_HU1y * N + c];
    S[4 * N + c] = U[P_HV1x * N + c];
    S[5 * N + c] = U[P_HV1y * N + c];
}

inline float dg2_minmod(float a, float b, float c)
{
    if (a * b > 0.0f && a * c > 0.0f) {
        float sign_a = (float)((a > 0.0f) - (a < 0.0f));
        return sign_a * min(abs(a), min(abs(b), abs(c)));
    }
    return 0.0f;
}

// cuda dg2 Stencil::limit (Krivodonova discontinuity detector + minmod)
inline float dg2_limit(float bc, float bs, float lc, float ls, float fc, float fs,
                       float mesh_delta, float stencil_minH, float max_h, float kthresh)
{
    if (stencil_minH <= 0.30f * max_h) return ls;
    float b_neg = bc + SQ3 * bs;
    float l_pos = lc - SQ3 * ls;
    float l_neg = lc + SQ3 * ls;
    float f_pos = fc - SQ3 * fs;
    float b_jump = abs(l_pos - b_neg);
    float f_jump = abs(l_neg - f_pos);
    float norm = max(abs(lc - bc), abs(fc - lc));
    float DS_backward = 0.0f, DS_forward = 0.0f;
    if (abs(norm) > 1e-6f) {
        DS_backward = b_jump / (0.5f * mesh_delta * norm);
        DS_forward = f_jump / (0.5f * mesh_delta * norm);
    }
    if (max(DS_backward, DS_forward) >= kthresh)
        return dg2_minmod(ls, (lc - bc) / SQ3, (fc - lc) / SQ3);
    return ls;
}

// cuda dg2 limit_slopes (stencils x: i - 1, i, i + 1; y: j + 1, j, j - 1)
kernel void dg2_limit_slopes(
    device const float* U [[buffer(0)]],
    device const float* Z [[buffer(1)]],
    device const float* S [[buffer(2)]],
    constant Geometry& geom [[buffer(3)]],
    constant Dg2Params& dp [[buffer(4)]],
    device const uint* max_bits [[buffer(5)]],
    device float* LS [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = (int)gid.x + 1, j = geom.row_begin + (int)gid.y;
    if (i > geom.xsz || j >= geom.row_end) return;
    int N = dg2_plane(geom), p = geom.pitch;
    int c = j * p + i;
    int w = c - 1, e = c + 1, s = c + p, n = c - p;
    float max_h = as_type<float>(max_bits[0]);
    float kt = dp.krivodonova_thresh;
    device const float* H = U;
    device const float* H1x = U + P_H1x * N;
    device const float* H1y = U + P_H1y * N;
    device const float* HU = U + P_HU * N;
    device const float* HV = U + P_HV * N;

    float minH_x = min(min(H[w] - H1x[w], H[w] + H1x[w]),
                   min(min(H[c] - H1x[c], H[c] + H1x[c]), min(H[e] - H1x[e], H[e] + H1x[e])));
    float minH_y = min(min(H[s] - H1y[s], H[s] + H1y[s]),
                   min(min(H[c] - H1y[c], H[c] + H1y[c]), min(H[n] - H1y[n], H[n] + H1y[n])));

    float H1x_new = dg2_limit(Z[w] + H[w], S[w], Z[c] + H[c], S[c], Z[e] + H[e], S[e],
                              geom.dx, minH_x, max_h, kt) - Z[N + c];
    float HU1x_new = dg2_limit(HU[w], S[2 * N + w], HU[c], S[2 * N + c], HU[e], S[2 * N + e],
                               geom.dx, minH_x, max_h, kt);
    float HV1x_new = dg2_limit(HV[w], S[4 * N + w], HV[c], S[4 * N + c], HV[e], S[4 * N + e],
                               geom.dx, minH_x, max_h, kt);
    float H1y_new = dg2_limit(Z[s] + H[s], S[N + s], Z[c] + H[c], S[N + c], Z[n] + H[n], S[N + n],
                              geom.dy, minH_y, max_h, kt) - Z[2 * N + c];
    float HU1y_new = dg2_limit(HU[s], S[3 * N + s], HU[c], S[3 * N + c], HU[n], S[3 * N + n],
                               geom.dy, minH_y, max_h, kt);
    float HV1y_new = dg2_limit(HV[s], S[5 * N + s], HV[c], S[5 * N + c], HV[n], S[5 * N + n],
                               geom.dy, minH_y, max_h, kt);
    // written to LS and copied back by dg2_apply_limited: cuda limits in
    // place, so a neighbour's minH may see slopes already limited, depending
    // on thread timing; here every cell reads the unlimited slopes
    LS[c] = H1x_new;
    LS[N + c] = HU1x_new;
    LS[2 * N + c] = HV1x_new;
    LS[3 * N + c] = H1y_new;
    LS[4 * N + c] = HU1y_new;
    LS[5 * N + c] = HV1y_new;
}

kernel void dg2_apply_limited(
    device float* U [[buffer(0)]],
    device const float* LS [[buffer(1)]],
    constant Geometry& geom [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = (int)gid.x + 1, j = geom.row_begin + (int)gid.y;
    if (i > geom.xsz || j >= geom.row_end) return;
    int N = dg2_plane(geom);
    int c = j * geom.pitch + i;
    U[P_H1x * N + c] = LS[c];
    U[P_HU1x * N + c] = LS[N + c];
    U[P_HV1x * N + c] = LS[2 * N + c];
    U[P_H1y * N + c] = LS[3 * N + c];
    U[P_HU1y * N + c] = LS[4 * N + c];
    U[P_HV1y * N + c] = LS[5 * N + c];
}

// cuda dg2 zero_perimeter_slopes: the cells along the edge are first order
kernel void dg2_zero_perimeter_slopes(
    device float* U [[buffer(0)]],
    constant Geometry& geom [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    int N = dg2_plane(geom), p = geom.pitch;
    int k = (int)gid + 1;
    if (k <= geom.xsz) {
        for (int pl = 1; pl < 9; pl++) {
            if (pl == P_HU || pl == P_HV) continue;
            U[pl * N + p + k] = 0.0f;
            U[pl * N + geom.ysz * p + k] = 0.0f;
        }
    }
    if (k <= geom.ysz) {
        for (int pl = 1; pl < 9; pl++) {
            if (pl == P_HU || pl == P_HV) continue;
            U[pl * N + k * p + 1] = 0.0f;
            U[pl * N + k * p + geom.xsz] = 0.0f;
        }
    }
}

// cuda dg2 update_ghost_cells (Boundary::outside_x/_y; HV positive north, so
// a discharge into the domain is +q on the west and south edges); a dry cell
// on a FREE edge is emptied, slopes included, as in cuda
kernel void dg2_update_ghost_cells(
    device float* U [[buffer(0)]],
    device const float* Z [[buffer(1)]],
    device const float* Zstar_x [[buffer(2)]],
    device const float* Zstar_y [[buffer(3)]],
    constant Geometry& geom [[buffer(4)]],
    constant SolverParams& solver [[buffer(5)]],
    device const int* boundary_types [[buffer(6)]],
    device const float* boundary_values [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int N = dg2_plane(geom), p = geom.pitch, xsz = geom.xsz, ysz = geom.ysz;
    float th = solver.depth_thresh;
    device float* H = U;
    device float* HU = U + P_HU * N;
    device float* HV = U + P_HV * N;
    int k = (int)gid + 1;
    if (k <= ysz) {
        int j = k;
        for (int side = 0; side < 2; side++) {
            bool west = (side == 0);
            int c = west ? j * p + 1 : j * p + xsz;
            int b = west ? 2 * xsz + 2 * ysz - j : xsz + (j - 1);
            float Zs = west ? Zstar_x[c - 1] : Zstar_x[c];
            int type = boundary_types[b];
            EdgeState o = fv2_outside(type, boundary_values[b], Zs, th, west ? 1.0f : -1.0f,
                                      EdgeState{H[c], HU[c], HV[c]});
            if (type == BC_FREE && H[c] <= th) {
                H[c] = 0.0f; U[P_H1x * N + c] = 0.0f; U[P_H1y * N + c] = 0.0f;
                o.H = 0.0f;
            }
            int gc = west ? c - 1 : c + 1;
            H[gc] = o.H; HU[gc] = o.Qn; HV[gc] = o.Qt;
        }
    }
    if (k <= xsz) {
        int i = k;
        // North ghost (j=0 ghost) — only the rank that owns row 1 writes it
        if (geom.owns_north_ghost) {
            int c = p + i;
            int b = i - 1;
            float Zs = Zstar_y[i];
            int type = boundary_types[b];
            EdgeState o = fv2_outside(type, boundary_values[b], Zs, th, -1.0f,
                                      EdgeState{H[c], HV[c], HU[c]});
            if (type == BC_FREE && H[c] <= th) {
                H[c] = 0.0f; U[P_H1x * N + c] = 0.0f; U[P_H1y * N + c] = 0.0f;
                o.H = 0.0f;
            }
            H[c - p] = o.H; HV[c - p] = o.Qn; HU[c - p] = o.Qt;
        }
        // South ghost (j=ysz+1 ghost) — only the rank that owns row ysz writes it
        if (geom.owns_south_ghost) {
            int c = ysz * p + i;
            int b = 2 * xsz + ysz - i;
            float Zs = Zstar_y[c];
            int type = boundary_types[b];
            EdgeState o = fv2_outside(type, boundary_values[b], Zs, th, 1.0f,
                                      EdgeState{H[c], HV[c], HU[c]});
            if (type == BC_FREE && H[c] <= th) {
                H[c] = 0.0f; U[P_H1x * N + c] = 0.0f; U[P_H1y * N + c] = 0.0f;
                o.H = 0.0f;
            }
            H[c + p] = o.H; HV[c + p] = o.Qn; HU[c + p] = o.Qt;
        }
    }
}

// cuda dg2 update_dt_per_element over every element, ghost cells included
kernel void dg2_compute_dt(
    device const float* U [[buffer(0)]],
    constant Geometry& geom [[buffer(1)]],
    constant PhysicalParams& params [[buffer(2)]],
    constant SolverParams& solver [[buffer(3)]],
    constant float& cfl [[buffer(4)]],
    constant float& max_dt [[buffer(5)]],
    volatile device atomic_uint* min_dt_atomic [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    int N = dg2_plane(geom);
    int i = (int)gid.x, j = (int)gid.y;
    float local_dt = max_dt;
    if (i < geom.pitch && j < geom.ysz + 2) {
        int c = j * geom.pitch + i;
        float h = U[c];
        if (h > solver.depth_thresh) {
            float c0 = sqrt(params.g * h);
            float dt_x = cfl * geom.dx / (abs(U[P_HU * N + c] / h) + c0);
            float dt_y = cfl * geom.dy / (abs(U[P_HV * N + c] / h) + c0);
            local_dt = min(local_dt, min(dt_x, dt_y));
        }
    }
    local_dt = simd_min(local_dt);
    if (simd_is_first())
        atomic_fetch_min_explicit(min_dt_atomic, as_type<uint>(local_dt), memory_order_relaxed);
}

// cuda dg2 zero_discharge
kernel void dg2_zero_discharge(
    device float* U [[buffer(0)]],
    constant Geometry& geom [[buffer(1)]],
    constant SolverParams& solver [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = (int)gid.x + 1, j = geom.row_begin + (int)gid.y;
    if (i > geom.xsz || j >= geom.row_end) return;
    int N = dg2_plane(geom);
    int c = j * geom.pitch + i;
    if (U[c] < solver.depth_thresh) {
        for (int pl = P_HU; pl <= P_HV1y; pl++) U[pl * N + c] = 0.0f;
    }
}

// cuda drain_nodata_water: only the depth is removed
kernel void dg2_drain_nodata(
    device float* U [[buffer(0)]],
    device const float* Z [[buffer(1)]],
    constant Geometry& geom [[buffer(2)]],
    constant float& nodata [[buffer(3)]],
    device atomic_uint* mass_totals [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = (int)gid.x + 1, j = geom.row_begin + (int)gid.y;
    if (i > geom.xsz || j >= geom.row_end) return;
    int c = j * geom.pitch + i;
    if (abs(Z[c] - nodata) < 1e-6f) {
        float h = U[c];
        U[c] = 0.0f;
        if (h != 0.0f) atomic_add_float(&mass_totals[2], h * geom.dx * geom.dy);
    }
}

// One Runge-Kutta stage of cuda dg2 update_flow_variables_x/_y. The output is
//   mode 0: A + dt L      mode 1: 0.5 (A + B + dt L)      mode 2: A + 0.5 dt L
// for all nine coefficients (L is zero in the planes a sweep does not touch).
inline void dg2_store(device float* O, device const float* A, device const float* B,
                      int mode, float dt, int c, int N, thread const float* L)
{
    for (int k = 0; k < 9; k++) {
        float a = A[k * N + c];
        float o;
        if (mode == 0) o = a + dt * L[k];
        else if (mode == 1) o = 0.5f * (a + B[k * N + c] + dt * L[k]);
        else o = a + 0.5f * dt * L[k];
        O[k * N + c] = o;
    }
}

kernel void dg2_flow_x(
    device const float* Uin [[buffer(0)]],
    device const float* A [[buffer(1)]],
    device const float* B [[buffer(2)]],
    device float* O [[buffer(3)]],
    device const float* Z [[buffer(4)]],
    device const float* Zstar_x [[buffer(5)]],
    constant Geometry& geom [[buffer(6)]],
    constant PhysicalParams& params [[buffer(7)]],
    constant SolverParams& solver [[buffer(8)]],
    device const int* boundary_types [[buffer(9)]],
    device atomic_uint* mass_totals [[buffer(10)]],
    constant Fv2Stage& stage [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup FlowVector F[TG_Y][TG_X];
    threadgroup FlowVector Ustar_sh[TG_Y][TG_X];

    int N = dg2_plane(geom);
    int i = (int)(tgid.x * (TG_X - 1) + tid.x);   // face between i and i+1; cell i
    int j = geom.row_begin + (int)(tgid.y * TG_Y + tid.y);
    bool active = (i <= geom.xsz && j < geom.row_end);
    int c = j * geom.pitch + i;
    float th = solver.depth_thresh, g = params.g;

    Coeffs U_w;
    FlowVector Ustar_neg = FlowVector{0.0f, 0.0f, 0.0f};
    FlowVector F_e = FlowVector{0.0f, 0.0f, 0.0f};

    if (active) {
        U_w = dg2_load(Uin, c, N);
        Coeffs U_e = dg2_load(Uin, c + 1, N);
        float Zstar = Zstar_x[c];
        Ustar_neg = dg2_star(dg2_neg_x(U_w), dem_neg_x(Z, c, N), Zstar, th);
        FlowVector Ustar_pos = dg2_star(dg2_pos_x(U_e), dem_pos_x(Z, c + 1, N), Zstar, th);
        Ustar_sh[tid.y][tid.x] = Ustar_pos;

        FlowVector Ustar_neg_flux = Ustar_neg;
        if (i == 0) {
            int b = 2 * geom.xsz + 2 * geom.ysz - j;
            FlowVector Ue0 = dg2_get0(U_e);
            EdgeState in_s = fv2_inside(boundary_types[b],
                EdgeState{Ustar_neg.H, Ustar_neg.HU, Ustar_neg.HV},
                EdgeState{Ue0.H, Ue0.HU, Ue0.HV},
                EdgeState{Ustar_pos.H, Ustar_pos.HU, Ustar_pos.HV});
            Ustar_pos = FlowVector{in_s.H, in_s.Qn, in_s.Qt};
        } else if (i == geom.xsz) {
            int b = geom.xsz + (j - 1);
            FlowVector Uw0 = dg2_get0(U_w);
            EdgeState in_s = fv2_inside(boundary_types[b],
                EdgeState{Ustar_pos.H, Ustar_pos.HU, Ustar_pos.HV},
                EdgeState{Uw0.H, Uw0.HU, Uw0.HV},
                EdgeState{Ustar_neg.H, Ustar_neg.HU, Ustar_neg.HV});
            Ustar_neg_flux = FlowVector{in_s.H, in_s.Qn, in_s.Qt};
        }
        F_e = hll_x(Ustar_neg_flux, Ustar_pos, g, th);
        F[tid.y][tid.x] = F_e;

        float vol = stage.mass_weight * F_e.H * solver.dt * geom.dy;
        if (i == 0) account_boundary(mass_totals, -vol);
        if (i == geom.xsz) account_boundary(mass_totals, vol);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.x == 0 || !active || i < 1) return;

    FlowVector F_w = F[tid.y][tid.x - 1];
    FlowVector Ustar_pos = Ustar_sh[tid.y][tid.x - 1];
    FlowVector Ustar_0x = fv_scale(0.5f, fv_add(Ustar_pos, Ustar_neg));
    FlowVector Ustar_1x = fv_scale(1.0f / (2.0f * SQ3), fv_sub(Ustar_neg, Ustar_pos));

    float L[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    FlowVector L0 = fv_scale(-1.0f / geom.dx, fv_sub(F_e, F_w));
    FlowVector fm = physical_flux_x(fv_sub(Ustar_0x, Ustar_1x), g, th);
    FlowVector fp = physical_flux_x(fv_add(Ustar_0x, Ustar_1x), g, th);
    FlowVector L1 = fv_scale(-SQ3 / geom.dx, fv_sub(fv_sub(fv_add(F_w, F_e), fm), fp));

    // bed source (cuda bed_source_x)
    float ETA_w = dg2_pos_x(U_w).H + dem_pos_x(Z, c, N);
    float ETA_e = dg2_neg_x(U_w).H + dem_neg_x(Z, c, N);
    float Zd_w = dg2_zdagger(ETA_w, Zstar_x[c - 1]);
    float Zd_e = dg2_zdagger(ETA_e, Zstar_x[c]);
    float Zd_1x = (Zd_e - Zd_w) / (2.0f * SQ3);
    float Sb0 = -2.0f * SQ3 * g * Ustar_0x.H * Zd_1x / geom.dx;
    float Sb1 = -2.0f * SQ3 * g * Ustar_1x.H * Zd_1x / geom.dx;

    L[P_H] = L0.H;   L[P_HU] = L0.HU + Sb0;   L[P_HV] = L0.HV;
    L[P_H1x] = L1.H; L[P_HU1x] = L1.HU + Sb1; L[P_HV1x] = L1.HV;
    dg2_store(O, A, B, stage.mode, solver.dt, c, N, L);
}

kernel void dg2_flow_y(
    device const float* Uin [[buffer(0)]],
    device const float* A [[buffer(1)]],
    device const float* B [[buffer(2)]],
    device float* O [[buffer(3)]],
    device const float* Z [[buffer(4)]],
    device const float* Zstar_y [[buffer(5)]],
    constant Geometry& geom [[buffer(6)]],
    constant PhysicalParams& params [[buffer(7)]],
    constant SolverParams& solver [[buffer(8)]],
    device const int* boundary_types [[buffer(9)]],
    device atomic_uint* mass_totals [[buffer(10)]],
    constant Fv2Stage& stage [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup FlowVector F[TG_Y][TG_X];
    threadgroup FlowVector Ustar_sh[TG_Y][TG_X];

    int N = dg2_plane(geom), p = geom.pitch;
    int i = (int)(tgid.x * TG_X + tid.x) + 1;
    int j = (geom.row_begin - 1) + (int)(tgid.y * (TG_Y - 1) + tid.y);
    bool active = (i <= geom.xsz && j <= geom.row_end - 1);
    int c = j * p + i;
    float th = solver.depth_thresh, g = params.g;

    Coeffs U_n;
    FlowVector Ustar_pos = FlowVector{0.0f, 0.0f, 0.0f};   // cell j's south face
    FlowVector F_s = FlowVector{0.0f, 0.0f, 0.0f};

    if (active) {
        U_n = dg2_load(Uin, c, N);
        Coeffs U_s = dg2_load(Uin, c + p, N);
        float Zstar = Zstar_y[c];
        FlowVector Ustar_neg = dg2_star(dg2_neg_y(U_s), dem_neg_y(Z, c + p, N), Zstar, th);
        Ustar_pos = dg2_star(dg2_pos_y(U_n), dem_pos_y(Z, c, N), Zstar, th);
        Ustar_sh[tid.y][tid.x] = Ustar_neg;

        FlowVector Ustar_pos_flux = Ustar_pos;
        if (j == 0) {
            int b = i - 1;
            FlowVector Us0 = dg2_get0(U_s);
            EdgeState in_s = fv2_inside(boundary_types[b],
                EdgeState{Ustar_pos.H, Ustar_pos.HV, Ustar_pos.HU},
                EdgeState{Us0.H, Us0.HV, Us0.HU},
                EdgeState{Ustar_neg.H, Ustar_neg.HV, Ustar_neg.HU});
            Ustar_neg = FlowVector{in_s.H, in_s.Qt, in_s.Qn};
        } else if (j == geom.ysz) {
            int b = 2 * geom.xsz + geom.ysz - i;
            FlowVector Un0 = dg2_get0(U_n);
            EdgeState in_s = fv2_inside(boundary_types[b],
                EdgeState{Ustar_neg.H, Ustar_neg.HV, Ustar_neg.HU},
                EdgeState{Un0.H, Un0.HV, Un0.HU},
                EdgeState{Ustar_pos.H, Ustar_pos.HV, Ustar_pos.HU});
            Ustar_pos_flux = FlowVector{in_s.H, in_s.Qt, in_s.Qn};
        }
        F_s = hll_y(Ustar_neg, Ustar_pos_flux, g, th);
        F[tid.y][tid.x] = F_s;

        // F_s.H is positive northward: out of the domain at the north edge
        float vol = stage.mass_weight * F_s.H * solver.dt * geom.dx;
        if (j == 0) account_boundary(mass_totals, vol);
        if (j == geom.ysz) account_boundary(mass_totals, -vol);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid.y == 0 || !active || j < 1) return;

    FlowVector F_n = F[tid.y - 1][tid.x];
    FlowVector Ustar_neg = Ustar_sh[tid.y - 1][tid.x];     // cell j's north face
    FlowVector Ustar_0y = fv_scale(0.5f, fv_add(Ustar_pos, Ustar_neg));
    FlowVector Ustar_1y = fv_scale(1.0f / (2.0f * SQ3), fv_sub(Ustar_neg, Ustar_pos));

    float L[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    FlowVector L0 = fv_scale(-1.0f / geom.dy, fv_sub(F_n, F_s));
    FlowVector fm = physical_flux_y(fv_sub(Ustar_0y, Ustar_1y), g, th);
    FlowVector fp = physical_flux_y(fv_add(Ustar_0y, Ustar_1y), g, th);
    FlowVector L1 = fv_scale(-SQ3 / geom.dy, fv_sub(fv_sub(fv_add(F_s, F_n), fm), fp));

    // bed source (cuda bed_source_y)
    float ETA_s = dg2_pos_y(U_n).H + dem_pos_y(Z, c, N);
    float ETA_n = dg2_neg_y(U_n).H + dem_neg_y(Z, c, N);
    float Zd_s = dg2_zdagger(ETA_s, Zstar_y[c]);
    float Zd_n = dg2_zdagger(ETA_n, Zstar_y[c - p]);
    float Zd_1y = (Zd_n - Zd_s) / (2.0f * SQ3);
    float Sb0 = -2.0f * SQ3 * g * Ustar_0y.H * Zd_1y / geom.dy;
    float Sb1 = -2.0f * SQ3 * g * Ustar_1y.H * Zd_1y / geom.dy;

    L[P_H] = L0.H;   L[P_HU] = L0.HU;   L[P_HV] = L0.HV + Sb0;
    L[P_H1y] = L1.H; L[P_HU1y] = L1.HU; L[P_HV1y] = L1.HV + Sb1;
    dg2_store(O, A, B, stage.mode, solver.dt, c, N, L);
}
