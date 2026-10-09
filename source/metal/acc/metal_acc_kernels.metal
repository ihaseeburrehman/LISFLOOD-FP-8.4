//
// LISFLOOD-FP Metal ACC Solver Kernels
//
// Port of the local-inertial ("acceleration") solver from cuda/acc/.
// The scheme is Bates et al. (2010) with the q-centred variant of
// de Almeida et al. (2012), and the formulation here follows
// CalcFPQxAcc_ / CalcFPQyAcc_ in cuda/acc/cuda_acc_solver.cu line for line.
//
// GRID LAYOUT (deliberately different from the FV1 backend):
//   H, DEM, manning :  xsz * ysz          index  j*xsz + i          (no ghost ring)
//   Qx, Qy          : (xsz+1) * (ysz+1)   index  j*(xsz+1) + i
// Qx[j][i] is the flux on the WEST face of cell (i,j); Qy[j][i] the NORTH face.
// This matches the CUDA layout exactly so results can be compared cell for cell.
//
// CONSERVATION. Unlike FV1, this scheme is structurally safe: every interface
// flux is computed once into Qx/Qy and the continuity update reads the two
// stored neighbours (Qx[i] and Qx[i+1]). No flux is ever evaluated twice, so
// the failure mode that affected the FV1 port cannot occur here. Keep it that
// way -- never inline a flux computation into update_H.
//

#include <metal_stdlib>
using namespace metal;

// Geometry (must match MetalAccGeometry on the host)
struct AccGeometry {
    int xsz;
    int ysz;
    float dx;
    float dy;
    float tly;
    float blx;
    float bly;
};

// Physical / solver parameters (must match the host structs)
struct AccPhysical {
    float g;
    float manning;     // uniform fallback when no Manning raster is supplied
};

struct AccSolver {
    float dt;
    float depth_thresh;
    float max_hflow;
    float theta;         // q-centred weighting; 1.0 reduces to Bates (2010)
    float nodata_elev;
    float max_dt;
    float cfl;
};

// Boundary types, matching ESourceType in lisflood.h
enum ACC_BC {
    ACC_NONE = 0,
    ACC_FREE = 1,
    ACC_HFIX = 2,
    ACC_HVAR = 3,
    ACC_QFIX = 4,
    ACC_QVAR = 5,
    ACC_FREE6 = 6
};

inline void atomic_add_f(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float f_new = as_type<float>(expected) + val;
        uint desired = as_type<uint>(f_new);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) break;
    }
}


// ------------------------------------------------------------ mass counters --
//
// Volumes for the .mass file (rain added, water drained, water created by the
// h < 0 clip, boundary in/out) are reduced to ONE partial sum per threadgroup
// and written without atomics; the host adds the partials in double precision
// after every step.
//
// The first version added every cell's contribution into a single float32
// counter with an atomic CAS loop. At 10 m that is ~10^7 increments of
// ~10^-4 m^3 per step into a counter that reaches 10^6 m^3 over a mass
// interval, and once the counter is large an increment below its ulp is simply
// lost: the Rain column under-reported by ~70x. It was also 10^7 contended CAS
// operations on one address per step.
//
// Every thread of the threadgroup must reach this call -- no early return
// before it, or the barrier deadlocks.
inline void store_tg_sum(float v, device float* partial, uint tg_index,
                         uint lid, uint lane, uint sg, uint nsg,
                         threadgroup float* scratch)
{
    float s = simd_sum(v);
    if (lane == 0) scratch[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float t = 0.0f;
        for (uint k = 0; k < nsg; ++k) t += scratch[k];
        partial[tg_index] = t;
    }
}

// ---------------------------------------------------------------- timestep --

// Per-cell CFL limit; the minimum is reduced into min_dt_atomic.
// Mirrors update_dt_per_element in cuda_acc_solver.cu: the ACC condition
// depends only on depth, dt = cfl*dx/sqrt(g*H).
kernel void acc_update_dt(
    device const float* H [[buffer(0)]],
    constant AccGeometry& geom [[buffer(1)]],
    constant AccPhysical& phys [[buffer(2)]],
    constant AccSolver& solv [[buffer(3)]],
    volatile device atomic_uint* min_dt_atomic [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x, j = gid.y;
    if (i >= geom.xsz || j >= geom.ysz) return;

    float h = H[j * geom.xsz + i];
    float dt_local = solv.max_dt;
    if (h > solv.depth_thresh) {
        dt_local = solv.cfl * geom.dx / sqrt(phys.g * h);
    }
    dt_local = simd_min(dt_local);
    if (simd_is_first()) {
        atomic_fetch_min_explicit(min_dt_atomic, as_type<uint>(dt_local),
                                  memory_order_relaxed);
    }
}

// ------------------------------------------------------------ flux updates --

// x-direction flux on the interface between cells (i,j) and (i+1,j),
// written to Qx[j*(xsz+1) + i+1]. Returns m3/s (the dx factor is applied here,
// as in CUDA), while Qxold/Qyold hold m2/s.
kernel void acc_floodplain_qx(
    device const float* H [[buffer(0)]],
    device const float* DEM [[buffer(1)]],
    device const float* manning [[buffer(2)]],
    device const float* Qxold [[buffer(3)]],
    device const float* Qyold [[buffer(4)]],
    device float* Qx [[buffer(5)]],
    device float* Vx [[buffer(6)]],
    constant AccGeometry& geom [[buffer(7)]],
    constant AccPhysical& phys [[buffer(8)]],
    constant AccSolver& solv [[buffer(9)]],
    constant int& use_friction [[buffer(10)]],
    constant int& voutput [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x, j = gid.y;
    if (i >= geom.xsz - 1 || j >= geom.ysz) return;

    int qw = geom.xsz + 1;
    int p0 = j * geom.xsz + i;
    int p1 = p0 + 1;
    int pq0 = j * qw + i + 1;

    float h0 = H[p0], h1 = H[p1];
    Qx[pq0] = 0.0f;
    if (!(h0 > solv.depth_thresh || h1 > solv.depth_thresh)) return;

    float z0 = DEM[p0], z1 = DEM[p1];
    float q0 = Qxold[pq0];
    float dt = solv.dt, g = phys.g;

    float qvect = q0;
    if (use_friction != 0) {
        float qy_avg = (Qyold[j * qw + i] + Qyold[j * qw + i + 1] +
                        Qyold[(j + 1) * qw + i] + Qyold[(j + 1) * qw + i + 1]) * 0.25f;
        qvect = sqrt(q0 * q0 + qy_avg * qy_avg);
    }

    float fn = (manning != nullptr) ? 0.5f * (manning[p0] + manning[p1])
                                    : phys.manning;
    float y0 = z0 + h0, y1 = z1 + h1;
    float dh = y0 - y1;
    float Sf = -dh / geom.dx;

    float hflow = max(y0, y1) - max(z0, z1);
    hflow = clamp(hflow, 0.0f, solv.max_hflow);

    float qup = Qxold[pq0 - 1];
    float qdown = Qxold[pq0 + 1];
    if (i == 0 && qup == 0.0f) qup = q0;
    if (i == geom.xsz - 2 && qdown == 0.0f) qdown = q0;

    float Q = 0.0f;
    if (hflow > solv.depth_thresh) {
        float denom = 1.0f + g * dt * hflow * fn * fn * fabs(qvect)
                            / pow(hflow, 10.0f / 3.0f);
        // q-centred scheme (de Almeida et al. 2012)
        Q = ((solv.theta * q0 + 0.5f * (1.0f - solv.theta) * (qup + qdown))
             - (g * dt * hflow * Sf)) / denom * geom.dx;
        // Fall back to the semi-implicit scheme of Bates et al. (2010) where
        // the centred form would drive flow against the water-surface slope.
        if (Q * dh < 0.0f) {
            Q = (q0 - (g * dt * hflow * Sf)) / denom * geom.dx;
        }
    }
    Qx[pq0] = Q;

    if (voutput != 0) {
        Vx[pq0] = (Q != 0.0f) ? (Q / geom.dx / hflow) : 0.0f;
    }
}

// y-direction flux between cells (i,j) and (i,j+1), written to Qy[(j+1)*qw + i].
kernel void acc_floodplain_qy(
    device const float* H [[buffer(0)]],
    device const float* DEM [[buffer(1)]],
    device const float* manning [[buffer(2)]],
    device const float* Qxold [[buffer(3)]],
    device const float* Qyold [[buffer(4)]],
    device float* Qy [[buffer(5)]],
    device float* Vy [[buffer(6)]],
    constant AccGeometry& geom [[buffer(7)]],
    constant AccPhysical& phys [[buffer(8)]],
    constant AccSolver& solv [[buffer(9)]],
    constant int& use_friction [[buffer(10)]],
    constant int& voutput [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x, j = gid.y;
    if (i >= geom.xsz || j >= geom.ysz - 1) return;

    int qw = geom.xsz + 1;
    int p0 = j * geom.xsz + i;
    int p1 = (j + 1) * geom.xsz + i;
    int pq0 = (j + 1) * qw + i;

    float h0 = H[p0], h1 = H[p1];
    Qy[pq0] = 0.0f;
    if (!(h0 > solv.depth_thresh || h1 > solv.depth_thresh)) return;

    float z0 = DEM[p0], z1 = DEM[p1];
    float q0 = Qyold[pq0];
    float dt = solv.dt, g = phys.g;

    float qvect = q0;
    if (use_friction != 0) {
        float qx_avg = (Qxold[j * qw + i] + Qxold[j * qw + i + 1] +
                        Qxold[(j + 1) * qw + i] + Qxold[(j + 1) * qw + i + 1]) * 0.25f;
        qvect = sqrt(q0 * q0 + qx_avg * qx_avg);
    }

    float fn = (manning != nullptr) ? 0.5f * (manning[p0] + manning[p1])
                                    : phys.manning;
    float y0 = z0 + h0, y1 = z1 + h1;
    float dh = y0 - y1;
    float Sf = -dh / geom.dy;

    float hflow = max(y0, y1) - max(z0, z1);
    hflow = clamp(hflow, 0.0f, solv.max_hflow);

    float qup = Qyold[pq0 - qw];
    float qdown = Qyold[pq0 + qw];
    if (j == 0 && qup == 0.0f) qup = q0;
    if (j == geom.ysz - 2 && qdown == 0.0f) qdown = q0;

    float Q = 0.0f;
    if (hflow > solv.depth_thresh) {
        float denom = 1.0f + g * dt * hflow * fn * fn * fabs(qvect)
                            / pow(hflow, 10.0f / 3.0f);
        Q = ((solv.theta * q0 + 0.5f * (1.0f - solv.theta) * (qup + qdown))
             - (g * dt * hflow * Sf)) / denom * geom.dy;
        if (Q * dh < 0.0f) {
            Q = (q0 - (g * dt * hflow * Sf)) / denom * geom.dy;
        }
    }
    Qy[pq0] = Q;

    if (voutput != 0) {
        Vy[pq0] = (Q != 0.0f) ? (Q / geom.dy / hflow) : 0.0f;
    }
}

// ------------------------------------------------------------- boundaries --

// Zero the fluxes on the four domain edges. Any prescribed boundary is applied
// afterwards by acc_boundary_conditions.
kernel void acc_zero_edge_flux(
    device float* Qx [[buffer(0)]],
    device float* Qy [[buffer(1)]],
    constant AccGeometry& geom [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int qw = geom.xsz + 1;
    if ((int)gid < geom.ysz) {
        int j = gid;
        Qx[j * qw] = 0.0f;                 // west
        Qx[j * qw + geom.xsz] = 0.0f;      // east
    }
    if ((int)gid < geom.xsz) {
        int i = gid;
        Qy[i] = 0.0f;                              // north
        Qy[i + geom.ysz * qw] = 0.0f;              // south
    }
}

// Apply the .bci boundary conditions on the domain edges.
// Index convention matches boundary_index_{w,e,n,s} in swe/boundary.cpp:
//   north: i            east: xsz + j
//   south: 2*xsz + ysz - i - 1     west: 2*xsz + 2*ysz - j - 1
// A positive prescribed discharge is an inflow on every edge.
// BCs() in boundary.cpp, acceleration branch, one thread per boundary cell
// (BCi = 0 .. 2*xsz + 2*ysz - 1, numbered as the CPU numbers them: north left to
// right, east top to bottom, south right to left, west bottom to top).
//
// Every edge face was zeroed by acc_zero_edge_flux; a face with no condition
// stays zero (closed). q0 is the face's flux from the previous step, in m^2/s,
// taken from Qxold/Qyold -- NOT from Qx/Qy, which have just been zeroed.
kernel void acc_boundary_conditions(
    device const float* H [[buffer(0)]],
    device const float* DEM [[buffer(1)]],
    device const float* manning [[buffer(2)]],
    device float* Qx [[buffer(3)]],
    device float* Qy [[buffer(4)]],
    device const int* bc_type [[buffer(5)]],
    device const float* bc_value [[buffer(6)]],
    constant AccGeometry& geom [[buffer(7)]],
    constant AccPhysical& phys [[buffer(8)]],
    constant AccSolver& solv [[buffer(9)]],
    device const float* Qxold [[buffer(10)]],
    device const float* Qyold [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    const int xsz = geom.xsz, ysz = geom.ysz, qw = xsz + 1;
    const int numBCs = 2 * xsz + 2 * ysz;
    const int BCi = (int)gid;
    if (BCi >= numBCs) return;
    const int bc = bc_type[BCi];
    if (bc == ACC_NONE) return;

    int p0, p1, sign, edge, face;
    bool xface;
    if (BCi <= xsz - 1) {                                   // north
        p0 = BCi; p1 = BCi + xsz; sign = -1; edge = 1;
        face = BCi; xface = false;
    } else if (BCi <= xsz + ysz - 1) {                      // east
        int j = BCi - xsz;
        p0 = xsz - 1 + j * xsz; p1 = xsz - 2 + j * xsz; sign = 1; edge = 2;
        face = xsz + j * qw; xface = true;
    } else if (BCi <= 2 * xsz + ysz - 1) {                  // south
        int i = 2 * xsz + ysz - 1 - BCi;
        p0 = i + (ysz - 1) * xsz; p1 = i + (ysz - 2) * xsz; sign = 1; edge = 3;
        face = i + ysz * qw; xface = false;
    } else {                                                // west
        int j = numBCs - 1 - BCi;
        p0 = j * xsz; p1 = 1 + j * xsz; sign = -1; edge = 4;
        face = j * qw; xface = true;
    }
    device float* Q = xface ? Qx : Qy;
    const float q0 = xface ? Qxold[face] : Qyold[face];
    const float g = phys.g, dt = solv.dt, dx = geom.dx;
    const float val = bc_value[BCi];
    const float fn = manning[p0];

    if (bc == ACC_FREE || bc == ACC_FREE6) {
        if (!(H[p0] > solv.depth_thresh)) return;
        float hflow = H[p0], h0 = H[p0], h1 = H[p1], z0 = DEM[p0], z1 = DEM[p1];
        float Sf;
        if (val < -0.999f) { float dh = z0 + h0 - z1 - h1; Sf = -dh / dx; }   // local slope
        else Sf = val;                                                        // given slope
        float qo = sign * (fabs(q0) + fabs(g * dt * hflow * Sf))
                   / (1.0f + g * dt * hflow * fn * fn * fabs(q0) / pow(hflow, 10.0f / 3.0f));
        float q = qo * dx;
        if (sign == -1 && q > 0.0f) q = 0.0f;      // a free boundary never flows in
        if (sign == 1 && q < 0.0f) q = 0.0f;
        Q[face] = q;
    } else if (bc == ACC_HFIX || bc == ACC_HVAR) {
        float hflow = max(H[p0], val - DEM[p0]);    // fills a dry edge cell too
        if (hflow < solv.depth_thresh) return;
        float h0 = val, h1 = H[p0], z1 = DEM[p0];
        float dh = h0 - z1 - h1;
        float Sf = (edge == 1 || edge == 4) ? -dh / dx : dh / dx;
        float qo = (q0 - g * dt * hflow * Sf)
                   / (1.0f + g * dt * hflow * fn * fn * fabs(q0) / pow(hflow, 10.0f / 3.0f));
        Q[face] = qo * dx;
    } else if (bc == ACC_QFIX || bc == ACC_QVAR) {
        Q[face] = -val * sign * dx;                 // val per unit width, m^2/s
    }
}

// ------------------------------------------------------------- dry check --
//
// Positivity limiter for cells that would be over-drained in one step.
//
// This MUST be two passes over two kernels, and the scaled flux MUST be
// written back into Qx/Qy, for the same reason the FV1 solver shares its
// interface fluxes through threadgroup memory: an interface flux belongs to
// two cells, so if one cell scales it privately the other still consumes the
// unscaled value and the difference is water created out of nothing.
//
// The CPU reference (DryCheck in util.cpp) gets this right -- it is a separate
// pass that writes *q1 *= WDweight back into the shared array. The CUDA
// reference (update_H in cuda/cuda_boundary.cu) does NOT: it scales local
// copies inside the update kernel, so each cell limits its own outflow while
// its neighbours still credit themselves the unlimited inflow. On a closed
// basin that creates 2.2 % of the domain volume in the first few steps.
//
// The CPU pass is order-dependent (its OpenMP pragma is commented out for
// exactly that reason), so it cannot be reproduced bit-for-bit in parallel.
// Instead each face is scaled by the weight of the cell it drains FROM, which
// is order-independent, deterministic, and gives every face exactly one
// well-defined scale factor that both adjacent cells then read.

kernel void acc_drycheck_weight(
    device float* W [[buffer(0)]],
    device const float* H [[buffer(1)]],
    device const float* Qx [[buffer(2)]],
    device const float* Qy [[buffer(3)]],
    constant AccGeometry& geom [[buffer(4)]],
    constant AccSolver& solv [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x, j = gid.y;
    if (i >= geom.xsz || j >= geom.ysz) return;

    int qw = geom.xsz + 1;
    float w = 1.0f;
    float h = H[j * geom.xsz + i];

    // Match the CPU: only cells above DepthThresh are limited at all.
    if (h > solv.depth_thresh) {
        float dV = solv.dt * (Qx[i + j * qw] - Qx[i + j * qw + 1]
                            + Qy[i + j * qw] - Qy[i + (j + 1) * qw]);
        float cv = h * geom.dx * geom.dy;
        if (cv + dV < 0.0f && dV < 0.0f) w = -cv / dV;   // in [0,1)
    }
    W[j * geom.xsz + i] = w;
}

kernel void acc_drycheck_scale(
    device float* Qx [[buffer(0)]],
    device float* Qy [[buffer(1)]],
    device const float* W [[buffer(2)]],
    constant AccGeometry& geom [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x, j = gid.y;
    int qw = geom.xsz + 1;

    // x-faces: i runs 0..xsz, Qx>0 flows +x so it drains cell (i-1,j).
    if (i <= geom.xsz && j < geom.ysz) {
        int f = i + j * qw;
        float q = Qx[f];
        if (q > 0.0f && i >= 1)          Qx[f] = q * W[j * geom.xsz + (i - 1)];
        else if (q < 0.0f && i <= geom.xsz - 1) Qx[f] = q * W[j * geom.xsz + i];
    }

    // y-faces: j runs 0..ysz, Qy>0 flows +j so it drains cell (i,j-1).
    if (i < geom.xsz && j <= geom.ysz) {
        int f = i + j * qw;
        float q = Qy[f];
        if (q > 0.0f && j >= 1)          Qy[f] = q * W[(j - 1) * geom.xsz + i];
        else if (q < 0.0f && j <= geom.ysz - 1) Qy[f] = q * W[j * geom.xsz + i];
    }
}

// ------------------------------------------- CPU-exact dry check (default) --
//
// DryCheck() in util.cpp visits cells in raster order and scales a draining
// cell's outflows IN PLACE. So when cell (i,j) computes its weight, the faces
// its left and upper neighbours drain into it already carry THOSE neighbours'
// weights, while its right and lower faces do not. Every face is scaled only by
// the cell it drains from, so the weights form a recurrence that runs left to
// right and top to bottom. acc_drycheck_weight (above) ignores that order;
// this kernel honours it.
//
// It is a Jacobi iteration on that recurrence. Start with every weight 1, and
// recompute each cell's weight from its neighbours' previous weights. The
// dependencies only point left and up, so the result is exact once the
// iteration stops changing, after as many passes as the longest chain of
// consecutively drained cells (usually one or two). acc_drycheck_scale then
// applies the converged weights. The arithmetic is DryCheck()'s, operation for
// operation, so the fluxes are bit-identical to the CPU's.
kernel void acc_drycheck_cpu_iter(
    device const float* H [[buffer(0)]],
    device const float* Qx [[buffer(1)]],
    device const float* Qy [[buffer(2)]],
    device const float* Win [[buffer(3)]],
    device float* Wout [[buffer(4)]],
    constant AccGeometry& geom [[buffer(5)]],
    constant AccSolver& solv [[buffer(6)]],
    device atomic_uint* changed [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = (int)gid.x, j = (int)gid.y;
    if (i >= geom.xsz || j >= geom.ysz) return;
    int qw = geom.xsz + 1, p = j * geom.xsz + i;
    float w = 1.0f;
    float h = H[p];
    if (h > solv.depth_thresh) {
        float q1 = Qx[i + j * qw];           // west face
        float q2 = Qx[i + j * qw + 1];       // east face
        float q3 = Qy[i + j * qw];           // north face
        float q4 = Qy[i + (j + 1) * qw];     // south face
        if (i > 0 && q1 > 0.0f) q1 *= Win[p - 1];          // drained in from the left
        if (j > 0 && q3 > 0.0f) q3 *= Win[p - geom.xsz];   // drained in from above
        float dV = solv.dt * (q1 - q2 + q3 - q4);
        float cv = h * (geom.dx * geom.dy);
        if (cv + dV < 0.0f) w = -cv / dV;
    }
    if (w != Win[p]) atomic_store_explicit(changed, 1u, memory_order_relaxed);
    Wout[p] = w;
}

kernel void acc_fill(
    device float* a [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) a[gid] = value;
}

// ------------------------------------------ CUDA-compatible timestep (verif.) --
//
// cuda/acc/cuda_acc_solver.cu, update_dt_per_element, reproduced for the
// CUDA-compatible verification mode only. Two ways it differs from the CPU:
//  * the per-cell dt is a running MINIMUM while the cell stays wet, so the
//    global step can shrink but never grow back after the flood peak;
//  * the field starts at zero, and a zero entry becomes max_dt, not the CFL
//    value, so the FIRST step is initial_tstep whatever CFL allows.
kernel void acc_update_dt_cuda(
    device const float* H [[buffer(0)]],
    device float* dt_field [[buffer(1)]],
    constant AccGeometry& geom [[buffer(2)]],
    constant AccPhysical& phys [[buffer(3)]],
    constant AccSolver& solv [[buffer(4)]],
    volatile device atomic_uint* min_dt_atomic [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = (int)gid.x, j = (int)gid.y;
    float d = solv.max_dt;
    if (i < geom.xsz && j < geom.ysz) {
        int p = j * geom.xsz + i;
        float h = H[p];
        if (h > solv.depth_thresh) {
            float dt_ = solv.cfl * geom.dx / sqrt(phys.g * h);
            float cur = dt_field[p];
            d = (fabs(cur - 0.0f) < 1e-6f) ? solv.max_dt : min(dt_, cur);
        }
        dt_field[p] = d;
    }
    d = simd_min(d);
    if (simd_is_first())
        atomic_fetch_min_explicit(min_dt_atomic, as_type<uint>(d), memory_order_relaxed);
}

// ------------------------------------------------------------ point sources --
//
// The P lines of a .bci file, applied exactly where UpdateH() applies them on
// the CPU: QFIX/QVAR add their volume before the continuity update (phase 0),
// HFIX/HVAR impose their stage after it (phase 1). `value` is the per-unit-width
// discharge (m^2/s) or the water-surface elevation (m) at this step, evaluated
// on the host. Each source writes the discharge it imposed (m^3/s) to q_out for
// the mass balance. There are few sources, so it is one thread each.
kernel void acc_point_sources(
    device float* H [[buffer(0)]],
    device const float* DEM [[buffer(1)]],
    device const int* ps_cell [[buffer(2)]],
    device const int* ps_type [[buffer(3)]],
    device const float* ps_value [[buffer(4)]],
    device float* q_out [[buffer(5)]],
    constant AccGeometry& geom [[buffer(6)]],
    constant AccSolver& solv [[buffer(7)]],
    constant int& n_ps [[buffer(8)]],
    constant int& phase [[buffer(9)]],
    uint k [[thread_position_in_grid]])
{
    if ((int)k >= n_ps) return;
    int p = ps_cell[k], t = ps_type[k];
    float dA = geom.dx * geom.dy;
    if (phase == 0 && (t == ACC_QFIX || t == ACC_QVAR)) {
        H[p] += ps_value[k] * geom.dx * solv.dt / dA;
        q_out[k] = ps_value[k] * geom.dx;
    } else if (phase == 1 && (t == ACC_HFIX || t == ACC_HVAR)) {
        float himp = ps_value[k] - DEM[p];
        if (himp < 0.0f) himp = 0.0f;
        q_out[k] = (himp - H[p]) * dA / solv.dt;
        H[p] = himp;
    }
}

// ------------------------------------------------------------- continuity --

// dV = dt*(Qx_west - Qx_east + Qy_north - Qy_south); h += dV/(dx*dy).
// Mirrors update_H in cuda/cuda_boundary.cu, including the wet/dry weighting
// that scales outgoing fluxes when a cell would otherwise go negative.
kernel void acc_update_h(
    device float* H [[buffer(0)]],
    device const float* Qx [[buffer(1)]],
    device const float* Qy [[buffer(2)]],
    constant AccGeometry& geom [[buffer(3)]],
    constant AccSolver& solv [[buffer(4)]],
    device float* clip_part [[buffer(5)]],
    constant int& cuda_drycheck [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgpg [[threadgroups_per_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]])
{
    threadgroup float scratch[32];
    int i = (int)gid.x, j = (int)gid.y;
    float created = 0.0f;

    if (i < geom.xsz && j < geom.ysz) {
        int qw = geom.xsz + 1;
        float qx0 = Qx[i + j * qw];
        float qx1 = Qx[i + j * qw + 1];
        float qy0 = Qy[i + j * qw];
        float qy1 = Qy[i + (j + 1) * qw];
        float h = H[j * geom.xsz + i];

        float h_new;
        if (cuda_drycheck == 0) {
            // CPU (default): fluxes arrive already limited by the dry-check
            // pass (acc_drycheck_cpu_iter + acc_drycheck_scale), exactly as
            // DryCheck() leaves them for UpdateH() in iterateq.cpp.
            float dV = solv.dt * (qx0 - qx1 + qy0 - qy1);
            h_new = h + dV / (geom.dx * geom.dy);
        } else {
            // CUDA-compatible verification mode only: update_H in
            // cuda/cuda_boundary.cu, including its one-sided limiter, which
            // scales the donor's outflow in LOCAL copies and so creates water
            // (see the note above acc_drycheck_weight). cuda_drycheck = 1:
            // limiter on; 2: CUDA arithmetic with the limiter off.
            float cv = h * (geom.dx * geom.dx);
            float dV = solv.dt * (qx0 - qx1 + qy0 - qy1);
            if (cuda_drycheck == 1 && cv + dV < 0.0f) {
                float w = -cv / dV;
                if (qx0 < 0.0f) qx0 *= w;
                if (qx1 > 0.0f) qx1 *= w;
                if (qy0 < 0.0f) qy0 *= w;
                if (qy1 > 0.0f) qy1 *= w;
            }
            dV = solv.dt * (qx0 - qx1 + qy0 - qy1);
            h_new = h + dV / (geom.dx * geom.dx);
        }
        if (h_new < 0.0f) {
            // Clipping creates water. Counted for diagnostics only: as on the
            // CPU, it shows up in the Verror column.
            created = -h_new * (geom.dx * geom.dy);
            h_new = 0.0f;
        }
        H[j * geom.xsz + i] = h_new;
    }
    store_tg_sum(created, clip_part, tgid.y * tgpg.x + tgid.x,
                 lid, lane, sg, nsg, scratch);
}

// Convert the stored fluxes from m3/s back to the m2/s that the next
// timestep's q0/qup/qdown terms expect (update_flow_variables_x/y in CUDA).
kernel void acc_store_qold(
    device const float* Qx [[buffer(0)]],
    device const float* Qy [[buffer(1)]],
    device float* Qxold [[buffer(2)]],
    device float* Qyold [[buffer(3)]],
    constant AccGeometry& geom [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x, j = gid.y;
    int qw = geom.xsz + 1;
    if (i > geom.xsz || j > geom.ysz) return;
    int p = j * qw + i;
    Qxold[p] = Qx[p] / geom.dx;
    Qyold[p] = Qy[p] / geom.dy;
}

// ------------------------------------------------------ sources and sinks --

kernel void acc_uniform_rain(
    device float* H [[buffer(0)]],
    device const float* DEM [[buffer(1)]],
    constant AccGeometry& geom [[buffer(2)]],
    constant AccSolver& solv [[buffer(3)]],
    constant float& rain_rate [[buffer(4)]],
    device float* rain_part [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgpg [[threadgroups_per_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]])
{
    threadgroup float scratch[32];
    int i = (int)gid.x, j = (int)gid.y;
    float added = 0.0f;

    if (i < geom.xsz && j < geom.ysz) {
        int p = j * geom.xsz + i;
        if (!(fabs(DEM[p] - solv.nodata_elev) < 1e-6f)) {
            float inc = max(-H[p], rain_rate * solv.dt);
            H[p] += inc;
            added = inc * geom.dx * geom.dy;
        }
    }
    store_tg_sum(added, rain_part, tgid.y * tgpg.x + tgid.x,
                 lid, lane, sg, nsg, scratch);
}

// Dynamic (NetCDF) rainfall, sampled with the same nearest-tile rule as
// DynamicRain::rate_at_cell in rain/rain.tpp.
kernel void acc_dynamic_rain(
    device float* H [[buffer(0)]],
    device const float* DEM [[buffer(1)]],
    device const float* rain [[buffer(2)]],
    constant AccGeometry& geom [[buffer(3)]],
    constant AccGeometry& rgeom [[buffer(4)]],
    constant AccSolver& solv [[buffer(5)]],
    device float* rain_part [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgpg [[threadgroups_per_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]])
{
    threadgroup float scratch[32];
    int i = (int)gid.x, j = (int)gid.y;
    float added = 0.0f;

    if (i < geom.xsz && j < geom.ysz) {
        int p = j * geom.xsz + i;
        if (!(fabs(DEM[p] - solv.nodata_elev) < 1e-6f)) {
            int ti = (int)floor((geom.blx - rgeom.blx + (float)i * geom.dx) / rgeom.dx);
            float top_gap = geom.tly - rgeom.tly;
            int tj = (int)floor(((float)j - top_gap / geom.dy) * geom.dy / rgeom.dy);
            if (ti >= 0 && ti < rgeom.xsz && tj >= 0 && tj < rgeom.ysz) {
                float rate = rain[tj * rgeom.xsz + ti];
                if (rate > 0.0f) {
                    float inc = rate * solv.dt;
                    H[p] += inc;
                    added = inc * geom.dx * geom.dy;
                }
            }
        }
    }
    store_tg_sum(added, rain_part, tgid.y * tgpg.x + tgid.x,
                 lid, lane, sg, nsg, scratch);
}

kernel void acc_drain_nodata(
    device float* H [[buffer(0)]],
    device const float* DEM [[buffer(1)]],
    constant AccGeometry& geom [[buffer(2)]],
    constant AccSolver& solv [[buffer(3)]],
    device float* drain_part [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgpg [[threadgroups_per_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]])
{
    threadgroup float scratch[32];
    int i = (int)gid.x, j = (int)gid.y;
    float drained = 0.0f;

    if (i < geom.xsz && j < geom.ysz) {
        int p = j * geom.xsz + i;
        if (fabs(DEM[p] - solv.nodata_elev) < 1.0f) {
            if (H[p] > 0.0f) drained = H[p] * geom.dx * geom.dy;
            H[p] = 0.0f;
        }
    }
    store_tg_sum(drained, drain_part, tgid.y * tgpg.x + tgid.x,
                 lid, lane, sg, nsg, scratch);
}

// Discharge across the domain edge, as BoundaryFlux() in boundary.cpp computes
// it: Qy is positive north->south and Qx west->east, so on the north and west
// edges a positive flux is inflow and on the south and east edges it is
// outflow. Run after the dry-check scaling so the fluxes measured are the ones
// update_h applies. Rates in m^3/s; the host multiplies by dt.
//
// One thread per edge face, 1-D, faces numbered north (xsz), south (xsz),
// west (ysz), east (ysz).
kernel void acc_boundary_flux(
    device const float* Qx [[buffer(0)]],
    device const float* Qy [[buffer(1)]],
    constant AccGeometry& geom [[buffer(2)]],
    device float* in_part [[buffer(3)]],
    device float* out_part [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]])
{
    threadgroup float scratch_in[32];
    threadgroup float scratch_out[32];
    int nx = geom.xsz, ny = geom.ysz, qw = geom.xsz + 1;
    int k = (int)gid;
    float q_in = 0.0f, q_out = 0.0f;

    if (k < nx) {                                   // north: Qy row 0
        float q = Qy[k];
        if (q > 0.0f) q_in = q; else q_out = -q;
    } else if (k < 2 * nx) {                        // south: Qy row ysz
        float q = Qy[(k - nx) + ny * qw];
        if (q > 0.0f) q_out = q; else q_in = -q;
    } else if (k < 2 * nx + ny) {                   // west: Qx column 0
        float q = Qx[(k - 2 * nx) * qw];
        if (q > 0.0f) q_in = q; else q_out = -q;
    } else if (k < 2 * nx + 2 * ny) {               // east: Qx column xsz
        float q = Qx[nx + (k - 2 * nx - ny) * qw];
        if (q > 0.0f) q_out = q; else q_in = -q;
    }
    store_tg_sum(q_in, in_part, tgid, lid, lane, sg, nsg, scratch_in);
    store_tg_sum(q_out, out_part, tgid, lid, lane, sg, nsg, scratch_out);
}

// Running maximum depth and the associated timings, as in update_max_field_ACC.
kernel void acc_update_max(
    device const float* H [[buffer(0)]],
    device float* maxH [[buffer(1)]],
    device float* maxHtm [[buffer(2)]],
    device float* totalHtm [[buffer(3)]],
    device float* initHtm [[buffer(4)]],
    constant AccGeometry& geom [[buffer(5)]],
    constant AccSolver& solv [[buffer(6)]],
    constant float& t_now [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    int i = gid.x, j = gid.y;
    if (i >= geom.xsz || j >= geom.ysz) return;
    int p = j * geom.xsz + i;

    float h = H[p];
    if (h > solv.depth_thresh) {
        if (initHtm[p] == -9999.0f) initHtm[p] = t_now / 3600.0f;
        totalHtm[p] += solv.dt / 3600.0f;
        if (h > maxH[p]) {
            maxH[p] = h;
            maxHtm[p] = t_now / 3600.0f;
        }
    }
}
