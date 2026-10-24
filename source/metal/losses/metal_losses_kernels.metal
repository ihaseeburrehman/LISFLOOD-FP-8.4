//
// LISFLOOD-FP Metal: floodplain infiltration and evaporation (8.4)
// The rules of cuda/cuda_losses.cu, for the Metal FV1, FV2, DG2 and ACC
// solvers:
//  - constant rate (uniform or per cell) plus evaporation: every wet cell
//    (deeper than DepthThresh, not NoData) loses (rate + evaporation) x dt of
//    depth, never below zero;
//  - Green-Ampt: f = Ks (1 + (psi + h) dtheta / F), advanced with implicit
//    Euler, limited by the water present and, optionally, by the soil depth;
//  - Horton, cumulative form: F = fc te + (f0 - fc)/k (1 - exp(-k te)); each
//    step takes F(te + dt) - F(te), limited by the water present (the
//    Green-Ampt buffers hold fc, f0 and k: ks, psi, dtheta).
// Depth is single precision and the loss per step is often close to its
// resolution, so each cell carries the part of the loss that rounding dropped
// to its next step (compensated subtraction), as in CUDA.
//
// Metal GPUs have no double precision, which CUDA uses for F. Here F is kept
// as a pair of floats (hi + lo, updated with an exact two-sum), and the step
// dF is computed in a form without cancellation (see greenampt_step).
//

#include <metal_stdlib>
using namespace metal;

struct LossParams {
    int greenampt;      // 0: constant rate, 1: Green-Ampt, 2: Horton
    int has_rate_grid;  // constant rate: per-cell rates
    int has_fmax;       // Green-Ampt: soil-depth cap
    int has_q;          // mode 0: scale HU, HV with the depth
    int mode;           // 0: depth (and discharge); 1: DG2 coefficient planes
    int xsz, ysz, pitch, i0, j0;  // cell (i, j) is at (j + j0) * pitch + i + i0
    int plane_n;        // DG2: floats per plane
    int ngx;            // threadgroups along x (for the partial sums)
    float uniform_rate; // m/s
    float evap_rate;    // m/s
    float dt;
    float depth_thresh;
    float nodata;       // DEM value of NoData cells
    float face;         // DG2: a face value is mean +- face x slope (sqrt 3)
};

// depth after losing rate x dt plus the residual (not below zero), or h if
// the cell is dry
inline float depth_after_loss(float h, float rate, float dt, float th,
                              thread float& residual)
{
    if (h <= th || rate <= 0.0f) { residual = 0.0f; return h; }
    float wanted = rate * dt + residual;
    float h_new = h - wanted;
    if (h_new <= 0.0f) { residual = 0.0f; return 0.0f; }
    residual = wanted - (h - h_new);   // h - h_new is exact (Sterbenz)
    return h_new;
}

// Implicit Euler for Green-Ampt: F_new = F + k (1 + M / F_new), k = Ks dt,
// M = (psi + h) dtheta, i.e. F_new = (a + sqrt(a^2 + 4 k M)) / 2, a = F + k.
// dF = F_new - F written without the cancellation of F_new - F in float:
//   F <= k:  dF = ((k - F) + sqrt(S)) / 2
//   F >  k:  dF = 2 (F k + k M) / (sqrt(S) + F - k),   S = (F + k)^2 + 4 k M
inline float greenampt_step(float F, float k, float M)
{
    float S = (F + k) * (F + k) + 4.0f * k * M;
    float sq = sqrt(S);
    if (F <= k) return 0.5f * ((k - F) + sq);
    return 2.0f * (F * k + k * M) / (sq + (F - k));
}

// 1 - exp(-x) for x >= 0 without cancellation at small x
inline float one_minus_exp(float x)
{
    if (x < 1e-3f) return x * (1.0f - x * (0.5f - x * (1.0f / 6.0f)));
    return 1.0f - exp(-x);
}

// Horton, cumulative form (cuda horton_capacity): the depth the soil can
// take in a step of dt after F has infiltrated. te solves
// F = fc te + A (1 - exp(-k te)), A = (f0 - fc)/k, by Newton from below.
inline float horton_capacity(float F, float f0, float fc, float k, float dt)
{
    if (k <= 0.0f || f0 <= fc) return max(f0, fc) * dt;
    float A = (f0 - fc) / k;
    if (fc <= 0.0f && F >= A) return 0.0f;
    float te = 0.0f;
    if (F > 0.0f) {
        te = F / f0;
        for (int n = 0; n < 30; n++) {
            float e = exp(-k * te);
            float step = (F - (fc * te + A * one_minus_exp(k * te))) / (fc + (f0 - fc) * e);
            te += step;
            if (abs(step) <= 1e-6f * (te + 1.0f)) break;
        }
    }
    return fc * dt + A * exp(-k * te) * one_minus_exp(k * dt);
}

// F (= hi + lo) += x, exactly up to the rounding of the pair
inline void add_F(device float* Fhi, device float* Flo, int c, float x)
{
    float hi = Fhi[c], lo = Flo[c];
    float s = hi + x;
    float bp = s - hi;
    float err = (hi - (s - bp)) + (x - bp);
    lo += err;
    float h2 = s + lo;
    Flo[c] = lo - (h2 - s);
    Fhi[c] = h2;
}

inline float depth_after_greenampt(constant LossParams& p, int c, float h,
                                   device float* residual,
                                   device float* Fhi, device float* Flo,
                                   device const float* ks, device const float* psi,
                                   device const float* dtheta, device const float* fmax)
{
    if (h <= 0.0f) { residual[c] = 0.0f; return h; }
    float F = Fhi[c] + Flo[c];
    float k = ks[c] * p.dt;
    float dF = 0.0f;
    if (p.greenampt == 2) {
        dF = horton_capacity(F, psi[c], ks[c], dtheta[c], p.dt);  // f0, fc, k
    } else if (ks[c] > 0.0f) {
        dF = greenampt_step(F, k, (psi[c] + h) * dtheta[c]);
        if (p.has_fmax) dF = min(dF, max(fmax[c] - F, 0.0f));
    }
    float e = p.evap_rate * p.dt;
    if (dF + e <= 0.0f) { residual[c] = 0.0f; return h; }
    float wanted = (dF + e) + residual[c];
    float h_new = h - wanted;
    if (h_new <= 0.0f) {
        // not enough water: infiltration takes it first, evaporation the rest
        add_F(Fhi, Flo, c, min(dF, h));
        residual[c] = 0.0f;
        return 0.0f;
    }
    residual[c] = wanted - (h - h_new);
    add_F(Fhi, Flo, c, dF);
    return h_new;
}

kernel void apply_losses(
    device float* U [[buffer(0)]],        // H (mode 0) or the DG2 planes (mode 1)
    device float* HU [[buffer(1)]],
    device float* HV [[buffer(2)]],
    device const float* DEM [[buffer(3)]],
    constant LossParams& p [[buffer(4)]],
    device const float* rate [[buffer(5)]],
    device float* residual [[buffer(6)]],
    device float* Fhi [[buffer(7)]],
    device float* Flo [[buffer(8)]],
    device const float* ks [[buffer(9)]],
    device const float* psi [[buffer(10)]],
    device const float* dtheta [[buffer(11)]],
    device const float* fmax [[buffer(12)]],
    device float* partial [[buffer(13)]],   // removed depth x 1, per threadgroup
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tidx [[thread_index_in_threadgroup]],
    uint sidx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sums[32];
    float removed = 0.0f;
    int i = (int)gid.x, j = (int)gid.y;
    if (i < p.xsz && j < p.ysz) {
        int c = (j + p.j0) * p.pitch + i + p.i0;
        if (abs(DEM[c] - p.nodata) >= 1e-6f) {
            float h = U[c];
            float h_new;
            if (p.greenampt) {
                h_new = depth_after_greenampt(p, c, h, residual, Fhi, Flo, ks, psi, dtheta, fmax);
            } else {
                float r = (p.has_rate_grid ? rate[c] : p.uniform_rate) + p.evap_rate;
                float res = residual[c];
                h_new = depth_after_loss(h, r, p.dt, p.depth_thresh, res);
                residual[c] = res;
            }
            if (h_new != h) {
                U[c] = h_new;
                removed = h - h_new;
                if (p.mode == 0 && p.has_q) {
                    if (h_new > p.depth_thresh) {
                        float s = h_new / h;      // keep the velocity
                        HU[c] *= s; HV[c] *= s;
                    } else {
                        HU[c] = 0.0f; HV[c] = 0.0f;
                    }
                } else if (p.mode == 1) {
                    // DG2 planes: H H1x H1y HU HU1x HU1y HV HV1x HV1y
                    int N = p.plane_n;
                    bool faces_dry = p.face * abs(U[N + c]) >= h_new ||
                                     p.face * abs(U[2 * N + c]) >= h_new;
                    if (h_new > p.depth_thresh && !faces_dry) {
                        float s = h_new / h;
                        for (int k = 3; k < 9; k++) U[k * N + c] *= s;
                    } else if (h_new > p.depth_thresh) {
                        // a face would become dry: the cell becomes first order
                        float s = h_new / h;
                        U[N + c] = 0.0f; U[2 * N + c] = 0.0f;
                        U[3 * N + c] *= s; U[4 * N + c] = 0.0f; U[5 * N + c] = 0.0f;
                        U[6 * N + c] *= s; U[7 * N + c] = 0.0f; U[8 * N + c] = 0.0f;
                    } else {
                        for (int k = 1; k < 9; k++) U[k * N + c] = 0.0f;
                    }
                }
            }
        }
    }
    // sum of the removed depth over the threadgroup
    float s = simd_sum(removed);
    if (lane == 0) sums[sidx] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tidx == 0) {
        float t = 0.0f;
        uint nsg = (16 * 16 + 31) / 32;
        for (uint k = 0; k < nsg; k++) t += sums[k];
        partial[tgid.y * (uint)p.ngx + tgid.x] = t;
    }
}
