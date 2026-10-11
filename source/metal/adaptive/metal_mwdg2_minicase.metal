#include <metal_stdlib>
using namespace metal;

constant float mini_sqrt3 = 1.7320508075688772f;

inline uint mini_dilate(uint v) {
    v = (v | (v << 8)) & 0x00ff00ffu;
    v = (v | (v << 4)) & 0x0f0f0f0fu;
    v = (v | (v << 2)) & 0x33333333u;
    return (v | (v << 1)) & 0x55555555u;
}

inline uint4 mini_geometry(uint node, uint maxLevel) {
    uint start = 0, count = 1, level = 0;
    while (node >= start + count) { start += count; count *= 4; ++level; }
    uint code = node - start, span = 1u << (maxLevel - level);
    uint x = 0, y = 0;
    for (uint b = 0; b < 16; ++b) {
        x |= ((code >> (2*b)) & 1u) << b;
        y |= ((code >> (2*b + 1)) & 1u) << b;
    }
    return uint4(x * span, y * span, span, level);
}

// Restrict the updated adaptive DG2 polynomial exactly onto every finest
// cell. A linear polynomial remains linear under affine cell restriction.
// Persistent MWDG2 state remains eta=h+z,qx,qy,z throughout the update.
// The bed (modes 9-11) is not written: the finest level keeps the input bed,
// as CUDA's scale coefficients do, so coarsening never smooths it.
kernel void mwdg2_mini_expand_finest(device const float *hydro [[buffer(0)]],
                                     device const uint *owners [[buffer(1)]],
                                     device float *scales [[buffer(2)]],
                                     constant uint2 &layout [[buffer(3)]],
                                     uint pixel [[thread_position_in_grid]]) {
    uint side = 1u << layout.y;
    uint cells = side * side;
    if (pixel >= cells) return;
    uint x = pixel % side, y = pixel / side;
    uint code = mini_dilate(x) | (mini_dilate(y) << 1);
    uint owner = owners[code];
    uint4 cell = mini_geometry(owner, layout.y);
    float xi = 2.0f * (float(x) + 0.5f - float(cell.x)) / float(cell.z) - 1.0f;
    float yi = 2.0f * (float(y) + 0.5f - float(cell.y)) / float(cell.z) - 1.0f;
    float invSpan = 1.0f / float(cell.z);
    uint outNode = (cells - 1) / 3 + code;
    for (uint mode = 0; mode < 3; ++mode) {
        uint p = owner * 12;
        uint q = outNode * 12;
        float eta = hydro[p] + mini_sqrt3 * (hydro[p+1] * xi + hydro[p+2] * yi);
        if (mode == 0) {
            scales[q] = eta;
            scales[q+3] = hydro[p+3] + mini_sqrt3 * (hydro[p+4] * xi + hydro[p+5] * yi);
            scales[q+6] = hydro[p+6] + mini_sqrt3 * (hydro[p+7] * xi + hydro[p+8] * yi);
        } else if (mode == 1) {
            scales[q+1] = hydro[p+1] * invSpan;
            scales[q+4] = hydro[p+4] * invSpan;
            scales[q+7] = hydro[p+7] * invSpan;
        } else {
            scales[q+2] = hydro[p+2] * invSpan;
            scales[q+5] = hydro[p+5] * invSpan;
            scales[q+8] = hydro[p+8] * invSpan;
        }
    }
}

kernel void mwdg2_mini_cfl(device const float *hydro [[buffer(0)]],
                           device const uint *active [[buffer(1)]],
                           device float *timesteps [[buffer(2)]],
                           constant uint2 &layout [[buffer(3)]],
                           constant float4 &physics [[buffer(4)]],
                           constant float2 &control [[buffer(5)]],
                           uint index [[thread_position_in_grid]]) {
    if (index >= layout.x) return;
    uint node = active[index];
    uint4 cell = mini_geometry(node, layout.y);
    uint p = node * 12;
    float h = max(hydro[p] - hydro[p+9], 0.0f);
    if (h <= control.y) { timesteps[index] = INFINITY; return; }
    float ux = hydro[p+3] / h, uy = hydro[p+6] / h;
    float c = sqrt(physics.z * h);
    float dx = physics.x * float(cell.z), dy = physics.y * float(cell.z);
    float tx = dx / (abs(ux) + c), ty = dy / (abs(uy) + c);
    timesteps[index] = control.x * min(tx, ty);
}

// Cell-centre raster output in internal south-to-north row order:
// planes are depth, free-surface elevation, qx, qy.
kernel void mwdg2_mini_project_output(device const float *scales [[buffer(0)]],
                                      device float *raster [[buffer(1)]],
                                      constant uint2 &layout [[buffer(2)]],
                                      uint pixel [[thread_position_in_grid]]) {
    uint side = 1u << layout.y, cells = side * side;
    if (pixel >= cells) return;
    uint x = pixel % side, y = pixel / side;
    uint code = mini_dilate(x) | (mini_dilate(y) << 1);
    uint node = (cells - 1) / 3 + code;
    uint p = node * 12;
    raster[pixel] = scales[p] - scales[p+9];
    raster[cells + pixel] = scales[p];
    raster[2*cells + pixel] = scales[p+3];
    raster[3*cells + pixel] = scales[p+6];
}

// End-of-step expansion used for remeshing. CUDA refines a leaf by decoding
// with zero details, which gives the children the parent's water level over
// their own (finest) beds: on a bed that varies inside the leaf some children
// get negative depths and others water that was not there (+/-10 m on the
// Alzette slopes with HWFV1). Here a leaf keeps CUDA's restriction when every
// finest cell stays at or below its water level, and otherwise gives every
// finest cell the leaf's mean depth over its own bed (same volume, no negative
// depths). Pass 1 flags such leaves; pass 2 writes the finest cells.
kernel void mwdg2_expand_flags(device const float *hydro [[buffer(0)]],
                               device const uint *owners [[buffer(1)]],
                               device const float *scales [[buffer(2)]],
                               device atomic_uint *flags [[buffer(3)]],
                               constant uint2 &layout [[buffer(4)]],
                               uint pixel [[thread_position_in_grid]]) {
    uint side = 1u << layout.y;
    uint cells = side * side;
    if (pixel >= cells) return;
    uint x = pixel % side, y = pixel / side;
    uint code = mini_dilate(x) | (mini_dilate(y) << 1);
    uint owner = owners[code];
    uint4 cell = mini_geometry(owner, layout.y);
    if (cell.z == 1) return;
    float xi = 2.0f * (float(x) + 0.5f - float(cell.x)) / float(cell.z) - 1.0f;
    float yi = 2.0f * (float(y) + 0.5f - float(cell.y)) / float(cell.z) - 1.0f;
    uint p = owner * 12;
    float eta = hydro[p] + mini_sqrt3 * (hydro[p+1] * xi + hydro[p+2] * yi);
    float z = scales[((cells - 1) / 3 + code) * 12 + 9];
    if (eta < z) atomic_store_explicit(&flags[owner], 1u, memory_order_relaxed);
}

kernel void mwdg2_expand_finest_positive(device const float *hydro [[buffer(0)]],
                                         device const uint *owners [[buffer(1)]],
                                         device float *scales [[buffer(2)]],
                                         device const uint *flags [[buffer(3)]],
                                         constant uint2 &layout [[buffer(4)]],
                                         uint pixel [[thread_position_in_grid]]) {
    uint side = 1u << layout.y;
    uint cells = side * side;
    if (pixel >= cells) return;
    uint x = pixel % side, y = pixel / side;
    uint code = mini_dilate(x) | (mini_dilate(y) << 1);
    uint owner = owners[code];
    uint4 cell = mini_geometry(owner, layout.y);
    float xi = 2.0f * (float(x) + 0.5f - float(cell.x)) / float(cell.z) - 1.0f;
    float yi = 2.0f * (float(y) + 0.5f - float(cell.y)) / float(cell.z) - 1.0f;
    float invSpan = 1.0f / float(cell.z);
    uint p = owner * 12, q = ((cells - 1) / 3 + code) * 12;
    if (flags[owner]) {
        float depth = hydro[p] - hydro[p+9];      // the leaf's mean depth
        scales[q] = scales[q+9] + depth;
        scales[q+1] = scales[q+10];
        scales[q+2] = scales[q+11];
    } else {
        scales[q] = hydro[p] + mini_sqrt3 * (hydro[p+1] * xi + hydro[p+2] * yi);
        scales[q+1] = hydro[p+1] * invSpan;
        scales[q+2] = hydro[p+2] * invSpan;
    }
    scales[q+3] = hydro[p+3] + mini_sqrt3 * (hydro[p+4] * xi + hydro[p+5] * yi);
    scales[q+6] = hydro[p+6] + mini_sqrt3 * (hydro[p+7] * xi + hydro[p+8] * yi);
    scales[q+4] = hydro[p+4] * invSpan; scales[q+5] = hydro[p+5] * invSpan;
    scales[q+7] = hydro[p+7] * invSpan; scales[q+8] = hydro[p+8] * invSpan;
}
