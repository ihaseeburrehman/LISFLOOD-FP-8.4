#include <metal_stdlib>
using namespace metal;
#include "mwdg2_transform.h"

inline uint morton_dilate(uint value) {
    value = (value | (value << 8)) & 0x00ff00ffu;
    value = (value | (value << 4)) & 0x0f0f0f0fu;
    value = (value | (value << 2)) & 0x33333333u;
    return (value | (value << 1)) & 0x55555555u;
}

inline uint morton_contract(uint value) {
    value &= 0x55555555u;
    value = (value | (value >> 1)) & 0x33333333u;
    value = (value | (value >> 2)) & 0x0f0f0f0fu;
    value = (value | (value >> 4)) & 0x00ff00ffu;
    return (value | (value >> 8)) & 0x0000ffffu;
}

inline uint4 node_geometry(uint node, uint maximum_level) {
    uint start = 0, count = 1, level = 0;
    while (node >= start + count) { start += count; count *= 4; ++level; }
    uint code = node - start, span = 1u << (maximum_level - level);
    return uint4(morton_contract(code) * span,
                 morton_contract(code >> 1) * span, span, level);
}

inline uint owner_at(device const uint *owners, int x, int y, uint side) {
    if (x < 0 || y < 0 || x >= int(side) || y >= int(side)) return 0xffffffffu;
    return owners[morton_dilate(uint(x)) | (morton_dilate(uint(y)) << 1)];
}

// Grading proposals use atomics because multiple finest interfaces can
// request the same coarse leaf. The owner map is immutable in this pass.
// Rebuild ownership between passes; at most L passes are needed.
kernel void mwdg2_grade(device const uint *owners [[buffer(0)]],
                        device atomic_uint *split [[buffer(1)]],
                        constant uint &maximum_level [[buffer(2)]],
                        uint code [[thread_position_in_grid]]) {
    uint side = 1u << maximum_level;
    if (code >= side*side) return;
    uint owner = owners[code];
    uint level = node_geometry(owner, maximum_level).w;
    int x = int(morton_contract(code)), y = int(morton_contract(code >> 1));
    uint neighbours[2] = {owner_at(owners,x+1,y,side),owner_at(owners,x,y+1,side)};
    for (uint i=0; i<2; ++i) {
        uint other=neighbours[i];
        if (other==0xffffffffu) continue;
        uint other_level=node_geometry(other,maximum_level).w;
        if (level+1<other_level)
            atomic_store_explicit(split+owner,1u,memory_order_relaxed);
        if (other_level+1<level)
            atomic_store_explicit(split+other,1u,memory_order_relaxed);
    }
}

// Stable compaction consumes the exclusive prefix computed by the generic
// tree scan. Breadth-first node IDs uniquely retain level and Morton code.
kernel void mwdg2_compact(device const uint *flags [[buffer(0)]],
                          device const uint *prefix [[buffer(1)]],
                          device uint *active [[buffer(2)]],
                          constant uint &count [[buffer(3)]],
                          uint node [[thread_position_in_grid]]) {
    if (node<count && flags[node]) active[prefix[node]]=node;
}

// Two entries per face, N/E/S/W. Equal entries mean one neighbour covers
// the full face; different entries mean two finer leaves cover its halves.
// The full-face verification flags meshes requiring further 2:1 grading.
kernel void mwdg2_leaf_neighbours(device const uint *active [[buffer(0)]],
                                 device const uint *owners [[buffer(1)]],
                                 device uint *neighbours [[buffer(2)]],
                                 device uint *unbalanced [[buffer(3)]],
                                 constant uint2 &layout [[buffer(4)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index >= layout.x) return;
    uint4 cell = node_geometry(active[index], layout.y);
    uint side = 1u << layout.y;
    uint q0 = cell.z / 4, q1 = min(cell.z - 1, (3 * cell.z) / 4);
    int x = int(cell.x), y = int(cell.y), span = int(cell.z);
    uint ids[8] = {
        owner_at(owners, x+q0, y+span, side), owner_at(owners, x+q1, y+span, side),
        owner_at(owners, x+span, y+q0, side), owner_at(owners, x+span, y+q1, side),
        owner_at(owners, x+q0, y-1, side), owner_at(owners, x+q1, y-1, side),
        owner_at(owners, x-1, y+q0, side), owner_at(owners, x-1, y+q1, side)
    };
    bool invalid = false;
    for (uint i = 0; i < 8; ++i) {
        neighbours[index*8+i] = ids[i];
        if (ids[i] != 0xffffffffu) {
            uint other_level = node_geometry(ids[i], layout.y).w;
            invalid |= abs(int(other_level)-int(cell.w)) > 1;
        }
    }
    // Catch a small refined patch that quarter-face samples could miss.
    for (int i = 0; i < span; ++i) {
        uint face_ids[4] = {owner_at(owners, x+i, y+span, side),
            owner_at(owners, x+span, y+i, side), owner_at(owners, x+i, y-1, side),
            owner_at(owners, x-1, y+i, side)};
        for (uint face = 0; face < 4; ++face)
            invalid |= face_ids[face] != ids[face*2+(i >= span/2)];
    }
    unbalanced[index] = invalid;
}

// Conservatively project four cell-average fields (eta, qx, qy, z) into
// finest-level DG nodes. The source layout is field-major; unresolved slopes
// are initialized to zero. ESRI rows run north to south while adaptive y
// increases northward, matching CUDA's raster import orientation.
kernel void mwdg2_import_raster(device const float *raster [[buffer(0)]],
                               device float *scales [[buffer(1)]],
                               constant uint4 &geometry [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
    // geometry = width, height, padded side, finest-level offset.
    if (index >= geometry.x * geometry.y) return;
    uint x = index % geometry.x;
    uint y = geometry.y - 1 - index / geometry.x;
    uint node = geometry.w + morton_dilate(x) + (morton_dilate(y) << 1);
    uint cells = geometry.x * geometry.y;
    for (uint field = 0; field < 4; ++field) {
        uint p = node * 12 + field * 3;
        scales[p] = raster[field * cells + index];
        scales[p + 1] = 0.0f;
        scales[p + 2] = 0.0f;
    }
}

// Bottom-up transform on the persistent tree. One invocation handles one
// parent and all four physical fields; distinct levels use distinct dispatches.
kernel void mwdg2_encode_level(device float *scales [[buffer(0)]],
                              device float *details [[buffer(1)]],
                              constant uint2 &level [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    if (index >= level.y) return;
    uint node = level.x + index;
    for (uint field = 0; field < 4; ++field) {
        for (uint row = 0; row < 12; ++row) {
            float sum = 0;
            for (uint child = 0; child < 4; ++child)
                for (uint mode = 0; mode < 3; ++mode)
                    sum += mwdg2_coefficient(row, child * 3 + mode) *
                        scales[(node * 4 + 1 + child) * 12 + field * 3 + mode];
            if (row < 3) scales[node * 12 + field * 3 + row] = .5f * sum;
            else details[node * 36 + field * 9 + row - 3] = .5f * sum;
        }
    }
}

kernel void mwdg2_decode_level(device float *scales [[buffer(0)]],
                              device const float *details [[buffer(1)]],
                              constant uint2 &level [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    if (index >= level.y) return;
    uint node = level.x + index;
    for (uint field = 0; field < 4; ++field) {
        for (uint col = 0; col < 12; ++col) {
            float sum = 0;
            for (uint row = 0; row < 12; ++row) {
                float value = row < 3 ? scales[node * 12 + field * 3 + row] :
                    details[node * 36 + field * 9 + row - 3];
                sum += mwdg2_coefficient(row, col) * value;
            }
            uint child = col / 3, mode = col % 3;
            scales[(node * 4 + 1 + child) * 12 + field * 3 + mode] = 2 * sum;
        }
    }
}

// Determine the owning adaptive leaf of each finest cell without a CPU
// neighbour search. This map also indexes finest-grid soil state.
kernel void mwdg2_owner_map(device const uint *split [[buffer(0)]],
                           device uint *owners [[buffer(1)]],
                           constant uint &maximum_level [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    uint count = 1u << (2 * maximum_level);
    if (index >= count) return;
    uint node = 0;
    for (uint level = 0; level < maximum_level && split[node]; ++level) {
        uint child = (index >> (2 * (maximum_level - level - 1))) & 3;
        node = node * 4 + 1 + child;
    }
    owners[index] = node;
}

// One work item handles one parent and one physical variable. Three scaling
// coefficients and nine detail coefficients retain the full DG2 polynomial.
kernel void mwdg2_encode(device const float *children [[buffer(0)]],
                        device float *encoded [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    for (uint row = 0; row < 12; ++row) {
        float sum = 0;
        for (uint col = 0; col < 12; ++col)
            sum += mwdg2_coefficient(row, col) * children[index * 12 + col];
        encoded[index * 12 + row] = sum * 0.5f;
    }
}

kernel void mwdg2_decode(device const float *encoded [[buffer(0)]],
                        device float *children [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    for (uint col = 0; col < 12; ++col) {
        float sum = 0;
        for (uint row = 0; row < 12; ++row)
            sum += mwdg2_coefficient(row, col) * encoded[index * 12 + row];
        children[index * 12 + col] = 2.0f * sum;
    }
}

// Node arrays are in breadth-first Morton order: children of node n are
// 4*n+1 ... 4*n+4. Launch separately for each level, bottom-up, to make every
// child decision visible before its parent. Preflags protect terrain, masks,
// boundaries and spatial soil-parameter discontinuities from coarsening.
kernel void mwdg2_mark_parents(device const float *details [[buffer(0)]],
                              device const uint *preflags [[buffer(1)]],
                              device uint *split [[buffer(2)]],
                              constant uint4 &layout [[buffer(3)]],
                              constant float4 &normalisers [[buffer(4)]],
                              constant float &epsilon [[buffer(5)]],
                              uint index [[thread_position_in_grid]]) {
    // layout = first node at level, number of nodes, level, maximum level.
    if (index >= layout.y) return;
    uint node = layout.x + index;
    bool significant = preflags[node] != 0;
    float threshold = epsilon / float(1u << (layout.w - layout.z));
    for (uint variable = 0; variable < 4; ++variable) {
        for (uint detail = 0; detail < 9; ++detail) {
            float magnitude = abs(details[node * 36 + variable * 9 + detail]);
            significant |= magnitude > threshold * normalisers[variable];
        }
    }
    for (uint child = 0; child < 4; ++child)
        significant |= split[node * 4 + 1 + child] != 0;
    split[node] = significant;
}

// Select only leaves reachable through split ancestors. This is topology
// selection, not raster masking: a coarsened node replaces its descendants.
kernel void mwdg2_leaf_flags(device const uint *split [[buffer(0)]],
                            device uint *leaves [[buffer(1)]],
                            constant uint &count [[buffer(2)]],
                            uint node [[thread_position_in_grid]]) {
    if (node >= count) return;
    bool selected = split[node] == 0;
    uint ancestor = node;
    while (ancestor > 0 && selected) {
        ancestor = (ancestor - 1) / 4;
        selected &= split[ancestor] != 0;
    }
    leaves[node] = selected;
}

// GPU reductions for CUDA-style MRA normalization. CUDA's adaptive MWDG2
// forms eta=h+z before computing its global scale maxima and thresholds eta
// details alongside qx, qy and z.
kernel void mwdg2_reduce_scale_partials(
    device const float *coefficients [[buffer(0)]],
    device const uint *active [[buffer(1)]],
    device float *partials [[buffer(2)]],
    constant uint4 &args [[buffer(3)]], // count, useActive, finestOffset, groupCount
    uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup float local[1024];
    for (uint field = 0; field < 4; ++field) {
        float magnitude = 1.0f;
        if (index < args.x) {
            uint node = args.y ? active[index] : args.z + index;
            const float value = coefficients[node * 12 + field * 3];
            magnitude = max(magnitude, abs(value));
        }
        local[field * 256 + lane] = magnitude;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            for (uint field = 0; field < 4; ++field)
                local[field * 256 + lane] = max(local[field * 256 + lane],
                                                local[field * 256 + lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0)
        for (uint field = 0; field < 4; ++field)
            partials[group * 4 + field] = local[field * 256];
}

kernel void mwdg2_reduce_scale_final(device const float *partials [[buffer(0)]],
                                     device float *normalizers [[buffer(1)]],
                                     constant uint &groupCount [[buffer(2)]],
                                     uint field [[thread_position_in_grid]]) {
    if (field >= 4) return;
    const uint sourceField = field == 2 ? 1 : field;
    float maximum = 1.0f;
    for (uint group = 0; group < groupCount; ++group)
        maximum = max(maximum, partials[group * 4 + sourceField]);
    normalizers[field] = maximum;
}

kernel void mwdg2_reduce_detail_maxima(device const float *details [[buffer(0)]],
                                       device float *componentMaxima [[buffer(1)]],
                                       constant uint &count [[buffer(2)]],
                                       uint index [[thread_position_in_grid]]) {
    if (index >= count * 4) return;
    uint field = index / count, node = index % count;
    float maximum = 0.0f;
    for (uint k = 0; k < 9; ++k) {
        const uint base = node * 36;
        const float detail = details[base + field * 9 + k];
        maximum = max(maximum, abs(detail));
    }
    componentMaxima[index] = maximum;
}

// Match CUDA preflag_loss_details: preserve hierarchy nodes that span
// NoData/domain padding or cell-wise loss-parameter discontinuities.
kernel void mwdg2_preflag_loss_details(device uchar *preflags [[buffer(0)]],
                                       device const float *dem [[buffer(1)]],
                                       device const float *rate [[buffer(2)]],
                                       device const float *ks [[buffer(3)]],
                                       device const float *psi [[buffer(4)]],
                                       device const float *dtheta [[buffer(5)]],
                                       device const float *fmax [[buffer(6)]],
                                       constant uint4 &layout [[buffer(7)]],
                                       constant uint3 &modes [[buffer(8)]],
                                       constant float &nodata [[buffer(9)]],
                                       uint node [[thread_position_in_grid]]) {
    if (node >= layout.x) return;
    const uint4 cell = node_geometry(node, layout.y);
    const uint endX = cell.x + cell.z;
    const uint endY = cell.y + cell.z;
    int firstIndex = -1;
    if (cell.x < layout.z && cell.y < layout.w) {
        uint idx = (layout.w - 1 - cell.y) * layout.z + cell.x;
        if (abs(dem[idx] - nodata) >= 1e-6f) firstIndex = int(idx);
    }
    bool different = false;
    for (uint y = cell.y; y < endY && !different; ++y) {
        for (uint x = cell.x; x < endX; ++x) {
            int idx = -1;
            if (x < layout.z && y < layout.w) {
                uint candidate = (layout.w - 1 - y) * layout.z + x;
                if (abs(dem[candidate] - nodata) >= 1e-6f) idx = int(candidate);
            }
            different = (idx < 0) != (firstIndex < 0);
            if (idx >= 0 && firstIndex >= 0) {
                different |= modes.x && rate[idx] != rate[firstIndex];
                if (modes.y) {
                    different |= ks[idx] != ks[firstIndex] ||
                                 psi[idx] != psi[firstIndex] ||
                                 dtheta[idx] != dtheta[firstIndex];
                    if (modes.z) different |= fmax[idx] != fmax[firstIndex];
                }
            }
            if (different) break;
        }
    }
    if (different) preflags[node] = 1;
}
