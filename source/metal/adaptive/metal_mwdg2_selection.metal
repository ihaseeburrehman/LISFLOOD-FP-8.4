#include <metal_stdlib>
using namespace metal;

// Kernels in this file implement the topology-selection portion of the CUDA
// adaptive MWDG2 path. Node arrays use CUDA's breadth-first quadtree indexing:
// level l begins at (4^l - 1) / 3 and a parent owns four consecutive children.

struct MWDG2LevelArgs {
    uint offset;
    uint count;
    uint level;
    uint maxLevel;
    float epsilon;
};

struct MWDG2RegulariseArgs {
    uint parentOffset;
    uint childOffset;
    uint parentCount;
};

struct MWDG2ExtraArgs {
    uint parentOffset;
    uint childOffset;
    uint parentCount;
    uint level;
    uint maxLevel;
    float epsilon;
    float extraFactor;
};

struct MWDG2DecodeArgs {
    uint parentOffset;
    uint childOffset;
    uint childCount;
};

struct MWDG2LeafArgs {
    uint nodeCount;
    uint finestOffset;
};

// Match DetailMW::get_norm_detail: maximum over eta, qx, qy and bed detail
// maxima after division by their corresponding global maximum magnitudes.
// CUDA's raw division is undefined for a zero maximum. Define the limiting
// behavior explicitly: 0/0 contributes zero, while nonzero/0 is infinite.
inline float safe_normalized_component(float detail, float normalizer) {
    detail = abs(detail);
    if (normalizer > 0.0f) return detail / normalizer;
    return detail == 0.0f ? 0.0f : INFINITY;
}

kernel void mwdg2_normalize_detail_maxima(
    device const float *componentMaxima [[buffer(0)]], // [eta,qx,qy,z][count]
    device const float *normalizers [[buffer(1)]],     // [eta,qx,qy,z]
    device float *normalized [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float n = 0.0f;
    for (uint field = 0; field < 4; ++field) {
        n = max(n, safe_normalized_component(componentMaxima[field * count + i], normalizers[field]));
    }
    normalized[i] = n;
}

// CUDA extra_significance uses d_norm_details from the flow encoder, where
// the bed child coefficients are zero. Terrain z still marks parents through
// the separate topography preflag, but must not broaden that refinement via
// the stronger extra-significance threshold.
kernel void mwdg2_normalize_flow_detail_maxima(
    device const float *componentMaxima [[buffer(0)]], // [eta,qx,qy,z][count]
    device const float *normalizers [[buffer(1)]],
    device float *normalized [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float n = 0.0f;
    for (uint field = 0; field < 3; ++field)
        n = max(n, safe_normalized_component(componentMaxima[field * count + i],
                                             normalizers[field]));
    normalized[i] = n;
}

// CUDA encode_and_thresh_flow: epsilon_local = epsilon / 2^(L-level), then
// significance is norm >= epsilon_local OR a caller-supplied preflag.
kernel void mwdg2_threshold_level(
    device const float *normDetails [[buffer(0)]],
    device const uchar *preflagged [[buffer(1)]],
    device uchar *significant [[buffer(2)]],
    constant MWDG2LevelArgs &args [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= args.count) return;
    const float epsilonLocal = args.epsilon / exp2(float(args.maxLevel - args.level));
    const uint node = args.offset + i;
    significant[node] = (normDetails[node] >= epsilonLocal || preflagged[node] != 0) ? 1 : 0;
}

// CUDA regularisation.cu's upward pass: a parent remains significant if it
// already was, and is set significant whenever any of its four children is.
kernel void mwdg2_regularise_upward(
    device uchar *significant [[buffer(0)]],
    constant MWDG2RegulariseArgs &args [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= args.parentCount) return;
    const uint child = args.childOffset + 4 * i;
    significant[args.parentOffset + i] = significant[args.parentOffset + i] ||
        significant[child] || significant[child + 1] || significant[child + 2] || significant[child + 3];
}

// CUDA extra_significance.cu: a significant detail above the stronger
// level-dependent threshold forces all four children to be refined.
kernel void mwdg2_propagate_extra_significance(
    device uchar *significant [[buffer(0)]],
    device const float *normDetails [[buffer(1)]],
    constant MWDG2ExtraArgs &args [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= args.parentCount || args.level + 1 >= args.maxLevel) return;
    const uint parent = args.parentOffset + i;
    const float epsilonLocal = args.epsilon / exp2(float(args.maxLevel - args.level));
    if (significant[parent] && normDetails[parent] >= epsilonLocal * args.extraFactor) {
        const uint child = args.childOffset + 4 * i;
        significant[child] = 1;
        significant[child + 1] = 1;
        significant[child + 2] = 1;
        significant[child + 3] = 1;
    }
}

// One level of CUDA decoding traversal. A child is reachable exactly when its
// parent is reachable and significant. The scale/detail inverse transform is
// intentionally a separate operation; this kernel only constructs topology.
kernel void mwdg2_decode_topology_level(
    device const uchar *significant [[buffer(0)]],
    device uchar *reachable [[buffer(1)]],
    constant MWDG2DecodeArgs &args [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= args.childCount) return;
    const uint parent = args.parentOffset + i / 4;
    reachable[args.childOffset + i] = reachable[parent] && significant[parent] ? 1 : 0;
}

kernel void mwdg2_mark_reachable_leaves(
    device const uchar *significant [[buffer(0)]],
    device const uchar *reachable [[buffer(1)]],
    device uchar *leaves [[buffer(2)]],
    constant MWDG2LeafArgs &args [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= args.nodeCount) return;
    // At L, the CUDA traversal stops even if details there would exceed the
    // threshold; no finer grid exists, so every reachable finest node is a leaf.
    leaves[i] = reachable[i] && (i >= args.finestOffset || !significant[i]) ? 1 : 0;
}
