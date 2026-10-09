#include <metal_stdlib>
using namespace metal;

// Deterministic exclusive scan for tree compaction. Dispatch one 256-thread
// group per tile, recursively scan tile sums, then add the scanned offsets.
kernel void tree_scan_tiles(device const uint *input [[buffer(0)]],
                            device uint *output [[buffer(1)]],
                            device uint *sums [[buffer(2)]],
                            constant uint &count [[buffer(3)]],
                            uint lane [[thread_index_in_threadgroup]],
                            uint group [[threadgroup_position_in_grid]]) {
    threadgroup uint scratch[256];
    uint index = group * 256 + lane;
    uint value = index < count ? input[index] : 0;
    scratch[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride <<= 1) {
        uint add = lane >= stride ? scratch[lane - stride] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[lane] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (index < count) output[index] = scratch[lane] - value;
    if (lane == 255) sums[group] = scratch[lane];
}

kernel void tree_add_tile_offsets(device uint *output [[buffer(0)]],
                                  device const uint *offsets [[buffer(1)]],
                                  constant uint &count [[buffer(2)]],
                                  uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] += offsets[index / 256];
}

// Stable scatter preserves Morton order when input node IDs are Morton sorted.
kernel void tree_compact_nodes(device const uint *flags [[buffer(0)]],
                               device const uint *prefix [[buffer(1)]],
                               device const uint *nodes [[buffer(2)]],
                               device uint *active [[buffer(3)]],
                               constant uint &count [[buffer(4)]],
                               uint index [[thread_position_in_grid]]) {
    if (index < count && flags[index]) active[prefix[index]] = nodes[index];
}
