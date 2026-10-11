#pragma once

#ifdef __APPLE__
#import <Metal/Metal.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lis {
namespace metal {
namespace mwdg2 {

// GPU exclusive scan for adaptive tree flags. Supports arbitrary counts by
// recursively scanning per-tile totals (tree_scan_tiles itself is a 256 item
// block scan). The returned total is the sum of all input elements.
class TreeScan {
public:
  TreeScan(id<MTLDevice> device, id<MTLLibrary> library,
           id<MTLCommandQueue> queue);
  uint32_t scan(id<MTLBuffer> input, id<MTLBuffer> output, uint32_t count);

private:
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> tileScan_;
  id<MTLComputePipelineState> addOffsets_;
  uint32_t scanLevel(id<MTLBuffer> input, id<MTLBuffer> output,
                     uint32_t count);
  void dispatch(id<MTLComputePipelineState> pipeline,
                const std::vector<id<MTLBuffer>> &buffers,
                const void *args, size_t argSize, uint32_t count,
                uint32_t width);
};

} // namespace mwdg2
} // namespace metal
} // namespace lis
#endif
