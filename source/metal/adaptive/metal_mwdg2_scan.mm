#ifdef __APPLE__
#include "metal_mwdg2_scan.h"
#include <stdexcept>
#include <string>

namespace lis {
namespace metal {
namespace mwdg2 {

TreeScan::TreeScan(id<MTLDevice> device, id<MTLLibrary> library,
                   id<MTLCommandQueue> queue)
    : device_(device), queue_(queue) {
  NSError *error = nil;
  id<MTLFunction> scanFunction = [library newFunctionWithName:@"tree_scan_tiles"];
  if (!scanFunction)
    throw std::runtime_error("Missing Metal kernel tree_scan_tiles");
  tileScan_ = [device newComputePipelineStateWithFunction:scanFunction error:&error];
  if (!tileScan_)
    throw std::runtime_error(std::string("Cannot create tree scan pipeline: ") +
                             error.localizedDescription.UTF8String);
  id<MTLFunction> addFunction = [library newFunctionWithName:@"tree_add_tile_offsets"];
  if (!addFunction)
    throw std::runtime_error("Missing Metal kernel tree_add_tile_offsets");
  addOffsets_ = [device newComputePipelineStateWithFunction:addFunction error:&error];
  if (!addOffsets_)
    throw std::runtime_error(std::string("Cannot create tree scan offset pipeline: ") +
                             error.localizedDescription.UTF8String);
}

void TreeScan::dispatch(id<MTLComputePipelineState> pipeline,
                        const std::vector<id<MTLBuffer>> &buffers,
                        const void *args, size_t argSize, uint32_t count,
                        uint32_t width) {
  @autoreleasepool {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger i = 0; i < buffers.size(); ++i)
      [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    if (args && argSize)
      [encoder setBytes:args length:argSize atIndex:buffers.size()];
    if (pipeline == tileScan_) {
      NSUInteger tiles = (count + 255u) / 256u;
      [encoder dispatchThreadgroups:MTLSizeMake(tiles, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
      [encoder dispatchThreadgroups:MTLSizeMake((count + width - 1) / width, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError)
      throw std::runtime_error(command.error.localizedDescription.UTF8String);
  }
}

uint32_t TreeScan::scanLevel(id<MTLBuffer> input, id<MTLBuffer> output,
                             uint32_t count) {
  if (count == 0) return 0;
  constexpr uint32_t tileWidth = 256;
  const uint32_t tiles = (count + tileWidth - 1) / tileWidth;
  id<MTLBuffer> sums = [device_ newBufferWithLength:tiles * sizeof(uint32_t)
                                           options:MTLResourceStorageModeShared];
  if (!sums) throw std::runtime_error("Cannot allocate adaptive scan tile sums");
  uint32_t tileCount = count;
  dispatch(tileScan_, {input, output, sums}, &tileCount, sizeof(tileCount), count,
           tileWidth);

  if (tiles > 1) {
    id<MTLBuffer> offsets = [device_ newBufferWithLength:tiles * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
    if (!offsets) throw std::runtime_error("Cannot allocate adaptive scan offsets");
    scanLevel(sums, offsets, tiles);
    dispatch(addOffsets_, {output, offsets}, &count, sizeof(count), count, 256);
  }

  const auto *in = static_cast<const uint32_t *>(input.contents);
  const auto *out = static_cast<const uint32_t *>(output.contents);
  return out[count - 1] + in[count - 1];
}

uint32_t TreeScan::scan(id<MTLBuffer> input, id<MTLBuffer> output,
                        uint32_t count) {
  if (count && (!input || !output))
    throw std::invalid_argument("Adaptive scan requires input and output buffers");
  return scanLevel(input, output, count);
}

} // namespace mwdg2
} // namespace metal
} // namespace lis
#endif
