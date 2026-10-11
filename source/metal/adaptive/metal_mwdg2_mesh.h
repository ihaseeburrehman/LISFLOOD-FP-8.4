#pragma once

#ifdef __APPLE__
#import <Metal/Metal.h>
#include "metal_mwdg2_scan.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lis { namespace metal { namespace mwdg2 {

struct AdaptiveMesh {
  __strong id<MTLBuffer> split = nil;
  __strong id<MTLBuffer> owners = nil;
  __strong id<MTLBuffer> leafFlags = nil;
  __strong id<MTLBuffer> prefix = nil;
  __strong id<MTLBuffer> active = nil;
  __strong id<MTLBuffer> nodes = nil;
  __strong id<MTLBuffer> neighbours = nil;
  __strong id<MTLBuffer> unbalanced = nil;
  uint32_t activeCount = 0;
  std::vector<uint32_t> activeHost;

  AdaptiveMesh() = default;
  AdaptiveMesh(const AdaptiveMesh &) = delete;
  AdaptiveMesh &operator=(const AdaptiveMesh &) = delete;
  AdaptiveMesh(AdaptiveMesh &&other) noexcept { moveFrom(other); }
  AdaptiveMesh &operator=(AdaptiveMesh &&other) noexcept {
    if (this != &other) { clear(); moveFrom(other); }
    return *this;
  }
  void clear() noexcept {
    split = nil; owners = nil; leafFlags = nil; prefix = nil;
    active = nil; nodes = nil; neighbours = nil; unbalanced = nil;
    activeCount = 0; activeHost.clear();
  }
private:
  void moveFrom(AdaptiveMesh &other) noexcept {
    split = other.split; other.split = nil;
    owners = other.owners; other.owners = nil;
    leafFlags = other.leafFlags; other.leafFlags = nil;
    prefix = other.prefix; other.prefix = nil;
    active = other.active; other.active = nil;
    nodes = other.nodes; other.nodes = nil;
    neighbours = other.neighbours; other.neighbours = nil;
    unbalanced = other.unbalanced; other.unbalanced = nil;
    activeCount = other.activeCount; other.activeCount = 0;
    activeHost = std::move(other.activeHost);
  }
public:
};

// Runs CUDA-style MRA significance selection, ungraded compaction, and
// arbitrary-level adaptive face-neighbour discovery for breadth-first Morton
// tree buffers. Neighbor CSR records use CUDA's common coarser face level.
// Input/output coefficient buffers use MWDG2's 12 float coefficients per node.
class AdaptiveMeshBuilder {
public:
  AdaptiveMeshBuilder(id<MTLDevice> device, id<MTLLibrary> library,
                      id<MTLCommandQueue> queue);
  AdaptiveMesh select(id<MTLBuffer> scales, id<MTLBuffer> details,
                      id<MTLBuffer> preflags, uint32_t maxLevel,
                      uint32_t nodeCount, float epsilon,
                      id<MTLBuffer> activeCoefficients = nil,
                      id<MTLBuffer> activeNodes = nil,
                      uint32_t activeCount = 0);

private:
  id<MTLDevice> device_;
  id<MTLLibrary> library_;
  id<MTLCommandQueue> queue_;
  TreeScan scan_;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
  id<MTLComputePipelineState> pipeline(const char *name);
  void dispatch(const char *name, const std::vector<id<MTLBuffer>> &buffers,
                const void *args, size_t argSize, uint32_t count,
                uint32_t width = 64);
  AdaptiveMesh build(const uint8_t *significant, uint32_t maxLevel,
                     uint32_t nodeCount, uint32_t finestCount);
};

} } } // namespace lis::metal::mwdg2
#endif
