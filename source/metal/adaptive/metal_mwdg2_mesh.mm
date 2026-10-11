#ifdef __APPLE__
#include "metal_mwdg2_mesh.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace lis { namespace metal { namespace mwdg2 {
namespace {
uint32_t levelOffset(uint32_t level) {
  return static_cast<uint32_t>((((uint64_t)1 << (2 * level)) - 1) / 3);
}
uint32_t levelCount(uint32_t level) { return 1u << (2 * level); }
uint32_t mortonDilate(uint32_t v) {
  v = (v | (v << 8)) & 0x00ff00ffu;
  v = (v | (v << 4)) & 0x0f0f0f0fu;
  v = (v | (v << 2)) & 0x33333333u;
  return (v | (v << 1)) & 0x55555555u;
}
uint32_t mortonContract(uint32_t v) {
  v &= 0x55555555u;
  v = (v | (v >> 1)) & 0x33333333u;
  v = (v | (v >> 2)) & 0x0f0f0f0fu;
  v = (v | (v >> 4)) & 0x00ff00ffu;
  return (v | (v >> 8)) & 0x0000ffffu;
}
struct NodeGeometry { uint32_t x, y, span, level; };
NodeGeometry geometry(uint32_t node, uint32_t maxLevel) {
  uint32_t start=0, count=1, level=0;
  while (node >= start+count) { start+=count; count*=4; ++level; }
  uint32_t code=node-start, span=1u<<(maxLevel-level);
  return {mortonContract(code)*span, mortonContract(code>>1)*span, span, level};
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, size_t bytes, bool zero = true) {
  if (!bytes) bytes = sizeof(uint32_t);
  id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
  if (!buffer) throw std::runtime_error("Cannot allocate adaptive MWDG2 buffer");
  if (zero) std::memset(buffer.contents, 0, bytes);
  return buffer;
}
}

AdaptiveMeshBuilder::AdaptiveMeshBuilder(id<MTLDevice> device,
                                         id<MTLLibrary> library,
                                         id<MTLCommandQueue> queue)
    : device_(device), library_(library), queue_(queue), scan_(device, library, queue) {
  if (!device_ || !library || !queue_)
    throw std::invalid_argument("Adaptive mesh builder needs Metal device, library, and queue");
  // Validate that the complete production mesh pipeline is in this library.
  for (const char *name : {"mwdg2_encode_level", "mwdg2_normalize_detail_maxima",
                           "mwdg2_normalize_flow_detail_maxima",
                           "mwdg2_threshold_level", "mwdg2_regularise_upward",
                           "mwdg2_propagate_extra_significance", "mwdg2_owner_map",
                           "mwdg2_grade", "mwdg2_leaf_flags", "mwdg2_compact",
                           "mwdg2_leaf_neighbours"})
    pipeline(name);
}

id<MTLComputePipelineState> AdaptiveMeshBuilder::pipeline(const char *name) {
  auto found = pipelines_.find(name);
  if (found != pipelines_.end()) return found->second;
  id<MTLFunction> function = [library_ newFunctionWithName:
      [NSString stringWithUTF8String:name]];
  if (!function)
    throw std::runtime_error(std::string("Missing adaptive MWDG2 kernel: ") + name);
  NSError *error = nil;
  id<MTLComputePipelineState> state =
      [device_ newComputePipelineStateWithFunction:function error:&error];
  if (!state)
    throw std::runtime_error(std::string("Cannot create adaptive MWDG2 pipeline: ") +
                             error.localizedDescription.UTF8String);
  pipelines_.emplace(name, state);
  return state;
}

void AdaptiveMeshBuilder::dispatch(const char *name,
                                   const std::vector<id<MTLBuffer>> &buffers,
                                   const void *args, size_t argSize,
                                   uint32_t count, uint32_t width) {
  id<MTLComputePipelineState> state = pipeline(name);
  @autoreleasepool {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:state];
    for (NSUInteger i = 0; i < buffers.size(); ++i)
      [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    if (args && argSize) [encoder setBytes:args length:argSize atIndex:buffers.size()];
    [encoder dispatchThreadgroups:MTLSizeMake((count + width - 1) / width, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError)
      throw std::runtime_error(command.error.localizedDescription.UTF8String);
  }
}

AdaptiveMesh AdaptiveMeshBuilder::select(id<MTLBuffer> scales,
                                         id<MTLBuffer> details,
                                         id<MTLBuffer> preflags,
                                         uint32_t maxLevel,
                                         uint32_t nodeCount,
                                         float epsilon,
                                         id<MTLBuffer> activeCoefficients,
                                         id<MTLBuffer> activeNodes,
                                         uint32_t activeCount) {
  if (!scales || !details || maxLevel == 0 || maxLevel > 15 || epsilon < 0.0f)
    throw std::invalid_argument("Invalid MWDG2 MRA selection inputs");
  const uint32_t finestCount = levelCount(maxLevel);
  const uint32_t finestOffset = levelOffset(maxLevel);
  const uint32_t internalCount = finestOffset;
  if (nodeCount != finestOffset + finestCount)
    throw std::invalid_argument("MWDG2 node count does not match maximum tree level");

  static double prof[6] = {0, 0, 0, 0, 0, 0};
  static long calls = 0;
  const bool profile = std::getenv("LISFLOOD_MWDG2_PROFILE") != nullptr;
  auto now = []() { return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count(); };
  double t0 = now();
  // CUDA encoding order: finest-to-coarsest, so each parent sees its four
  // already encoded child polynomials.
  for (int level = static_cast<int>(maxLevel) - 1; level >= 0; --level) {
    uint32_t args[2] = {levelOffset(level), levelCount(level)};
    dispatch(encodeKernel_.c_str(), {scales, details}, args, sizeof(args), args[1]);
  }

  if (profile) { double t = now(); prof[0] += t - t0; t0 = t; }
  // CUDA computes the normalizers over active leaf scaling coefficients (the
  // initial mesh is finest-level). Keep the reduction on Metal and retain the
  // CUDA floor of 1.0, including its observed qy <- qx0 behavior.
  if (activeCount && (!activeCoefficients || !activeNodes))
    throw std::invalid_argument("MWDG2 active normalization requires both coefficient and node buffers");
  const uint32_t reductionCount = activeCount ? activeCount : finestCount;
  const uint32_t groups = (reductionCount + 255) / 256;
  id<MTLBuffer> dummyActive = activeNodes ? activeNodes : makeBuffer(device_, sizeof(uint32_t));
  id<MTLBuffer> partials = makeBuffer(device_, size_t(groups) * 4 * sizeof(float));
  id<MTLBuffer> normBuffer = makeBuffer(device_, 4 * sizeof(float));
  struct ScaleArgs { uint32_t count, useActive, finestOffset, groupCount; };
  ScaleArgs scaleArgs{reductionCount, activeCount ? 1u : 0u, finestOffset, groups};
  dispatch("mwdg2_reduce_scale_partials",
           {activeCoefficients ? activeCoefficients : scales, dummyActive, partials},
           &scaleArgs, sizeof(scaleArgs), groups * 256, 256);
  dispatch("mwdg2_reduce_scale_final", {partials, normBuffer}, &groups,
           sizeof(groups), 4, 4);

  std::memcpy(lastNorms, normBuffer.contents, sizeof(lastNorms));
  id<MTLBuffer> componentMaxima = makeBuffer(device_,
      size_t(internalCount) * 4 * sizeof(float));
  dispatch("mwdg2_reduce_detail_maxima", {details, componentMaxima}, &internalCount,
           sizeof(internalCount), internalCount * 4);
  id<MTLBuffer> normalized = makeBuffer(device_, size_t(nodeCount) * sizeof(float));
  dispatch("mwdg2_normalize_detail_maxima",
           {componentMaxima, normBuffer, normalized}, &internalCount,
           sizeof(internalCount), internalCount);
  id<MTLBuffer> flowNormalized = makeBuffer(device_, size_t(nodeCount) * sizeof(float));
  dispatch("mwdg2_normalize_flow_detail_maxima",
           {componentMaxima, normBuffer, flowNormalized}, &internalCount,
           sizeof(internalCount), internalCount);

  if (profile) { double t = now(); prof[1] += t - t0; t0 = t; }
  id<MTLBuffer> significant = makeBuffer(device_, nodeCount);
  // Finest-level significance entries have no detail threshold pass, but
  // regularise_upward reads them as child flags. Match CUDA zero_details and
  // prevent allocator contents from spuriously refining their ancestors.
  std::memset(significant.contents, 0, nodeCount);
  id<MTLBuffer> flags = makeBuffer(device_, nodeCount);
  if (preflags) std::memcpy(flags.contents, preflags.contents, nodeCount);
  for (int level = static_cast<int>(maxLevel) - 1; level >= 0; --level) {
    struct Args { uint32_t offset, count, level, maxLevel; float epsilon; };
    Args args{levelOffset(level), levelCount(level), (uint32_t)level,
              maxLevel, epsilon};
    dispatch("mwdg2_threshold_level", {normalized, flags, significant},
             &args, sizeof(args), args.count);
  }
  if (std::getenv("LISFLOOD_MWDG2_DUMP_MRA")) {
    const float *d = static_cast<const float *>(details.contents);
    const float *n = static_cast<const float *>(normalized.contents);
    const uint8_t *s = static_cast<const uint8_t *>(significant.contents);
    const float *scale = static_cast<const float *>(scales.contents);
    std::fprintf(stderr, "MWDG2_MRA normalizers eta=%.9g qx=%.9g qy=%.9g z=%.9g\n",
        static_cast<const float *>(normBuffer.contents)[0],
        static_cast<const float *>(normBuffer.contents)[1],
        static_cast<const float *>(normBuffer.contents)[2],
        static_cast<const float *>(normBuffer.contents)[3]);
    for (uint32_t l=0, off=0; l<maxLevel; ++l) {
      const uint32_t count=levelCount(l);
      for (uint32_t code=0; code<count; ++code) {
        uint32_t node=off+code;
        float zm=0;
        for (uint32_t k=0;k<9;++k) zm=std::max(zm,std::fabs(d[node*36+3*9+k]));
        if (std::getenv("LISFLOOD_MWDG2_DUMP_MRA_ALL") || zm>0 || s[node]) {
          std::fprintf(stderr,"MWDG2_MRA node=%u level=%u code=%u sig=%u norm=%.9g zmax=%.9g zdetails=",node,l,code,s[node],n[node],zm);
          for(uint32_t k=0;k<9;++k) std::fprintf(stderr," %.9g",d[node*36+27+k]);
          std::fprintf(stderr," etadetails=");
          for(uint32_t k=0;k<9;++k) std::fprintf(stderr," %.9g",d[node*36+k]);
          std::fprintf(stderr," etascale=(%.9g %.9g %.9g) zscale=(%.9g %.9g %.9g) children=",scale[node*12],scale[node*12+1],scale[node*12+2],scale[node*12+9],scale[node*12+10],scale[node*12+11]);
          for(uint32_t child=0;child<4;++child) {
            const float *c=scale+(node*4+1+child)*12+9;
            const float *e=scale+(node*4+1+child)*12;
            std::fprintf(stderr," [%u:eta %.9g,%.9g,%.9g z %.9g,%.9g,%.9g]",child,e[0],e[1],e[2],c[0],c[1],c[2]);
          }
          std::fprintf(stderr,"\n");
        }
      }
      off+=count;
    }
  }
  for (int level = static_cast<int>(maxLevel) - 1; level >= 0; --level) {
    uint32_t args[3] = {levelOffset(level), levelOffset(level + 1),
                        levelCount(level)};
    dispatch("mwdg2_regularise_upward", {significant}, args, sizeof(args), args[2]);
  }
  for (uint32_t level = 0; level + 1 < maxLevel; ++level) {
    struct Args { uint32_t parentOffset, childOffset, parentCount, level, maxLevel;
                  float epsilon, extraFactor; };
    Args args{levelOffset(level), levelOffset(level + 1), levelCount(level),
              level, maxLevel, epsilon, std::pow(2.0f, 2.5f)};
    dispatch("mwdg2_propagate_extra_significance", {significant, flowNormalized},
             &args, sizeof(args), args.parentCount);
  }
  if (profile) { double t = now(); prof[2] += t - t0; t0 = t; }
  AdaptiveMesh built = build(static_cast<const uint8_t *>(significant.contents), maxLevel,
                             nodeCount, finestCount);
  if (profile) {
    prof[3] += now() - t0;
    if (++calls % 200 == 0)
      std::printf("mesh select profile (s): encode %.2f normalise %.2f threshold %.2f build %.2f\n",
                  prof[0], prof[1], prof[2], prof[3]);
  }
  return built;
}

AdaptiveMesh AdaptiveMeshBuilder::build(const uint8_t *significant,
                                        uint32_t maxLevel,
                                        uint32_t nodeCount,
                                        uint32_t finestCount) {
  AdaptiveMesh mesh;
  mesh.split = makeBuffer(device_, size_t(nodeCount) * sizeof(uint32_t));
  auto *split = static_cast<uint32_t *>(mesh.split.contents);
  for (uint32_t node = 0; node < nodeCount; ++node)
    split[node] = significant[node] ? 1u : 0u;

  mesh.owners = makeBuffer(device_, size_t(finestCount) * sizeof(uint32_t));
  dispatch("mwdg2_owner_map", {mesh.split, mesh.owners}, &maxLevel,
           sizeof(maxLevel), finestCount);

  mesh.leafFlags = makeBuffer(device_, size_t(nodeCount) * sizeof(uint32_t));
  dispatch("mwdg2_leaf_flags", {mesh.split, mesh.leafFlags}, &nodeCount,
           sizeof(nodeCount), nodeCount);
  mesh.prefix = makeBuffer(device_, size_t(nodeCount) * sizeof(uint32_t));
  mesh.activeCount = scan_.scan(mesh.leafFlags, mesh.prefix, nodeCount);
  if (!mesh.activeCount || mesh.activeCount > finestCount)
    throw std::runtime_error("Adaptive MWDG2 selected an invalid number of leaves");
  mesh.active = makeBuffer(device_, size_t(mesh.activeCount) * sizeof(uint32_t));
  mesh.nodes = makeBuffer(device_, size_t(nodeCount) * sizeof(uint32_t));
  auto *nodes = static_cast<uint32_t *>(mesh.nodes.contents);
  for (uint32_t node = 0; node < nodeCount; ++node) nodes[node] = node;
  dispatch("mwdg2_compact", {mesh.leafFlags, mesh.prefix, mesh.active},
           &nodeCount, sizeof(nodeCount), nodeCount);
  mesh.activeHost.assign(static_cast<uint32_t *>(mesh.active.contents),
                         static_cast<uint32_t *>(mesh.active.contents) + mesh.activeCount);
  // CUDA walks the ungraded tree and resolves a face neighbor at the coarser
  // of the two face levels. When the actual neighbor is finer, this is its
  // ancestor at the local leaf's level. Store one face segment per face in a
  // CSR layout so arbitrary level gaps need no 2:1 grading or fixed slots.
  const uint32_t side=1u<<maxLevel, faceCount=mesh.activeCount*4;
  const uint32_t *owner=static_cast<const uint32_t *>(mesh.owners.contents);
  const uint32_t recordsStart=(faceCount+4u)&~3u; // uint4 alignment in Metal
  std::vector<uint32_t> packed(recordsStart,0);
  std::vector<uint32_t> records;
  records.reserve(size_t(mesh.activeCount)*16);
  for (uint32_t i=0;i<mesh.activeCount;++i) {
    NodeGeometry c=geometry(mesh.activeHost[i],maxLevel);
    for (uint32_t face=0;face<4;++face) {
      packed[i*4+face]=static_cast<uint32_t>(records.size()/4);
      int x=static_cast<int>(c.x+c.span/2), y=static_cast<int>(c.y+c.span/2);
      if(face==0) y=static_cast<int>(c.y+c.span);
      if(face==1) x=static_cast<int>(c.x+c.span);
      if(face==2) y=static_cast<int>(c.y)-1;
      if(face==3) x=static_cast<int>(c.x)-1;
      uint32_t id=0xffffffffu;
      // find_neighbours.cu: N/E from the leaf's north-east finest cell, S/W
      // from its south-west one; that cell on the last raster row/column (or
      // the square's) has no neighbour on that face.
      const int vw=static_cast<int>(validW_?validW_:side), vh=static_cast<int>(validH_?validH_:side);
      const int xne=static_cast<int>(c.x+c.span-1), yne=static_cast<int>(c.y+c.span-1);
      const bool border=(face==0 && (yne==vh-1 || yne==static_cast<int>(side)-1)) ||
                        (face==1 && (xne==vw-1 || xne==static_cast<int>(side)-1)) ||
                        (face==2 && c.y==0) || (face==3 && c.x==0);
      if(!border && x>=0 && y>=0 && x<static_cast<int>(side) && y<static_cast<int>(side)) {
        id=owner[mortonDilate(static_cast<uint32_t>(x)) |
                 (mortonDilate(static_cast<uint32_t>(y))<<1)];
        NodeGeometry other=geometry(id,maxLevel);
        while(other.level>c.level) { id=(id-1u)/4u; other=geometry(id,maxLevel); }
      }
      records.push_back(id); records.push_back(0); records.push_back(c.span); records.push_back(0);
    }
  }
  packed[faceCount]=static_cast<uint32_t>(records.size()/4);
  const size_t packedBytes=(packed.size()+records.size())*sizeof(uint32_t);
  mesh.neighbours=makeBuffer(device_,packedBytes);
  auto *dst=static_cast<uint32_t *>(mesh.neighbours.contents);
  std::memcpy(dst,packed.data(),packed.size()*sizeof(uint32_t));
  std::memcpy(dst+packed.size(),records.data(),records.size()*sizeof(uint32_t));
  return mesh;
}

} } } // namespace lis::metal::mwdg2
#endif
