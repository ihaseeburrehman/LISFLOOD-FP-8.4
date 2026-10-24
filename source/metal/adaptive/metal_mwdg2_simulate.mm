#ifdef __APPLE__
#include "metal_mwdg2_simulate.h"
#include "metal_mwdg2_mesh.h"
#include "mwdg2_input_projection.h"
#include "metal_adaptive_bc.h"
#include "../losses/metal_losses.h"
#include "../../rain/rain.h"
#include "../../rain/rain.tpp"
#include "../../swe/output.h"
#include "../../swe/stats.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lis { namespace metal { namespace mwdg2 {
namespace {
struct GeometryArgs { uint32_t xsz, ysz, side, maxLevel; };
struct LayoutArgs { uint32_t count, maxLevel; };
struct PhysicsArgs { float dx, dy, gravity, cfl; };
struct StageArgs { float dt, blend; };

uint32_t dilate(uint32_t value) {
  value = (value | (value << 8)) & 0x00ff00ffu;
  value = (value | (value << 4)) & 0x0f0f0f0fu;
  value = (value | (value << 2)) & 0x33333333u;
  return (value | (value << 1)) & 0x55555555u;
}
uint32_t contract(uint32_t value) {
  value &= 0x55555555u;
  value = (value | (value >> 1)) & 0x33333333u;
  value = (value | (value >> 2)) & 0x0f0f0f0fu;
  value = (value | (value >> 4)) & 0x00ff00ffu;
  return (value | (value >> 8)) & 0x0000ffffu;
}
uint32_t nodeSpan(uint32_t node, uint32_t maxLevel, uint32_t *x, uint32_t *y) {
  uint64_t start = 0, count = 1; uint32_t level = 0;
  while (node >= start + count) { start += count; count *= 4; ++level; }
  const uint32_t code = node - static_cast<uint32_t>(start);
  if (x) *x = contract(code) << (maxLevel-level);
  if (y) *y = contract(code >> 1) << (maxLevel-level);
  return 1u << (maxLevel-level);
}
uint32_t levelOffset(uint32_t level) {
  return static_cast<uint32_t>((((uint64_t)1 << (2 * level)) - 1) / 3);
}

class Runtime {
public:
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  std::vector<std::string> names;
  std::vector<id<MTLComputePipelineState>> pipelines;

  Runtime(int verbose) {
    device = MTLCreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("Metal device unavailable");
    queue = [device newCommandQueue];
    if (!queue) throw std::runtime_error("Cannot create Metal command queue");
    NSError *error = nil;
    NSString *path = [[NSBundle mainBundle] pathForResource:@"metal_mwdg2_native"
                                                      ofType:@"metallib"];
    if (!path && [[NSFileManager defaultManager]
                    fileExistsAtPath:@"metal_mwdg2_native.metallib"])
      path = @"metal_mwdg2_native.metallib";
    if (!path) {
      NSString *exe = [[[NSProcessInfo processInfo] arguments][0]
                        stringByDeletingLastPathComponent];
      NSString *candidate = [exe stringByAppendingPathComponent:
                               @"metal_mwdg2_native.metallib"];
      if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) path = candidate;
    }
    if (!path) throw std::runtime_error("metal_mwdg2_native.metallib not found next to executable");
    library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if (!library) throw std::runtime_error(std::string("Cannot load native MWDG2 kernels: ") +
                                          error.localizedDescription.UTF8String);
    if (verbose) {
      std::printf("Metal device: %s\n", device.name.UTF8String);
      std::printf("Native adaptive MWDG2 kernels: %s\n", path.UTF8String);
    }
  }

  id<MTLBuffer> buffer(size_t bytes) {
    id<MTLBuffer> result = [device newBufferWithLength:std::max<size_t>(bytes, 4)
                                               options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("Metal MWDG2 buffer allocation failed");
    std::memset(result.contents, 0, std::max<size_t>(bytes, 4));
    return result;
  }

  id<MTLComputePipelineState> pipeline(const char *name) {
    for (size_t i = 0; i < names.size(); ++i) if (names[i] == name) return pipelines[i];
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    if (!function) throw std::runtime_error(std::string("Missing MWDG2 Metal kernel ") + name);
    NSError *error = nil;
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function
                                                                               error:&error];
    if (!state) throw std::runtime_error(std::string("Cannot create MWDG2 pipeline: ") +
                                         error.localizedDescription.UTF8String);
    names.emplace_back(name); pipelines.push_back(state); return state;
  }

  void dispatch(const char *name, const std::vector<id<MTLBuffer>> &buffers,
                const std::vector<const void *> &constants,
                const std::vector<size_t> &sizes, uint32_t count,
                uint32_t width = 64) {
    id<MTLComputePipelineState> state = pipeline(name);
    @autoreleasepool {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:state];
      for (NSUInteger i = 0; i < buffers.size(); ++i)
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
      for (NSUInteger i = 0; i < constants.size(); ++i)
        [encoder setBytes:constants[i] length:sizes[i] atIndex:buffers.size() + i];
      [encoder dispatchThreadgroups:MTLSizeMake((count + width - 1) / width, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
      if (command.status == MTLCommandBufferStatusError)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    }
  }

  // GPU copy of whole buffers (the 12-float trees), instead of a CPU memcpy
  void copy(id<MTLBuffer> dst, id<MTLBuffer> src, size_t bytes) {
    @autoreleasepool {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
      [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:bytes];
      [blit endEncoding]; [command commit]; [command waitUntilCompleted];
    }
  }

  void dispatch(const char *name, const std::vector<id<MTLBuffer>> &buffers,
                const void *constant, size_t size, uint32_t count,
                uint32_t width = 64) {
    std::vector<const void *> constants(1, constant);
    std::vector<size_t> lengths(1, size);
    dispatch(name, buffers, constants, lengths, count, width);
  }
};

void requireSupported(const Fnames &filenames, const States &states, const Pars &pars,
                      const BoundCs &bc, const Arrays &arrays) {
  auto unsupported = [](const char *name) {
    throw std::runtime_error(std::string("native adaptive Metal MWDG2 currently does not support ") + name);
  };
  if (states.rainfallmask == ON) unsupported("distributed rainfall masks");
  if (states.ChannelPresent || states.SGC == ON ||
      states.weirs == ON || states.DamMode == ON || states.routing == ON ||
      states.porosity == ON || states.acceleration == ON)
    unsupported("channels, structures, routing, porosity, or local-inertial physics");
  if (states.latlong == ON) unsupported("lat/long cell metrics");
  if (pars.xsz <= 0 || pars.ysz <= 0 || !arrays.H || !arrays.HU || !arrays.HV || !arrays.DEM)
    throw std::runtime_error("native adaptive MWDG2 received incomplete raster arrays");
}

// After each mesh selection: mark FREE faces of finest edge leaves in the
// neighbour table (ghost 0xfffffffe) and list the HFIX/HVAR/QFIX/QVAR leaves
// (cells[i] = type*4 + side, else -1), as add_ghost_cells.cu applies them to
// finest leaves on the raster edge within each side's cell range.
void applyAdaptiveBoundaries(AdaptiveMesh &mesh, const AdaptiveBoundaries &bc,
                             uint32_t maxLevel, int xsz, int ysz,
                             std::vector<int> &cells, uint32_t westInflow = 0) {
  cells.assign(mesh.activeCount, -1);
  auto *packed = static_cast<uint32_t *>(mesh.neighbours.contents);
  const uint32_t recordsStart = (mesh.activeCount * 4 + 4u) & ~3u;
  auto *segments = reinterpret_cast<uint32_t (*)[4]>(packed + recordsStart);
  for (uint32_t i = 0; i < mesh.activeCount; ++i) {
    uint32_t x = 0, y = 0;
    if (nodeSpan(mesh.activeHost[i], maxLevel, &x, &y) != 1) continue;
    const bool onSide[4] = {static_cast<int>(y) == ysz - 1, static_cast<int>(x) == xsz - 1,
                            y == 0, x == 0};
    const int along[4] = {static_cast<int>(x), static_cast<int>(y),
                          static_cast<int>(x), static_cast<int>(y)};
    for (int side = 0; side < 4; ++side) {
      const AdaptiveSide &b = bc.side[side];
      if (!onSide[side] || !b.contains(along[side])) continue;
      if (b.type == BC_FREE) {
        for (uint32_t j = packed[i * 4 + side]; j < packed[i * 4 + side + 1]; ++j)
          if (segments[j][0] == 0xffffffffu) segments[j][0] = 0xfffffffeu;
      } else if (westInflow && side == SIDE_W && (b.type == BC_HFIX || b.type == BC_HVAR)) {
        // a benchmark inflow sets the ghost, not the edge cell (CUDA)
        for (uint32_t j = packed[i * 4 + side]; j < packed[i * 4 + side + 1]; ++j)
          if (segments[j][0] == 0xffffffffu) segments[j][0] = westInflow;
      } else {
        cells[i] = b.type * 4 + side;
      }
    }
  }
}

FILE *openStageOutput(const Fnames &filenames, const States &states,
                      const Stage &stage, const Pars &pars, const Arrays &arrays) {
  if (states.save_stages != ON) return nullptr;
  char path[512];
  std::snprintf(path, sizeof(path), "%s.stage", filenames.resrootname);
  FILE *fp = std::fopen(path, "w");
  if (!fp) throw std::runtime_error(std::string("cannot open stage output: ") + path);
  std::fprintf(fp, "Stage output, depth (m). Stage locations from: %s\n\n",
               filenames.stagefilename);
  std::fprintf(fp, "Stage information (stage,x,y,elev):\n");
  for (int i=0; i<stage.Nstages; ++i) {
    const int x=stage.stage_grid_x[i], y=stage.stage_grid_y[i];
    if (stage.stage_check[i] && x>=0 && x<pars.xsz && y>=0 && y<pars.ysz)
      std::fprintf(fp, "%d\t%.4f\t%.4f\t%.4f\n", i+1,
          static_cast<double>(stage.stage_loc_x[i]), static_cast<double>(stage.stage_loc_y[i]),
          static_cast<double>(arrays.DEM[size_t(y)*pars.xsz+x]));
    else
      std::fprintf(fp, "%d\t%.4f\t%.4f\tn/a\n", i+1,
          static_cast<double>(stage.stage_loc_x[i]), static_cast<double>(stage.stage_loc_y[i]));
  }
  std::fprintf(fp, "\nOutput, depths:\nTime; stages 1 to %d\n", stage.Nstages);
  std::fflush(fp);
  return fp;
}

void writeStageRecord(FILE *fp, const Stage &stage, const Pars &pars,
                      const Solver &solver, const Arrays &arrays) {
  if (!fp) return;
  std::fprintf(fp, "%12.3f", static_cast<double>(solver.t));
  for (int i=0; i<stage.Nstages; ++i) {
    const int x=stage.stage_grid_x[i], y=stage.stage_grid_y[i];
    if (stage.stage_check[i] && x>=0 && x<pars.xsz && y>=0 && y<pars.ysz)
      std::fprintf(fp, "%10.4f", static_cast<double>(arrays.H[size_t(y)*pars.xsz+x]));
    else std::fprintf(fp, "-\t");
  }
  std::fputc('\n', fp);
  std::fflush(fp);
}

double waterVolume(const AdaptiveMesh &mesh, const float *scales,
                   uint32_t maxLevel, uint32_t validWidth,
                   uint32_t validHeight, double cellArea) {
  double volume = 0.0;
  for (uint32_t node : mesh.activeHost) {
    uint32_t start = 0, count = 1, level = 0;
    while (node >= start + count) { start += count; count *= 4; ++level; }
    const uint32_t span = 1u << (maxLevel - level);
    const uint32_t code = node - start;
    uint32_t cellX = 0, cellY = 0;
    for (uint32_t bit = 0; bit < 16; ++bit) {
      cellX |= ((code >> (2 * bit)) & 1u) << bit;
      cellY |= ((code >> (2 * bit + 1)) & 1u) << bit;
    }
    const uint32_t x0 = cellX * span, y0 = cellY * span;
    const uint32_t coveredWidth = x0 < validWidth ? std::min(span, validWidth - x0) : 0;
    const uint32_t coveredHeight = y0 < validHeight ? std::min(span, validHeight - y0) : 0;
    volume += static_cast<double>(scales[node * 12] - scales[node * 12 + 9]) *
              coveredWidth * coveredHeight * cellArea;
  }
  return volume;
}

double signedLeafVolume(const AdaptiveMesh &mesh, const float *scales,
                        uint32_t maxLevel, uint32_t validWidth,
                        uint32_t validHeight, double cellArea,
                        double *negativeDepthVolume = nullptr) {
  double volume = 0.0, negative = 0.0;
  for (uint32_t node : mesh.activeHost) {
    uint32_t start = 0, count = 1, level = 0;
    while (node >= start + count) { start += count; count *= 4; ++level; }
    const uint32_t span = 1u << (maxLevel - level);
    const uint32_t code = node - start;
    uint32_t cellX = 0, cellY = 0;
    for (uint32_t bit = 0; bit < 16; ++bit) {
      cellX |= ((code >> (2 * bit)) & 1u) << bit;
      cellY |= ((code >> (2 * bit + 1)) & 1u) << bit;
    }
    const uint32_t x0 = cellX * span, y0 = cellY * span;
    const uint32_t coveredWidth = x0 < validWidth ? std::min(span, validWidth - x0) : 0;
    const uint32_t coveredHeight = y0 < validHeight ? std::min(span, validHeight - y0) : 0;
    const double area = coveredWidth * coveredHeight * cellArea;
    const double depth = static_cast<double>(scales[node * 12] - scales[node * 12 + 9]);
    volume += depth * area;
    negative += std::max(0.0, -depth) * area;
  }
  if (negativeDepthVolume) *negativeDepthVolume = negative;
  return volume;
}
}

void simulate(Fnames &filenames, Files *files, States &states, Pars &pars,
              Solver &solver, BoundCs &boundCs, Stage &stage,
              Arrays &arrays, int verbose) {
  @autoreleasepool {
    try {
      requireSupported(filenames, states, pars, boundCs, arrays);
      // HWFV1 runs on the same adaptive tree with Haar wavelets on the cell
      // means (all slope modes zero) and a first-order update.
      const bool haar = states.hwfv1 == ON;
      AdaptiveBoundaries bc;
      bc.read(filenames.bcifilename, filenames.bdyfilename, pars.blx, pars.bly, pars.dx,
              pars.xsz, pars.ysz);
      // cuda/adaptive hard-codes the west HFIX/HVAR inflow of two of its
      // benchmarks, chosen by the DEMfile name as given (read_sim_params.cpp):
      // Monai's non-reflective wave and the Oregon Seaside wavemaker.
      const bool westSpecial = bc.side[SIDE_W].type == BC_HFIX || bc.side[SIDE_W].type == BC_HVAR;
      const uint32_t westInflow =
          westSpecial && !std::strncmp("monai.dem", filenames.demfilename, 9) ? 0xfffffffdu :
          westSpecial && !std::strncmp("oregon-seaside-0p02m.dem", filenames.demfilename, 24) ? 0xfffffffcu : 0u;
      if (westInflow && verbose)
        std::printf("Adaptive west inflow as cuda/adaptive's %s benchmark\n",
                    westInflow == 0xfffffffdu ? "Monai (non-reflective wave)" : "Oregon Seaside (wavemaker)");
      bool prescribedCells = false;
      for (int k = 0; k < 4; ++k)
        prescribedCells |= bc.side[k].type >= BC_HFIX && !(westInflow && k == SIDE_W);
      FILE *stageOutput = openStageOutput(filenames, states, stage, pars, arrays);
      ::DynamicRain<std::allocator<NUMERIC_TYPE>> dynamicRain(
          filenames.dynamicrainfilename, verbose);
      if (states.dynamicrainfall == ON && !dynamicRain.enabled())
        throw std::runtime_error("dynamic rainfall is enabled but dynamicrainfile is missing");
      if (states.rainfall == ON && !arrays.rain)
        throw std::runtime_error("rainfall is enabled but its time series was not loaded");
      if (solver.L < 1 || solver.L > 11)
        throw std::runtime_error("max_ref_lvl must be between 1 and 11 for native Metal MWDG2");
      const uint32_t maxLevel = static_cast<uint32_t>(solver.L);
      const uint32_t side = 1u << maxLevel;
      if (static_cast<uint32_t>(pars.xsz) > side || static_cast<uint32_t>(pars.ysz) > side)
        throw std::runtime_error("max_ref_lvl is too small for the LISFLOOD raster dimensions");
      const uint32_t finestCount = side * side;
      const uint32_t finestOffset = levelOffset(maxLevel);
      const uint32_t nodeCount = finestOffset + finestCount;
      const size_t rasterCells = size_t(pars.xsz) * pars.ysz;
      const size_t treeCells = size_t(side) * side;
      if (nodeCount > std::numeric_limits<uint32_t>::max() / 36u)
        throw std::runtime_error("max_ref_lvl exceeds Metal adaptive tree index capacity");

      Runtime runtime(verbose);
      AdaptiveMeshBuilder meshBuilder(runtime.device, runtime.library, runtime.queue);
      if (haar) meshBuilder.setEncodeKernel("hwfv1_encode_level");
      meshBuilder.setValidSize(static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz));
      Losses losses;
      const bool lossesEnabled = losses.setup(runtime.device, filenames, states, pars,
          &arrays, pars.xsz, pars.ysz, pars.xsz, 0, 0, rasterCells, verbose);
      id<MTLBuffer> lossDepth = nil, lossQx = nil, lossQy = nil, lossDem = nil;
      lossDem = runtime.buffer(rasterCells * sizeof(float));
      auto *demData = static_cast<float *>(lossDem.contents);
      for (size_t i = 0; i < rasterCells; ++i)
        demData[i] = static_cast<float>(arrays.DEM[i]);
      if (lossesEnabled) {
        lossDepth = runtime.buffer(rasterCells * sizeof(float));
        lossQx = runtime.buffer(rasterCells * sizeof(float));
        lossQy = runtime.buffer(rasterCells * sizeof(float));
      }
      const bool rainEnabled = states.rainfall == ON || states.dynamicrainfall == ON;
      id<MTLBuffer> rainRates = rainEnabled ? runtime.buffer(rasterCells * sizeof(float)) : nil;
      std::vector<float> hostRainRates(rainEnabled ? rasterCells : 0, 0.0f);
      id<MTLBuffer> scales = runtime.buffer(size_t(nodeCount) * 12 * sizeof(float));
      id<MTLBuffer> details = runtime.buffer(size_t(nodeCount) * 36 * sizeof(float));
      id<MTLBuffer> modalRaster = runtime.buffer(treeCells * 12 * sizeof(float));
      id<MTLBuffer> packed = runtime.buffer(treeCells * 4 * sizeof(float));
      id<MTLBuffer> preflags = runtime.buffer(nodeCount);

      // Build the square finest raster required by the Morton tree. Input
      // arrays are file-row-major (north-to-south); the projection buffer is
      // south-to-north as used by the CUDA adaptive mesh.
      auto *packedData = static_cast<float *>(packed.contents);
      const float wall = static_cast<float>(pars.wall_height);
      const size_t plane = treeCells;
      // CUDA preflags prescribed stage/BC/source regions, not every physical
      // edge. Keep NoData transitions finest for safe drain/mask handling;
      // ordinary reflecting boundaries remain eligible for coarsening.
      auto *protectedNodes = static_cast<uint8_t *>(preflags.contents);
      // MTLBuffer storage is not guaranteed to be zeroed. CUDA's zero_details
      // initializes the significance/preflag hierarchy every adaptation pass;
      // clear the Metal preflags before setting any protected nodes.
      std::memset(protectedNodes, 0, nodeCount);
      const uint32_t boundaryLevel = maxLevel > 1 ? maxLevel - 1 : 0;
      const uint32_t boundaryOffset = levelOffset(boundaryLevel);
      auto markProtectedParent = [&](uint32_t x, uint32_t y) {
        const uint32_t code = dilate(x >> 1) | (dilate(y >> 1) << 1);
        protectedNodes[boundaryOffset + code] = 1;
      };
      for (uint32_t y = 0; y < side; ++y) {
        for (uint32_t x = 0; x < side; ++x) {
          const size_t internal = size_t(y) * side + x;
          // Padding beyond the raster repeats the edge cells. (CUDA pads with
          // wall_height and no water; coarse leaves next to the raster edge
          // then read ancestors that mix wall and water and a lake at rest
          // starts to flow. Edge leaves see ghost cells as in CUDA, and DG2
          // does not update padding leaves, so the padding values only enter
          // through those ancestors and the multiresolution analysis.)
          // NoData cells are walls without water.
          float h = 0.0f, qx = 0.0f, qy = 0.0f, z = wall;
          {
            const size_t input = size_t(pars.ysz - 1 - std::min(y, uint32_t(pars.ysz - 1))) * pars.xsz +
                                 std::min(x, uint32_t(pars.xsz - 1));
            z = static_cast<float>(arrays.DEM[input]);
            if (std::fabs(z - static_cast<float>(pars.nodata_elevation)) < 1.0f) {
              z = wall;
            } else {
              h = static_cast<float>(std::max<NUMERIC_TYPE>(0, arrays.H[input]));
              qx = static_cast<float>(arrays.HU[input]);
              qy = static_cast<float>(arrays.HV[input]);
            }
          }
          // projection kernel expects h,qx,qy,z and converts h to eta=h+z.
          packedData[internal] = h;
          packedData[plane + internal] = qx;
          packedData[2 * plane + internal] = qy;
          packedData[3 * plane + internal] = z;
        }
      }
      for (int side = 0; side < 4; ++side) {
        const AdaptiveSide &b = bc.side[side];
        if (b.type == BC_CLOSED) continue;
        for (int c = b.start; c <= b.end; ++c) {
          if (side == SIDE_N) markProtectedParent(c, pars.ysz - 1);
          else if (side == SIDE_S) markProtectedParent(c, 0);
          else if (side == SIDE_E) markProtectedParent(pars.xsz - 1, c);
          else markProtectedParent(0, c);
        }
      }
      for (const auto &src : bc.sources) markProtectedParent(src.x, src.y);
      // Rain (a Metal extension: CUDA's adaptive solver has none) and
      // drain_nodata act per NoData cell, so with either one the NoData edges
      // stay at the finest level (the losses preflag below does the same).
      if (states.rainfall == ON || states.dynamicrainfall == ON || pars.drain_nodata == ON)
        for (uint32_t y = 0; y < static_cast<uint32_t>(pars.ysz); ++y)
          for (uint32_t x = 0; x < static_cast<uint32_t>(pars.xsz); ++x) {
            auto noData = [&](int cx, int cy) {
              return std::fabs(static_cast<float>(arrays.DEM[size_t(pars.ysz - 1 - cy) * pars.xsz + cx]) -
                               static_cast<float>(pars.nodata_elevation)) < 1.0f;
            };
            bool mark = noData(int(x), int(y));
            const int dx4[4] = {-1, 1, 0, 0}, dy4[4] = {0, 0, -1, 1};
            for (int f = 0; f < 4 && !mark; ++f) {
              const int nx = int(x) + dx4[f], ny = int(y) + dy4[f];
              if (nx >= 0 && ny >= 0 && nx < pars.xsz && ny < pars.ysz) mark = noData(nx, ny);
            }
            if (mark) markProtectedParent(x, y);
          }
      if (pars.refine_wall == ON)  // refine_high_wall.cu: north and east strips
        for (int j = 0; j < pars.ref_thickness; ++j) {
          for (int i = 0; i < pars.xsz; ++i) markProtectedParent(i, pars.ysz - 1 - j);
          for (int i = 0; i < pars.ysz; ++i) markProtectedParent(pars.xsz - 1 - j, i);
        }
      for (int i = 0; i < stage.Nstages; ++i)
        if (stage.stage_check[i] && stage.stage_grid_x[i] >= 0 && stage.stage_grid_y[i] >= 0 &&
            stage.stage_grid_x[i] < pars.xsz && stage.stage_grid_y[i] < pars.ysz)
          markProtectedParent(stage.stage_grid_x[i], pars.ysz - 1 - stage.stage_grid_y[i]);
      if (lossesEnabled) {
        uint32_t lossPreflagLayout[4] = {nodeCount, maxLevel,
            static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz)};
        uint32_t lossPreflagModes[3] = {losses.uses_rate_grid() ? 1u : 0u,
            losses.uses_soil_params() ? 1u : 0u,
            losses.uses_soil_depth() ? 1u : 0u};
        const float nodata = static_cast<float>(pars.nodata_elevation);
        runtime.dispatch("mwdg2_preflag_loss_details",
            {preflags, lossDem, losses.rate_buffer(), losses.ks_buffer(),
             losses.psi_buffer(), losses.dtheta_buffer(), losses.fmax_buffer()},
            {lossPreflagLayout, lossPreflagModes, &nodata},
            {sizeof(lossPreflagLayout), sizeof(lossPreflagModes), sizeof(nodata)},
            nodeCount);
      }
      uint32_t wallBits = 0;
      if (!std::getenv("LISFLOOD_ADAPTIVE_CUDA_NODATA_INPUT"))  // CUDA's wall averaging
        std::memcpy(&wallBits, &wall, sizeof(wallBits));
      uint32_t inputGeometry[4] = {side, side, static_cast<uint32_t>(pars.xsz), wallBits};
      runtime.dispatch("mwdg2_project_input", {packed, modalRaster}, inputGeometry,
                       sizeof(inputGeometry), finestCount);
      const float *modal = static_cast<const float *>(modalRaster.contents);
      if (std::getenv("LISFLOOD_MWDG2_DUMP_INPUT_MODES")) {
        for (uint32_t y=0; y<side; ++y) for (uint32_t x=0; x<side; ++x) {
          const float *m=modal+(size_t(y)*side+x)*12;
          if (std::fabs(m[1])+std::fabs(m[2])+std::fabs(m[10])+std::fabs(m[11]) > 1e-6f)
            std::fprintf(stderr,"MWDG2_INPUT_MODE cell=(%u,%u) eta=(%.9g %.9g %.9g) z=(%.9g %.9g %.9g)\n",
              x,y,double(m[0]),double(m[1]),double(m[2]),double(m[9]),double(m[10]),double(m[11]));
        }
      }
      float *tree = static_cast<float *>(scales.contents);
      for (uint32_t y = 0; y < side; ++y) for (uint32_t x = 0; x < side; ++x) {
        const uint32_t source = (y * side + x) * 12;
        const uint32_t target = (finestOffset + dilate(x) + (dilate(y) << 1)) * 12;
        std::copy(modal + source, modal + source + 12, tree + target);
        if (haar) {  // read_and_project_modes_fv1: the raster values are the means
          const size_t internal = size_t(y) * side + x;
          const float zc = packedData[3 * plane + internal];
          float *node = tree + target;
          for (int k = 0; k < 12; ++k) node[k] = 0.0f;
          node[0] = packedData[internal] + zc;
          node[3] = packedData[plane + internal];
          node[6] = packedData[2 * plane + internal];
          node[9] = zc;
        }
      }

      // reinsert_point_srcs.cu, before every selection with the coming step's
      // time and dt: HFIX/HVAR set the finest cell's water level, QFIX/QVAR add
      // q*dt/dx_finest of depth. Returns the volume added (m3).
      auto insertPointSources = [&](float time, float stepDt) {
        double added = 0.0;
        if (bc.sources.empty()) return added;
        bc.update(time);
        for (const auto &src : bc.sources) {
          float *node = tree + (finestOffset + dilate(src.x) + (dilate(src.y) << 1)) * 12;
          const float before = node[0];
          if (src.type == BC_HFIX || src.type == BC_HVAR) node[0] = src.value;
          else node[0] += src.value * stepDt / static_cast<float>(pars.dx);
          added += static_cast<double>(node[0] - before) * pars.dx * pars.dy;
        }
        return added;
      };
      boundCs.VolInMT += static_cast<NUMERIC_TYPE>(insertPointSources(0.001f, 0.001f));
      // Initial active mesh from MRA, including protected physical boundary
      // and NoData-transition cells.
      AdaptiveMesh mesh = meshBuilder.select(scales, details, preflags,
                                             maxLevel, nodeCount,
                                             static_cast<float>(solver.epsilon));
      std::vector<int> bcCellHost;
      applyAdaptiveBoundaries(mesh, bc, maxLevel, pars.xsz, pars.ysz, bcCellHost, westInflow);
      if (verbose) std::printf("Metal MWDG2 native adaptive leaves: %u / %u finest cells\n",
                               mesh.activeCount, finestCount);
      if (std::getenv("LISFLOOD_MWDG2_DUMP_LEAVES")) {
        std::fprintf(stderr, "MWDG2_LEAF_IDS");
        for (uint32_t node : mesh.activeHost) std::fprintf(stderr, " %u", node);
        std::fprintf(stderr, "\n");
      }
      id<MTLBuffer> rhs[2], state[3], localDt, projected, leafManning, boundaryFlux, limitedSlopes;
      uint32_t workCapacity = mesh.activeCount;
      for (int i = 0; i < 2; ++i) rhs[i] = runtime.buffer(size_t(mesh.activeCount) * 9 * sizeof(float));
      for (int i = 0; i < 3; ++i) state[i] = runtime.buffer(size_t(nodeCount) * 12 * sizeof(float));
      localDt = runtime.buffer(size_t(mesh.activeCount) * sizeof(float));
      projected = runtime.buffer(rasterCells * 4 * sizeof(float));
      leafManning = runtime.buffer(size_t(mesh.activeCount) * sizeof(float));
      boundaryFlux = runtime.buffer(size_t(mesh.activeCount) * sizeof(float));
      id<MTLBuffer> expandFlags = runtime.buffer(size_t(nodeCount) * sizeof(uint32_t));
      id<MTLBuffer> frictionFlags = runtime.buffer(size_t(mesh.activeCount) * sizeof(uint32_t));
      id<MTLBuffer> bcCells = runtime.buffer(size_t(mesh.activeCount) * sizeof(int));
      // frictionFlags bit 1: leaf centre outside the raster (dg2_update.cu skips it)
      auto markPaddingLeaves = [&]() {
        auto *f = static_cast<uint32_t *>(frictionFlags.contents);
        for (uint32_t i = 0; i < mesh.activeCount; ++i) {
          uint32_t x = 0, y = 0;
          const uint32_t span = nodeSpan(mesh.activeHost[i], maxLevel, &x, &y);
          const bool outside = 2 * x + span >= 2u * static_cast<uint32_t>(pars.xsz) ||
                               2 * y + span >= 2u * static_cast<uint32_t>(pars.ysz);
          f[i] = (f[i] & 1u) | (outside ? 2u : 0u);
        }
      };
      markPaddingLeaves();
      id<MTLBuffer> bcDelta = runtime.buffer(size_t(mesh.activeCount) * sizeof(float));
      std::memcpy(bcCells.contents, bcCellHost.data(), bcCellHost.size() * sizeof(int));
      if (pars.limit_slopes == ON)
        limitedSlopes = runtime.buffer(size_t(mesh.activeCount) * 6 * sizeof(float));
      std::vector<float> manning(mesh.activeCount, static_cast<float>(pars.FPn));
      const auto *active = mesh.activeHost.data();
      for (uint32_t i = 0; i < mesh.activeCount; ++i) {
        uint32_t node = active[i], start = 0, count = 1, level = 0;
        while (node >= start + count) { start += count; count *= 4; ++level; }
        const uint32_t span = 1u << (maxLevel - level);
        const uint32_t code = node - start;
        uint32_t cx = 0, cy = 0;
        for (uint32_t bit = 0; bit < 16; ++bit) {
          cx |= ((code >> (2 * bit)) & 1u) << bit;
          cy |= ((code >> (2 * bit + 1)) & 1u) << bit;
        }
        const uint32_t x = cx * span + span / 2;
        const uint32_t y = cy * span + span / 2;
        if (arrays.Manningsn && x < static_cast<uint32_t>(pars.xsz) &&
            y < static_cast<uint32_t>(pars.ysz))
          manning[i] = static_cast<float>(arrays.Manningsn[size_t(pars.ysz - 1 - y) * pars.xsz + x]);
      }
      std::memcpy(leafManning.contents, manning.data(), manning.size() * sizeof(float));
      std::memcpy(state[0].contents, scales.contents, size_t(nodeCount) * 12 * sizeof(float));

      // CUDA friction_implicit also updates the neighbour coefficients it has
      // loaded (d_neighbours), which may be ancestors of leaves. Keep a copy
      // of the whole tree with friction applied at every node for stage 1.
      bool frictionOn = pars.FPn > 0;
      if (arrays.Manningsn)
        for (size_t i = 0; i < rasterCells && !frictionOn; ++i) frictionOn = arrays.Manningsn[i] > 0;
      id<MTLBuffer> allNodes = nil, nodeManning = nil, neighbourTree = nil;
      if (frictionOn) {
        allNodes = runtime.buffer(size_t(nodeCount) * sizeof(uint32_t));
        nodeManning = runtime.buffer(size_t(nodeCount) * sizeof(float));
        neighbourTree = runtime.buffer(size_t(nodeCount) * 12 * sizeof(float));
        auto *ids = static_cast<uint32_t *>(allNodes.contents);
        auto *n = static_cast<float *>(nodeManning.contents);
        for (uint32_t node = 0; node < nodeCount; ++node) {
          ids[node] = node;
          uint32_t x = 0, y = 0;
          const uint32_t span = nodeSpan(node, maxLevel, &x, &y);
          x += span / 2; y += span / 2;
          n[node] = static_cast<float>(pars.FPn);
          if (arrays.Manningsn && x < static_cast<uint32_t>(pars.xsz) &&
              y < static_cast<uint32_t>(pars.ysz))
            n[node] = static_cast<float>(arrays.Manningsn[size_t(pars.ysz - 1 - y) * pars.xsz + x]);
        }
      }

      auto applyCudaSlopeLimiter = [&](id<MTLBuffer> stageState, id<MTLBuffer> neighbourState) {
        if (pars.limit_slopes != ON) return;
        const float *u = static_cast<const float *>(stageState.contents);
        float maxDepth = -std::numeric_limits<float>::infinity();
        for (uint32_t node : mesh.activeHost)
          maxDepth = std::max(maxDepth, u[node*12]-u[node*12+9]);
        uint32_t limiterLayout[4] = {mesh.activeCount, maxLevel,
            static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz)};
        float limiterPhysics[4] = {static_cast<float>(pars.dx), static_cast<float>(pars.dy),
            static_cast<float>(solver.g), static_cast<float>(solver.tol_Krivo)};
        runtime.dispatch("mwdg2_limit_slopes",
            {stageState, mesh.active, mesh.neighbours, limitedSlopes, neighbourState, scales, frictionFlags},
            {limiterLayout, limiterPhysics, &maxDepth},
            {sizeof(limiterLayout), sizeof(limiterPhysics), sizeof(maxDepth)}, mesh.activeCount);
        if (std::getenv("LISFLOOD_MWDG2_DUMP_LIMITER")) {
          const float *u=static_cast<const float *>(stageState.contents);
          const float *lim=static_cast<const float *>(limitedSlopes.contents);
          for(uint32_t i=0;i<mesh.activeCount;++i) {
            const uint32_t node=mesh.activeHost[i]; uint32_t x=0,y=0;
            const uint32_t span=nodeSpan(node,maxLevel,&x,&y);
            if(x>=12 && x<=16 && y>=14 && y<=18)
              std::fprintf(stderr,"MWDG2_LIMIT leaf=%u cell=(%u,%u,%u) eta=(%.9g %.9g %.9g) bed=(%.9g %.9g %.9g) limited=(%.9g %.9g %.9g %.9g %.9g %.9g)\n",node,x,y,span,double(u[node*12]),double(u[node*12+1]),double(u[node*12+2]),double(u[node*12+9]),double(u[node*12+10]),double(u[node*12+11]),double(lim[i*6]),double(lim[i*6+1]),double(lim[i*6+2]),double(lim[i*6+3]),double(lim[i*6+4]),double(lim[i*6+5]));
          }
        }
        runtime.dispatch("mwdg2_apply_limited_slopes",
            {stageState, mesh.active, limitedSlopes}, &mesh.activeCount,
            sizeof(mesh.activeCount), mesh.activeCount);
      };

      double cellArea = static_cast<double>(pars.dx) * static_cast<double>(pars.dy);
      double initialVolume = waterVolume(mesh, tree, maxLevel,
          static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz), cellArea);
      const bool traceMass = std::getenv("LISFLOOD_MWDG2_TRACE_MASS") != nullptr;
      const bool positiveExpand = std::getenv("LISFLOOD_ADAPTIVE_POSITIVE_EXPAND") != nullptr;
      double traceUpdate = 0.0, traceRemesh = 0.0, traceLoss = 0.0, traceRain = 0.0;
      const double initialRasterVolume = DomainVol(&states, &pars, nullptr, &arrays, nullptr);
      solver.vol1 = static_cast<NUMERIC_TYPE>(initialVolume);
      if (verbose)
        std::printf("MWDG2 volume audit init: raw raster=%.9g m3, active-leaf=%.9g m3\n",
                    initialRasterVolume, initialVolume);
      double cumulativeLossVolume = 0.0;
      double integratedLossTime = 0.0;
      bool firstLossRecord = true;
      solver.MinTstep = solver.InitTstep;
      if (verbose) {
        const char *forcing = lossesEnabled && rainEnabled ? "raster losses and rainfall enabled" :
            lossesEnabled ? "raster infiltration/evaporation enabled" :
            rainEnabled ? "rainfall enabled" : "no losses/forcing";
        std::printf("Starting native adaptive Metal %s (%s; boundary sides N,E,S,W types %d %d %d %d; %zu point sources)\n",
                    haar ? "HWFV1" : "MWDG2", forcing, bc.side[0].type, bc.side[1].type,
                    bc.side[2].type, bc.side[3].type, bc.sources.size());
        std::fflush(stdout);
      }

      // CUDA acc_nugrid applies the forcing for solver.t using the current
      // Tstep before computing the next CFL-limited step, including at t=0.
      // Keep that first application (InitTstep) and carry each selected dt
      // forward to the following forcing update.
      // Match the fixed-grid Metal DG2 forcing schedule: load the field at
      // t=0, but do not integrate rainfall over a timestep before the first
      // hydrodynamic update.
      float previousDt = 0.0f;  // CFL proposal for the next step
      float previousStepDt = 0.0f;  // step actually taken (rain integration)
      const bool diagSteps = std::getenv("LISFLOOD_DIAG_STEPS") != nullptr;
      auto drainNoData = [&](id<MTLBuffer> stateBuffer) {
        if (pars.drain_nodata != ON) return;
        uint32_t layout[4] = {mesh.activeCount, maxLevel,
            static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz)};
        float physics[2] = {static_cast<float>(pars.nodata_elevation),
                            static_cast<float>(cellArea)};
        runtime.dispatch("mwdg2_drain_nodata",
            {stateBuffer, mesh.active, lossDem, localDt},
            {layout, physics}, {sizeof(layout), sizeof(physics)}, mesh.activeCount);
        const float *volumes = static_cast<const float *>(localDt.contents);
        double drained = 0.0;
        for (uint32_t i = 0; i < mesh.activeCount; ++i) drained += volumes[i];
        boundCs.VolOutMT += static_cast<NUMERIC_TYPE>(drained);
      };
      // LISFLOOD_MWDG2_PROFILE: wall time per phase, printed every 200 steps
      const bool profile = std::getenv("LISFLOOD_MWDG2_PROFILE") != nullptr;
      bool cumuStarted = false;
      // cuda/adaptive raster_out writes its grids with 8 decimals (float build)
      if (pars.raster_out == ON && pars.output_precision_set != ON) pars.output_precision = 8;
      double phase[5] = {0, 0, 0, 0, 0};
      auto clock = []() { return std::chrono::duration<double>(
          std::chrono::steady_clock::now().time_since_epoch()).count(); };
      double mark = clock();
      auto lap = [&](int k) { const double now = clock(); phase[k] += now - mark; mark = now; };
      while (solver.t < solver.Sim_Time) {
        @autoreleasepool {
        const double stepStartVolume = traceMass ? waterVolume(mesh,
            static_cast<const float *>(scales.contents), maxLevel,
            static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz), cellArea) : 0.0;
        drainNoData(scales);
        zero_flux_stats(&boundCs);
        if (rainEnabled) {
          std::fill(hostRainRates.begin(), hostRainRates.end(), 0.0f);
          if (dynamicRain.enabled()) {
            dynamicRain.update_time(solver.t);
            const Geometry &rainGeometry = dynamicRain.geometry();
            const NUMERIC_TYPE *rainData = dynamicRain.data();
            for (int row = 0; row < pars.ysz; ++row) {
              for (int col = 0; col < pars.xsz; ++col) {
                const int ri = static_cast<int>(std::floor((pars.blx - rainGeometry.blx +
                    col * pars.dx) / rainGeometry.dx));
                const int rj = static_cast<int>(std::floor((row * pars.dy -
                    (pars.tly - rainGeometry.tly)) / rainGeometry.dy));
                if (ri >= 0 && ri < rainGeometry.xsz && rj >= 0 && rj < rainGeometry.ysz)
                  hostRainRates[size_t(row) * pars.xsz + col] =
                      std::max(0.0f, static_cast<float>(rainData[size_t(rj) * rainGeometry.xsz + ri]));
              }
            }
          }
          float uniformRate = 0.0f;
          if (states.rainfall == ON)
            // LoadRain stores uniform rainfall in m/s (the source file is
            // mm/hour), matching the dynamic-rain rate units.
            uniformRate = static_cast<float>(InterpolateTimeSeries(arrays.rain, solver.t));
          if (uniformRate != 0.0f)
            for (float &rate : hostRainRates) rate += uniformRate;
          std::memcpy(rainRates.contents, hostRainRates.data(), rasterCells * sizeof(float));
          if (previousStepDt > 0.0f) {
            double addedVolume = 0.0;
            const uint32_t *owners = static_cast<const uint32_t *>(mesh.owners.contents);
            const float *stateData = static_cast<const float *>(scales.contents);
            for (int row = 0; row < pars.ysz; ++row) {
              for (int col = 0; col < pars.xsz; ++col) {
                const size_t raster = size_t(row) * pars.xsz + col;
                if (std::fabs(demData[raster] - static_cast<float>(pars.nodata_elevation)) < 1e-6f)
                  continue;
                const uint32_t y = static_cast<uint32_t>(pars.ysz - 1 - row);
                const uint32_t code = dilate(static_cast<uint32_t>(col)) | (dilate(y) << 1);
                const uint32_t node = owners[code];
                const float h = std::max(0.0f, stateData[node * 12] - stateData[node * 12 + 9]);
                float increment = hostRainRates[raster] * previousStepDt;
                if (states.rainfall == ON && uniformRate < 0.0f)
                  increment = std::max(-h, increment);
                else increment = std::max(0.0f, increment);
                addedVolume += static_cast<double>(increment) * pars.dx * pars.dy;
              }
            }
            uint32_t rainLayout[4] = {mesh.activeCount, maxLevel,
                static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz)};
            float rainPhysics[3] = {previousStepDt, static_cast<float>(pars.nodata_elevation),
                states.rainfall == ON && uniformRate < 0.0f ? 1.0f : 0.0f};
            runtime.dispatch("mwdg2_apply_rain", {scales, mesh.active, rainRates, lossDem},
                {rainLayout, rainPhysics}, {sizeof(rainLayout), sizeof(rainPhysics)},
                mesh.activeCount);
            boundCs.VolInMT += static_cast<NUMERIC_TYPE>(addedVolume);
            traceRain += addedVolume;
            pars.RainTotalLoss += static_cast<NUMERIC_TYPE>(addedVolume);
          }
        }
        LayoutArgs layout{mesh.activeCount, maxLevel};
        PhysicsArgs cflPhysics{static_cast<float>(pars.dx), static_cast<float>(pars.dy),
                               static_cast<float>(solver.g), static_cast<float>(solver.cfl)};
        PhysicsArgs flowPhysics{static_cast<float>(pars.dx), static_cast<float>(pars.dy),
                                static_cast<float>(solver.g), static_cast<float>(solver.DepthThresh)};
        const float depthTolerance = static_cast<float>(solver.DepthThresh);
        // CUDA computes dt_CFL at the end of the preceding DG2 update, before
        // adapting the tree. Carry that proposal across regridding so the CFL
        // estimate uses the same updated state and leaf sizes as CUDA.
        float dt = solver.t == 0.0 ? 0.001f : previousDt;
        // As cuda_adaptive_simulate.cu: only the last step is shortened (to
        // end at sim_time); outputs are written after the first step that
        // reaches each save or mass time.
        const double deadline = static_cast<double>(solver.Sim_Time);
        dt = std::min(dt, static_cast<float>(deadline - static_cast<double>(solver.t)));
        if (!(dt > 0.0f) || !std::isfinite(dt))
          throw std::runtime_error("adaptive CFL produced a non-positive timestep");

        // Stage-1 view (state[0]), as CUDA's assembled solution: load the
        // tree, apply friction, then limit. CUDA's copy_to_buf_assem_sol buffer is
        // overwritten by the stage-1 update, so the SSP-RK2 average uses this
        // limited view. Neighbours read the tree with friction applied but not
        // limited (stage1Tree).
        uint32_t activeCount = mesh.activeCount;
        runtime.copy(state[0], scales, size_t(nodeCount) * 12 * sizeof(float));
        // Boundaries::update_all_inlets at the step's end time, then
        // add_ghost_cells on each stage's own leaf state.
        bc.update(static_cast<float>(solver.t) + dt);
        float bcInlet[4] = {bc.side[0].inlet, bc.side[1].inlet, bc.side[2].inlet, bc.side[3].inlet};
        // benchmark west inflow: {inlet, dt, finest dx, end-of-step time}
        float waveArgs[4] = {bc.side[SIDE_W].inlet, dt, static_cast<float>(pars.dx),
                             static_cast<float>(solver.t + dt)};
        float bcParams[4] = {dt, static_cast<float>(pars.dx),
                             static_cast<float>(pars.dx * pars.dy), 0.0f};
        auto applyBoundaryCells = [&](id<MTLBuffer> view) {
          if (!prescribedCells) return;
          runtime.dispatch("adaptive_boundary_cells", {view, mesh.active, bcCells, bcDelta},
              {&activeCount, bcInlet, bcParams},
              {sizeof(activeCount), sizeof(bcInlet), sizeof(bcParams)}, mesh.activeCount);
          double added = 0.0;
          const float *d = static_cast<const float *>(bcDelta.contents);
          for (uint32_t i = 0; i < mesh.activeCount; ++i) added += d[i];
          boundCs.VolInMT += static_cast<NUMERIC_TYPE>(added);
        };
        applyBoundaryCells(state[0]);
        auto edgeRate = [&]() {
          double rate = 0.0;
          const float *r = static_cast<const float *>(boundaryFlux.contents);
          for (uint32_t i = 0; i < mesh.activeCount; ++i) rate += r[i];
          return rate;
        };
        float frictionParams[4] = {static_cast<float>(solver.g), dt,
                                   depthTolerance, 1e-9f};  // CUDA adaptive tol_s
        id<MTLBuffer> stage1Tree = scales;
        if (frictionOn) {
          runtime.dispatch("mwdg2_friction_own", {state[0], mesh.active, leafManning, frictionFlags},
                           {&activeCount, frictionParams}, {sizeof(activeCount), sizeof(frictionParams)},
                           mesh.activeCount);
          runtime.copy(neighbourTree, scales, size_t(nodeCount) * 12 * sizeof(float));
          runtime.dispatch("mwdg2_friction", {neighbourTree, allNodes, nodeManning},
                           {&nodeCount, frictionParams}, {sizeof(nodeCount), sizeof(frictionParams)},
                           nodeCount);
          stage1Tree = neighbourTree;
        }
        lap(0);
        LayoutArgs flowLayout{mesh.activeCount, maxLevel};
        double boundaryRateStage1 = 0.0, boundaryRateStage2 = 0.0;
        double rk1Signed = 0.0, rk1Negative = 0.0;
        const auto *invalid = static_cast<const uint32_t *>(localDt.contents);
        float hwNextDt = 0.0f;
        if (haar) {
          // HWFV1: one forward-Euler finite-volume stage (fv1_update.cu).
          // The leaf CFL values go to rhs[1]; localDt receives the
          // positivity check's invalid flags.
          float hwParams[4] = {dt, static_cast<float>(solver.InitTstep),
                               static_cast<float>(solver.cfl), 0.0f};
          runtime.dispatch("hwfv1_update",
              {state[0], mesh.active, mesh.neighbours, state[2], stage1Tree, rhs[1], boundaryFlux, scales, frictionFlags},
              {&flowLayout, &flowPhysics, hwParams, waveArgs},
              {sizeof(flowLayout), sizeof(flowPhysics), sizeof(hwParams), sizeof(waveArgs)}, mesh.activeCount);
          boundaryRateStage1 = boundaryRateStage2 = edgeRate();
          runtime.dispatch("mwdg2_positivity", {state[2], mesh.active, localDt},
              {&activeCount, &depthTolerance}, {sizeof(activeCount), sizeof(depthTolerance)},
              mesh.activeCount);
          for (uint32_t i = 0; i < mesh.activeCount; ++i)
            if (invalid[i]) {
              const uint32_t node = mesh.activeHost[i];
              const float *bad = static_cast<const float *>(state[2].contents) + node * 12;
              char message[256];
              std::snprintf(message, sizeof(message),
                  "HWFV1 update produced an invalid state at leaf %u (h=%g, qx=%g, qy=%g, dt=%g)",
                  node, double(bad[0]-bad[9]), double(bad[3]), double(bad[6]), double(dt));
              throw std::runtime_error(message);
            }
          const float *cfl = static_cast<const float *>(rhs[1].contents);
          hwNextDt = cfl[0];
          for (uint32_t i = 1; i < mesh.activeCount; ++i) hwNextDt = std::min(hwNextDt, cfl[i]);
        } else {
        applyCudaSlopeLimiter(state[0], stage1Tree);

        runtime.dispatch("mwdg2_rhs", {state[0], mesh.active, mesh.neighbours, rhs[0], stage1Tree, boundaryFlux, scales, frictionFlags},
            {&flowLayout, &flowPhysics, waveArgs}, {sizeof(flowLayout), sizeof(flowPhysics), sizeof(waveArgs)}, mesh.activeCount);
        if (solver.t == 0.0 && std::getenv("LISFLOOD_MWDG2_TRACE_INITIAL_RHS")) {
          const float *deriv = static_cast<const float *>(rhs[0].contents);
          const uint32_t *packed = static_cast<const uint32_t *>(mesh.neighbours.contents);
          const uint32_t *faces = packed + ((mesh.activeCount*4u+4u)&~3u);
          float largest = 0.0f; uint32_t at = 0;
          for (uint32_t i=0; i<mesh.activeCount; ++i) {
            const float v = std::max({std::fabs(deriv[i*9+6]),
                                      std::fabs(deriv[i*9+7]),
                                      std::fabs(deriv[i*9+8])});
            if (v > largest) { largest=v; at=i; }
          }
          const uint32_t node=mesh.activeHost[at]; uint32_t x=0,y=0;
          const uint32_t span=nodeSpan(node,maxLevel,&x,&y);
          const float *local=static_cast<const float *>(state[0].contents)+node*12;
          std::fprintf(stderr,"MWDG2 first-RHS qy max=%g index=%u leaf=%u cell=(%u,%u,%u) rhs=(%.9g %.9g %.9g)\n",
              double(largest),at,node,x,y,span,double(deriv[at*9+6]),
              double(deriv[at*9+7]),double(deriv[at*9+8]));
          std::fprintf(stderr,"  local eta=(%.9g %.9g %.9g) bed=(%.9g %.9g %.9g) qx=(%.9g %.9g %.9g) qy=(%.9g %.9g %.9g)\n",
              double(local[0]),double(local[1]),double(local[2]),
              double(local[9]),double(local[10]),double(local[11]),
              double(local[3]),double(local[4]),double(local[5]),
              double(local[6]),double(local[7]),double(local[8]));
          for (uint32_t face=0; face<4; ++face) {
            const uint32_t seg=packed[at*4+face];
            const uint32_t other=faces[seg*4];
            if (other < nodeCount) {
              uint32_t ox=0,oy=0; const uint32_t os=nodeSpan(other,maxLevel,&ox,&oy);
              const float *c=static_cast<const float *>(scales.contents)+other*12;
              std::fprintf(stderr,"  face%u neighbor=%u cell=(%u,%u,%u) eta=(%.8g %.8g %.8g) bed=(%.8g %.8g %.8g) qx=(%.8g %.8g %.8g) qy=(%.8g %.8g %.8g)\n",
                  face,other,ox,oy,os,double(c[0]),double(c[1]),double(c[2]),
                  double(c[9]),double(c[10]),double(c[11]),
                  double(c[3]),double(c[4]),double(c[5]),
                  double(c[6]),double(c[7]),double(c[8]));
            } else std::fprintf(stderr,"  face%u ghost=%u\n",face,other);
          }
        }
        boundaryRateStage1 = edgeRate();
        StageArgs rk1{dt, 0.0f};
        runtime.dispatch("mwdg2_rk", {state[0], state[0], rhs[0], mesh.active, state[1], scales},
            {&activeCount, &rk1}, {sizeof(activeCount), sizeof(rk1)}, mesh.activeCount);
        if (verbose && solver.t == 0.0)
          rk1Signed = signedLeafVolume(mesh, static_cast<const float *>(state[1].contents),
              maxLevel, static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz),
              cellArea, &rk1Negative);
        runtime.dispatch("mwdg2_positivity", {state[1], mesh.active, localDt},
            {&activeCount, &depthTolerance}, {sizeof(activeCount), sizeof(depthTolerance)},
            mesh.activeCount);
        for (uint32_t i = 0; i < mesh.activeCount; ++i)
          if (invalid[i]) {
            const uint32_t node = mesh.activeHost[i];
            const float *bad = static_cast<const float *>(state[1].contents) + node * 12;
          const float *before = static_cast<const float *>(state[0].contents) + node * 12;
          const float *derivative = static_cast<const float *>(rhs[0].contents) + i * 9;
          const uint32_t *packed=static_cast<const uint32_t *>(mesh.neighbours.contents);
          const uint32_t *neighbour=packed+((mesh.activeCount*4+4u)&~3u);
          const uint32_t begin=packed[i*4],end=packed[i*4+1];
          std::fprintf(stderr, "MWDG2 failure trace step=%lld t=%.9g leaf=%u localCFL=%.9g dt=%.9g\n",
              static_cast<long long>(solver.itCount), static_cast<double>(solver.t),
              node, static_cast<double>(previousDt), static_cast<double>(dt));
          std::fprintf(stderr, "  preState h=(%.9g %.9g %.9g) qx=(%.9g %.9g %.9g) qy=(%.9g %.9g %.9g) z0=%.9g\n",
              before[0]-before[9], before[1]-before[10], before[2]-before[11],
              before[3], before[4], before[5], before[6], before[7], before[8], before[9]);
          std::fprintf(stderr, "  rhs h=(%.9g %.9g %.9g) qx=(%.9g %.9g %.9g) qy=(%.9g %.9g %.9g)\n",
              derivative[0], derivative[1], derivative[2], derivative[3],
              derivative[4], derivative[5], derivative[6], derivative[7], derivative[8]);
          std::fprintf(stderr, "  neighbours");
          const float *tree = static_cast<const float *>(scales.contents);
          for (uint32_t face = 0; face < 4; ++face) {
            for (uint32_t part = packed[i*4+face]; part < packed[i*4+face+1]; ++part) {
              const uint32_t other = neighbour[part*4];
              if (other >= 0xfffffffcu) continue;
              const float *c = tree + other*12;
              std::fprintf(stderr, " [%u h=%g qx=%g qy=%g z=%g]", other,
                  c[0]-c[9], c[3], c[6], c[9]);
            }
          }
          std::fprintf(stderr, "\n");
            char message[256];
            std::snprintf(message, sizeof(message),
                "DG2 stage 1 produced an invalid state at leaf %u (eta-z=%g, eta=%g, z=%g, qx=%g, qy=%g, dt=%g)",
                node, bad[0] - bad[9], bad[0], bad[9], bad[3], bad[6], dt);
            throw std::runtime_error(message);
          }

        // CUDA reinserts the Euler-stage leaves and re-encodes the hierarchy
        // before it constructs the second-stage neighbor coefficients. In an
        // ungraded tree, a coarse leaf adjacent to fine leaves reads their
        // common ancestor at its own level; stale parent scales would erase
        // the evolving coarse/fine flux on RK stage two.
        LayoutArgs stageExpand{finestCount,maxLevel};
        runtime.dispatch("mwdg2_mini_expand_finest",{state[1],mesh.owners,scales},
                         &stageExpand,sizeof(stageExpand),finestCount);
        for(int level=static_cast<int>(maxLevel)-1;level>=0;--level) {
          uint32_t encodeArgs[2]={levelOffset(static_cast<uint32_t>(level)),
                                  1u << (2*static_cast<uint32_t>(level))};
          runtime.dispatch(haar ? "hwfv1_encode_level" : "mwdg2_encode_level",{scales,details},encodeArgs,
                           sizeof(encodeArgs),encodeArgs[1]);
        }
        runtime.copy(state[1], scales, size_t(nodeCount) * 12 * sizeof(float));
        applyBoundaryCells(state[1]);
        applyCudaSlopeLimiter(state[1], scales);
        runtime.dispatch("mwdg2_rhs", {state[1], mesh.active, mesh.neighbours, rhs[1], scales, boundaryFlux, scales, frictionFlags},
            {&flowLayout, &flowPhysics, waveArgs}, {sizeof(flowLayout), sizeof(flowPhysics), sizeof(waveArgs)}, mesh.activeCount);
        boundaryRateStage2 = edgeRate();
        StageArgs rk2{dt, 0.5f};
        runtime.dispatch("mwdg2_rk", {state[0], state[1], rhs[1], mesh.active, state[2], scales},
            {&activeCount, &rk2}, {sizeof(activeCount), sizeof(rk2)}, mesh.activeCount);
        runtime.dispatch("mwdg2_positivity", {state[2], mesh.active, localDt},
            {&activeCount, &depthTolerance}, {sizeof(activeCount), sizeof(depthTolerance)},
            mesh.activeCount);
        if (verbose && solver.t == 0.0) {
          double signed2 = 0.0, negative2 = 0.0;
          signed2 = signedLeafVolume(mesh, static_cast<const float *>(state[2].contents),
              maxLevel, static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz),
              cellArea, &negative2);
          std::printf("MWDG2 volume audit first step: dt=%.9g RK1 signed=%.9g negative=%.9g RK2 signed=%.9g negative=%.9g\n",
                      dt, rk1Signed, rk1Negative, signed2, negative2);
        }
        for (uint32_t i = 0; i < mesh.activeCount; ++i)
          if (invalid[i]) {
            const uint32_t node = mesh.activeHost[i];
            const float *bad = static_cast<const float *>(state[2].contents) + node * 12;
            char message[256];
            std::snprintf(message, sizeof(message),
                "DG2 stage 2 produced invalid state at leaf %u (eta-z=%g, eta=%g, z=%g, qx=%g, qy=%g, dt=%g)",
                node, double(bad[0]-bad[9]), double(bad[0]), double(bad[9]),
                double(bad[3]), double(bad[6]), double(dt));
            throw std::runtime_error(message);
          }
        }
        const double stepRKVolume = traceMass ? waterVolume(mesh,
            static_cast<const float *>(state[2].contents), maxLevel,
            static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz), cellArea) : 0.0;
        // CUDA (dg2_update/fv1_update) seeds every leaf's CFL value with
        // initial_tstep and overwrites it only for wet leaves (h >= tol_h)
        // after the update; get_dt_CFL takes the minimum over all leaves, so a
        // dry leaf caps the next step at initial_tstep.
        const float cflFallback = static_cast<float>(solver.InitTstep);
        float nextDt = hwNextDt;
        if (!haar) {
          runtime.dispatch("mwdg2_cfl", {state[2], mesh.active, localDt},
              {&layout, &cflPhysics, &depthTolerance, &cflFallback},
              {sizeof(layout), sizeof(cflPhysics), sizeof(depthTolerance), sizeof(cflFallback)},
              mesh.activeCount);
          nextDt = cflFallback;
          const auto *nextCflValues = static_cast<const float *>(localDt.contents);
          for (uint32_t i = 0; i < mesh.activeCount; ++i)
            nextDt = std::min(nextDt, nextCflValues[i]);
        }
        if (std::getenv("LISFLOOD_MWDG2_TRACE_Q")) {
          const float *trace = static_cast<const float *>(state[2].contents);
          float maxAbsQ = 0.0f;
          uint32_t maxNode = 0, maxIndex = 0;
          for (uint32_t i = 0; i < mesh.activeCount; ++i) {
            const uint32_t node = mesh.activeHost[i];
            const float *c = trace + node * 12;
            const float value = std::max(std::fabs(c[3]), std::fabs(c[6]));
            if (value > maxAbsQ) { maxAbsQ = value; maxNode = node; maxIndex = i; }
          }
          static float nextQThreshold = 10.0f;
          if (maxAbsQ >= nextQThreshold) {
            const float *c = trace + maxNode * 12;
            std::fprintf(stderr,
                "MWDG2 q-growth step=%lld t=%.9g dt=%.9g maxQ=%.9g leaf=%u h=%.9g qx=%.9g qy=%.9g z=%.9g\n",
                static_cast<long long>(solver.itCount), static_cast<double>(solver.t),
                static_cast<double>(dt), static_cast<double>(maxAbsQ), maxNode,
                static_cast<double>(c[0]-c[9]), static_cast<double>(c[3]),
                static_cast<double>(c[6]), static_cast<double>(c[9]));
            if (maxAbsQ >= 1000.0f) {
              const uint32_t *packed=static_cast<const uint32_t *>(mesh.neighbours.contents);
              const uint32_t *near=packed+((mesh.activeCount*4+4u)&~3u);
              std::fprintf(stderr, "  modes eta=(%.7g %.7g %.7g) qx=(%.7g %.7g %.7g) qy=(%.7g %.7g %.7g) z=(%.7g %.7g %.7g)\n",
                  c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11]);
              for (uint32_t face=0; face<4; ++face) {
                const uint32_t other=near[packed[maxIndex*4+face]*4];
                if (other<nodeCount) {
                  const float *n=trace+other*12;
                  std::fprintf(stderr,"  face%u neighbor=%u h=%.7g qx=%.7g qy=%.7g z=%.7g\n",
                      face,other,n[0]-n[9],n[3],n[6],n[9]);
                } else std::fprintf(stderr,"  face%u ghost=%u\n",face,other);
              }
            }
            nextQThreshold *= 10.0f;
          }
        }
        const double boundaryRate = 0.5 * (boundaryRateStage1 + boundaryRateStage2);
        if (boundaryRate >= 0.0) {
          boundCs.Qout += static_cast<NUMERIC_TYPE>(boundaryRate);
          boundCs.VolOutMT += static_cast<NUMERIC_TYPE>(boundaryRate * dt);
        } else {
          boundCs.Qin += static_cast<NUMERIC_TYPE>(-boundaryRate);
          boundCs.VolInMT += static_cast<NUMERIC_TYPE>(-boundaryRate * dt);
        }

        // Clear NoData leaves again after the RK update so scheduled outputs
        // and mass diagnostics cannot expose transient water deposited there
        // during the just-completed step. CUDA clears at loop entry; the next
        // loop would otherwise clear this same state before advancing again.
        drainNoData(state[2]);

        double stepLossVolume = 0.0;
        lap(1);
        if (lossesEnabled) {
          uint32_t lossLayout[4] = {mesh.activeCount, maxLevel,
                                    static_cast<uint32_t>(pars.xsz),
                                    static_cast<uint32_t>(pars.ysz)};
          runtime.dispatch("mwdg2_loss_prepare",
              {state[2], mesh.owners, lossDepth, lossQx, lossQy},
              lossLayout, sizeof(lossLayout), static_cast<uint32_t>(rasterCells));
          @autoreleasepool {
            id<MTLCommandBuffer> lossCommand = [runtime.queue commandBuffer];
            id<MTLComputeCommandEncoder> lossEncoder = [lossCommand computeCommandEncoder];
            losses.encode(lossEncoder, lossDepth, lossQx, lossQy, true, 0, 0,
                          0.0f, lossDem, static_cast<float>(pars.nodata_elevation),
                          static_cast<float>(solver.DepthThresh), dt, solver.t);
            [lossEncoder endEncoding]; [lossCommand commit]; [lossCommand waitUntilCompleted];
            if (lossCommand.status == MTLCommandBufferStatusError)
              throw std::runtime_error(lossCommand.error.localizedDescription.UTF8String);
          }
          losses.collect();
          uint32_t restrictArgs[4] = {mesh.activeCount, maxLevel,
              static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz)};
          float restrictPhysics[2] = {static_cast<float>(pars.nodata_elevation),
                                      static_cast<float>(solver.DepthThresh)};
          id<MTLBuffer> leafRemoved = runtime.buffer(size_t(mesh.activeCount) * sizeof(float));
          runtime.dispatch("mwdg2_restrict_losses",
              {state[2], mesh.active, lossDepth, lossDem, losses.residual_buffer(), leafRemoved},
              {restrictArgs, restrictPhysics}, {sizeof(restrictArgs), sizeof(restrictPhysics)},
              mesh.activeCount);
          // Book what the leaf levels actually lost; the raster tally also
          // counts residual carried from earlier steps.
          losses.take_interval_volume();
          const float *taken = static_cast<const float *>(leafRemoved.contents);
          for (uint32_t i = 0; i < mesh.activeCount; ++i)
            stepLossVolume += static_cast<double>(taken[i]) * cellArea;
          boundCs.VolOutMT += static_cast<NUMERIC_TYPE>(stepLossVolume);
          pars.InfilTotalLoss += static_cast<NUMERIC_TYPE>(stepLossVolume);
          cumulativeLossVolume += stepLossVolume;
          integratedLossTime += static_cast<double>(dt);
        }

        lap(2);
        // Reinsert every evolved leaf on the finest grid, then encode it up
        // the full hierarchy before selecting the next adaptive topology.
        LayoutArgs expand{finestCount, maxLevel};
        std::memset(expandFlags.contents, 0, size_t(nodeCount) * sizeof(uint32_t));
        // CUDA's restriction by default (exact parity). The depth-preserving
        // refinement (no eta < z on finest cells of a coarse wet leaf on a
        // steep bed) is an opt-in diagnostic: LISFLOOD_ADAPTIVE_POSITIVE_EXPAND.
        if (positiveExpand)
          runtime.dispatch("mwdg2_expand_flags", {state[2], mesh.owners, scales, expandFlags},
                           &expand, sizeof(expand), finestCount);
        runtime.dispatch("mwdg2_expand_finest_positive", {state[2], mesh.owners, scales, expandFlags},
                         &expand, sizeof(expand), finestCount);
        const bool traceMeshMemory = verbose && (solver.itCount % 25 == 0);
        if (traceMeshMemory)
          std::printf("MWDG2 mesh memory before select: step=%lld %.3f GiB\n",
              static_cast<long long>(solver.itCount),
              static_cast<double>(runtime.device.currentAllocatedSize) /
                  (1024.0 * 1024.0 * 1024.0));
        {
          const double stepEnd = static_cast<double>(solver.t) + dt;
          const float dtNext = static_cast<float>(std::min<double>(nextDt, solver.Sim_Time - stepEnd));
          if (dtNext > 0.0f)
            boundCs.VolInMT += static_cast<NUMERIC_TYPE>(
                insertPointSources(static_cast<float>(stepEnd) + dtNext, dtNext));
        }
        AdaptiveMesh next = meshBuilder.select(scales, details, preflags,
                                               maxLevel, nodeCount,
                                               static_cast<float>(solver.epsilon),
                                               state[2], mesh.active, mesh.activeCount);
        if (traceMeshMemory)
          std::printf("MWDG2 mesh memory after select: step=%lld %.3f GiB\n",
              static_cast<long long>(solver.itCount),
              static_cast<double>(runtime.device.currentAllocatedSize) /
                  (1024.0 * 1024.0 * 1024.0));
        mesh = AdaptiveMesh{};
        if (traceMeshMemory)
          std::printf("MWDG2 mesh memory after old release: step=%lld %.3f GiB\n",
              static_cast<long long>(solver.itCount),
              static_cast<double>(runtime.device.currentAllocatedSize) /
                  (1024.0 * 1024.0 * 1024.0));
        lap(3);
        mesh = std::move(next);
        applyAdaptiveBoundaries(mesh, bc, maxLevel, pars.xsz, pars.ysz, bcCellHost, westInflow);
        if (const char *dumpStep = std::getenv("LISFLOOD_DIAG_DUMP_STEP"))
          if (solver.itCount + 2 == std::atoll(dumpStep)) {  // mesh of that (1-based) step
            std::printf("DIAG_LEAVES step %lld maxes %.9g %.9g %.9g %.9g n %u:",
                std::atoll(dumpStep), meshBuilder.lastNorms[0], meshBuilder.lastNorms[1],
                meshBuilder.lastNorms[2], meshBuilder.lastNorms[3], mesh.activeCount);
            for (uint32_t node : mesh.activeHost) std::printf(" %u", node);
            std::printf("\n");
          }
        if (traceMass) {
          const double selectedVolume = waterVolume(mesh,
              static_cast<const float *>(scales.contents), maxLevel,
              static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz), cellArea);
          // RK change (fluxes, rain, edges) and the change from losses,
          // point sources and remeshing, summed over the run.
          traceUpdate += stepRKVolume - stepStartVolume;
          traceRemesh += selectedVolume - stepRKVolume + stepLossVolume;
          traceLoss += stepLossVolume;
          if (solver.itCount % 40 == 0)
            std::fprintf(stderr, "MWDG2 mass budget step=%lld t=%.6g volume=%.9g update=%.9g remesh+sources=%.9g losses=%.9g rain=%.9g\n",
                static_cast<long long>(solver.itCount), static_cast<double>(solver.t),
                selectedVolume, traceUpdate, traceRemesh, traceLoss, traceRain);
        }
        if (traceMass && solver.itCount < 20) {
          const double selectedVolume = waterVolume(mesh,
              static_cast<const float *>(scales.contents), maxLevel,
              static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz), cellArea);
          std::fprintf(stderr,
              "MWDG2 mass trace step=%lld t=%.9g dt=%.9g before=%.9g rk=%.9g selected=%.9g leaves=%u->%u\n",
              static_cast<long long>(solver.itCount), static_cast<double>(solver.t),
              static_cast<double>(dt), stepStartVolume, stepRKVolume,
              selectedVolume, activeCount, mesh.activeCount);
        }
        // Rebuild the host-side Manning values for the new active leaves.
        manning.assign(mesh.activeCount, static_cast<float>(pars.FPn));
        const auto *newActive = mesh.activeHost.data();
        for (uint32_t i = 0; i < mesh.activeCount; ++i) {
          uint32_t node = newActive[i], start = 0, count = 1, level = 0;
          while (node >= start + count) { start += count; count *= 4; ++level; }
          uint32_t span = 1u << (maxLevel - level), code = node - start, cx = 0, cy = 0;
          for (uint32_t bit = 0; bit < 16; ++bit) {
            cx |= ((code >> (2 * bit)) & 1u) << bit;
            cy |= ((code >> (2 * bit + 1)) & 1u) << bit;
          }
          uint32_t x = cx * span + span / 2, y = cy * span + span / 2;
          if (arrays.Manningsn && x < static_cast<uint32_t>(pars.xsz) &&
              y < static_cast<uint32_t>(pars.ysz))
            manning[i] = static_cast<float>(arrays.Manningsn[size_t(pars.ysz - 1 - y) * pars.xsz + x]);
        }
        // Grow leaf-count workspaces only when the adaptive mesh grows. The
        // old code reallocated these fixed-size Metal buffers every step,
        // causing the driver to retain gigabytes during long high-resolution
        // runs even though the active count was unchanged.
        if (mesh.activeCount > workCapacity) {
          workCapacity = mesh.activeCount;
          for (int i = 0; i < 2; ++i)
            rhs[i] = runtime.buffer(size_t(workCapacity) * 9 * sizeof(float));
          localDt = runtime.buffer(size_t(workCapacity) * sizeof(float));
          leafManning = runtime.buffer(size_t(workCapacity) * sizeof(float));
          boundaryFlux = runtime.buffer(size_t(workCapacity) * sizeof(float));
          bcCells = runtime.buffer(size_t(workCapacity) * sizeof(int));
          frictionFlags = runtime.buffer(size_t(workCapacity) * sizeof(uint32_t));
          bcDelta = runtime.buffer(size_t(workCapacity) * sizeof(float));
          if (pars.limit_slopes == ON)
            limitedSlopes = runtime.buffer(size_t(workCapacity) * 6 * sizeof(float));
        }
        std::memcpy(leafManning.contents, manning.data(), manning.size() * sizeof(float));
        std::memcpy(bcCells.contents, bcCellHost.data(), bcCellHost.size() * sizeof(int));
        markPaddingLeaves();
        // RK workspaces are sized to the fixed hierarchy and reused across
        // regrids; only leaf-count-sized work buffers above need replacement.

        // Verbose memory telemetry makes adaptive buffer lifetime regressions
        // visible on long runs without requiring Instruments.
        if (verbose && (solver.itCount % 25 == 0 || solver.itCount == 1))
          std::printf("MWDG2 Metal buffers: step=%lld leaves=%u deviceAllocated=%.3f GiB\n",
              static_cast<long long>(solver.itCount), mesh.activeCount,
              static_cast<double>(runtime.device.currentAllocatedSize) /
                  (1024.0 * 1024.0 * 1024.0));

        lap(4);
        if (profile && solver.itCount % 200 == 199)
          std::printf("MWDG2 profile (s) setup %.2f flow %.2f losses %.2f select %.2f after %.2f\n",
                      phase[0], phase[1], phase[2], phase[3], phase[4]);
        solver.MinTstep = std::min(solver.MinTstep, static_cast<NUMERIC_TYPE>(dt));
        solver.Tstep = static_cast<NUMERIC_TYPE>(dt);
        solver.t += static_cast<NUMERIC_TYPE>(dt);
        // dt is a float from the Metal CFL calculation. Snap when the float
        // conversion leaves time a few ulps short of the scheduled event.
        const double scheduleTolerance = 8.0 * std::numeric_limits<float>::epsilon() *
            std::max(1.0, std::fabs(deadline));
        if (std::fabs(static_cast<double>(solver.t) - deadline) <= scheduleTolerance)
          solver.t = static_cast<NUMERIC_TYPE>(deadline);
        solver.itCount += 1;
        previousDt = nextDt;
        previousStepDt = dt;
        if (diagSteps)  // same line as cuda_adaptive_simulate.cu with LISFLOOD_DIAG_STEPS
          std::printf("Elements: %u, compression: %f%%, time step: %.15f, steps: %lld, sim time: %f\n",
              mesh.activeCount, 100.0 - 100.0 * mesh.activeCount / (double(pars.xsz) * pars.ysz),
              double(nextDt), static_cast<long long>(solver.itCount), double(solver.t));
        if (verbose) std::printf("t=%f\tdt=%f\n", (double)(solver.t - dt), (double)dt);

        // Project cell-centred output for standard LISFLOOD output/statistics.
        GeometryArgs outputGeometry{static_cast<uint32_t>(pars.xsz),
                                    static_cast<uint32_t>(pars.ysz), side, maxLevel};
        runtime.dispatch("mwdg2_project_raster", {scales, mesh.owners, projected},
                         &outputGeometry, sizeof(outputGeometry),
                         static_cast<uint32_t>(rasterCells));
        const float *out = static_cast<const float *>(projected.contents);
        if (verbose && solver.itCount == 1) {
          double rasterSigned = 0.0, rasterNegative = 0.0;
          for (size_t i = 0; i < rasterCells; ++i) {
            const double h = out[i];
            rasterSigned += h * cellArea;
            rasterNegative += std::max(0.0, -h) * cellArea;
          }
          double leafNegative = 0.0;
          const double leafSigned = signedLeafVolume(mesh,
              static_cast<const float *>(scales.contents), maxLevel,
              static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz),
              cellArea, &leafNegative);
          std::printf("MWDG2 volume audit t=%.9g: leaf signed=%.9g negative=%.9g raster signed=%.9g negative=%.9g clamped raster=%.9g m3\n",
              static_cast<double>(solver.t), leafSigned, leafNegative,
              rasterSigned, rasterNegative, rasterSigned + rasterNegative);
        }
        for (size_t i = 0; i < rasterCells; ++i) {
          // CUDA adaptive raster projection preserves signed DG2 cell-centre
          // depths (including undershoots); keep the raw value for parity.
          arrays.H[i] = out[i];
          arrays.HU[i] = out[rasterCells + i];
          arrays.HV[i] = out[2 * rasterCells + i];
          if (arrays.maxH) arrays.maxH[i] = std::max(arrays.maxH[i], arrays.H[i]);
        }

        if (pars.cumulative == ON && solver.t >= pars.MassTotal) {
          // cuda/adaptive write_cumulative_data: one line per mass interval
          char cumuPath[512];
          std::snprintf(cumuPath, sizeof(cumuPath), "%s.cumu", filenames.resrootname);
          FILE *cumu = std::fopen(cumuPath, cumuStarted ? "a" : "w");
          if (cumu) {
            if (!cumuStarted)
              std::fprintf(cumu, "simtime,runtime_mra,runtime_solver,runtime_total,dt,reduction\n");
            const double mra = phase[0] + phase[3], flow = phase[1] + phase[2] + phase[4];
            std::fprintf(cumu, "%f,%f,%f,%f,%f,%f\n", double(solver.t), mra, flow, mra + flow,
                double(dt), (100.0 - 100.0 * mesh.activeCount / (double(pars.xsz) * pars.ysz)) / 100.0);
            std::fclose(cumu);
            cumuStarted = true;
          }
        }
        if (solver.t >= pars.MassTotal) {
          // The shared raster statistics routine clamps negative projected
          // depths and resets both the previous volume and interval fluxes.
          // Preserve the adaptive solver's signed DG2 mean and flux ledger
          // before calling it, then restore the physically relevant balance.
          const NUMERIC_TYPE previousSignedVolume = solver.vol1;
          const NUMERIC_TYPE intervalIn = boundCs.VolInMT;
          const NUMERIC_TYPE intervalOut = boundCs.VolOutMT;
          const NUMERIC_TYPE intervalMass = pars.MassInt;
          update_mass_stats(&states, &pars, &solver, &boundCs, &arrays);
          // The MWDG2 conserved quantity is the signed sum of leaf mean
          // depths. Raster output clamps negative reconstructed depths to
          // zero, so DomainVol over that output would invent water whenever
          // a DG2 mean becomes negative (CUDA keeps that mean through RK).
          solver.vol2 = static_cast<NUMERIC_TYPE>(signedLeafVolume(mesh,
              static_cast<const float *>(scales.contents), maxLevel,
              static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz),
              cellArea));
          solver.Verror = intervalIn - intervalOut -
                          (solver.vol2 - previousSignedVolume);
          solver.Qerror = solver.Verror / intervalMass;
          solver.vol1 = solver.vol2;
          if (files) write_mass_stats(files, &pars, &solver, &boundCs);
          writeStageRecord(stageOutput, stage, pars, solver, arrays);
          if (lossesEnabled) {
            char lossPath[512];
            std::snprintf(lossPath, sizeof(lossPath), "%s.loss", filenames.resrootname);
            FILE *lossFile = std::fopen(lossPath, firstLossRecord ? "w" : "a");
            if (!lossFile) throw std::runtime_error(std::string("cannot open loss diagnostic: ") + lossPath);
            if (firstLossRecord)
              std::fprintf(lossFile, "time,integrated_time,loss_volume,water_volume\n");
            const double water = waterVolume(mesh,
                static_cast<const float *>(scales.contents), maxLevel,
                static_cast<uint32_t>(pars.xsz), static_cast<uint32_t>(pars.ysz), cellArea);
            std::fprintf(lossFile, "%.17g,%.17g,%.17g,%.17g\n",
                static_cast<double>(solver.t), integratedLossTime,
                cumulativeLossVolume, water);
            std::fclose(lossFile);
            firstLossRecord = false;
          }
        }
        if (solver.t >= pars.SaveTotal) write_solution(&filenames, &states, &pars, &solver, &arrays);
        }
      }

      // Force a final output with the final state and max-depth record.
      GeometryArgs outputGeometry{static_cast<uint32_t>(pars.xsz),
                                  static_cast<uint32_t>(pars.ysz), side, maxLevel};
      runtime.dispatch("mwdg2_project_raster", {scales, mesh.owners, projected},
                       &outputGeometry, sizeof(outputGeometry),
                       static_cast<uint32_t>(rasterCells));
      const float *out = static_cast<const float *>(projected.contents);
      for (size_t i = 0; i < rasterCells; ++i) {
        arrays.H[i] = out[i];
        arrays.HU[i] = out[rasterCells + i];
        arrays.HV[i] = out[2 * rasterCells + i];
      }
      solver.vol2 = waterVolume(mesh, static_cast<const float *>(scales.contents),
                                maxLevel, static_cast<uint32_t>(pars.xsz),
                                static_cast<uint32_t>(pars.ysz), cellArea);
      if (verbose)
        std::printf("Finished native adaptive MWDG2: leaves=%u, initial water volume=%.9g m3, final=%.9g m3\n",
                    mesh.activeCount, initialVolume, solver.vol2);
      write_solution(&filenames, &states, &pars, &solver, &arrays);
      write_max_field(&filenames, &states, &pars, &arrays);
      if (stageOutput) std::fclose(stageOutput);
    } catch (const std::exception &error) {
      std::fprintf(stderr, "ERROR: native Metal MWDG2: %s\n", error.what());
      std::exit(EXIT_FAILURE);
    }
  }
}

} } } // namespace lis::metal::mwdg2
#endif
