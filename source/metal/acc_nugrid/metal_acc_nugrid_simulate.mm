#ifdef __APPLE__
// ACC on the static non-uniform grid (ACC_NUGRID) on the Metal GPU, ported
// from cuda/acc_nugrid/cuda_acc_nugrid_simulate.cu. The mesh, the neighbour
// records and the interfaces are built once on the host exactly as the CUDA
// kernels build them (get_modal_values, preflag_details, preflag_topo,
// get_reg_tree, traverse_tree_of_sig_details, find_nonuniform_neighbours,
// count_interfaces_per_neighbours, find_interfaces); the time loop runs on
// the GPU in the CUDA order (drain, rain, dt, sources and boundaries,
// discharges, depths, losses).
// multiwavelet filters first: the header defines and then removes C()
#include "../adaptive/mwdg2_transform.h"
#include "metal_acc_nugrid_simulate.h"
#include "../losses/metal_losses.h"
#include "../adaptive/metal_adaptive_bc.h"
#include "../../rain/rain.h"
#include "../../rain/rain.tpp"
#include "../../swe/output.h"
#include "../../swe/stats.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lis { namespace metal { namespace acc_nugrid {
namespace {

using mwdg2::AdaptiveSeries;

struct Leaf {      // NugLeaf in metal_acc_nugrid.metal
  float h, z, n, dx;
  uint32_t li, lj, x0, y0;
  uint32_t span, recStart, recCount, level;
};
struct Record {    // NugRecord
  uint32_t owner;
  int32_t nghbr;
  uint32_t dir;
  float length;
  float ownerFlag, nghbrFlag;
  uint32_t itfStart, itfCount;
  float dxSum;
  int32_t side;
  uint32_t bound, pad;
};
struct BcArgs { int32_t type[4]; float value[4]; };
static_assert(sizeof(Leaf) == 48 && sizeof(Record) == 48, "layout must match the kernels");

enum { DIR_N = 1, DIR_E = 2, DIR_S = 3, DIR_W = 4 };

uint32_t dilate(uint32_t v) {
  v = (v | (v << 8)) & 0x00ff00ffu;
  v = (v | (v << 4)) & 0x0f0f0f0fu;
  v = (v | (v << 2)) & 0x33333333u;
  return (v | (v << 1)) & 0x55555555u;
}
uint32_t contract(uint32_t v) {
  v &= 0x55555555u;
  v = (v | (v >> 1)) & 0x33333333u;
  v = (v | (v >> 2)) & 0x0f0f0f0fu;
  v = (v | (v >> 4)) & 0x00ff00ffu;
  return (v | (v >> 8)) & 0x0000ffffu;
}
uint32_t levelOffset(uint32_t level) {
  return static_cast<uint32_t>((((uint64_t)1 << (2 * level)) - 1) / 3);
}

// ESRI ASCII raster, row 0 = north, as cuda/acc_nugrid read_raster_file:
// a missing file keeps the defaults; the file's NODATA_value is replaced by
// `noData` (pars.nodata_elevation, by default a 1e7 m wall, for the bed).
struct Raster {
  int xsz = 0, ysz = 0;
  std::vector<double> v;
};
Raster readAscii(const std::string &path, bool required, double noData) {
  Raster r;
  std::ifstream in(path);
  if (!in) {
    if (required) throw std::runtime_error("ACC_NUGRID cannot open raster " + path);
    return r;
  }
  std::string key;
  double value = 0, fileNoData = 0;
  for (int i = 0; i < 6 && (in >> key >> value); ++i) {
    for (auto &c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (key == "ncols") r.xsz = static_cast<int>(value);
    else if (key == "nrows") r.ysz = static_cast<int>(value);
    else if (key == "nodata_value") fileNoData = value;
  }
  r.v.resize(size_t(r.xsz) * r.ysz);
  for (auto &x : r.v) {
    if (!(in >> x)) break;
    if (std::fabs(x - fileNoData) < 1e-5) x = noData;
  }
  return r;
}

// cuda/acc_nugrid Boundary.h: the first bcifile line for each side, the
// range clamped to the raster, FREE's value is the friction slope.
struct Side {
  int type = mwdg2::BC_CLOSED;
  int start = 0, end = -1;
  float inlet = 0.0f;
  AdaptiveSeries series;
  bool bound(int c) const { return c >= start && c <= end; }
};
void readSides(Side side[4], const Fnames &filenames, const Pars &pars) {
  if (!filenames.bcifilename[0]) return;
  const char dirs[4] = {'N', 'E', 'S', 'W'};
  for (int k = 0; k < 4; ++k) {
    FILE *fp = std::fopen(filenames.bcifilename, "r");
    if (!fp) throw std::runtime_error(std::string("cannot open bcifile ") + filenames.bcifilename);
    char str[255];
    while (std::fgets(str, sizeof(str), fp)) {
      char dir = 0, type[16] = {0}, extra[64] = {0};
      float lower = 0, upper = 0;
      if (std::sscanf(str, "%c %f %f %15s %63s", &dir, &lower, &upper, type, extra) < 4 ||
          dir != dirs[k])
        continue;
      const bool alongX = k == 0 || k == 2;
      const double lo = alongX ? pars.blx : pars.bly;
      const double hi = alongX ? pars.blx + pars.xsz * pars.dx : pars.tly;
      lower = static_cast<float>(std::min(std::max<double>(lower, lo), hi));
      upper = static_cast<float>(std::min(std::max<double>(upper, lo), hi));
      Side &s = side[k];
      s.type = mwdg2::adaptiveBcType(type);
      s.start = static_cast<int>((lower - lo) / pars.dx);
      s.end = static_cast<int>((upper - lo) / pars.dx) - 1;
      if (s.type == mwdg2::BC_FREE || s.type == mwdg2::BC_HFIX || s.type == mwdg2::BC_QFIX)
        s.inlet = std::strtof(extra, nullptr);
      if (s.type == mwdg2::BC_HVAR || s.type == mwdg2::BC_QVAR)
        s.series = mwdg2::readAdaptiveSeries(filenames.bdyfilename, extra);
      break;
    }
    std::fclose(fp);
  }
}

// RainfallUniform.h, including its row stepping (rate in m/s).
struct UniformRain {
  std::vector<std::pair<float, float>> rows;  // value (mm/h), time
  int multiplier = 1, declared = 0, row = 0;
  float value = 0.0f;
  void load(const char *path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot open rainfall file ") + path);
    std::string line, unit;
    std::getline(in, line);
    std::getline(in, line);
    std::istringstream head(line);
    head >> declared >> unit;
    multiplier = !unit.compare(0, 7, "seconds") ? 1 : !unit.compare(0, 7, "minutes") ? 60 : 3600;
    float v, t;
    while (std::getline(in, line)) {
      std::istringstream row(line);
      if (row >> v >> t) rows.emplace_back(v, t);
    }
  }
  void update(float now) {
    auto at = [&](int i) { return rows.empty() ? std::make_pair(0.0f, 0.0f)
                                               : rows[std::min<int>(std::max(i, 0), rows.size() - 1)]; };
    float v1 = at(row).first, t1 = at(row).second;
    float v2 = at(row + 1).first, t2 = at(row + 1).second;
    if (now > t2 * multiplier) {
      ++row;
      t1 = t2 * multiplier;
      v1 = v2;
      v2 = at(row + 1).first; t2 = at(row + 1).second;
    }
    if (row + 1 >= declared) { --row; value = 0.0f; }
    else value = (v1 + (v2 - v1) / ((t2 - t1) * multiplier) * (now - t1 * multiplier)) /
                 (1000.0f * 3600.0f);
  }
};

class Runtime {
public:
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  std::vector<std::string> names;
  std::vector<id<MTLComputePipelineState>> pipelines;
  explicit Runtime(int verbose) {
    device = MTLCreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("Metal device unavailable");
    queue = [device newCommandQueue];
    NSString *path = nil;
    NSString *exe = [[[NSProcessInfo processInfo] arguments][0] stringByDeletingLastPathComponent];
    for (NSString *candidate in @[ @"metal_acc_nugrid.metallib",
                                   [exe stringByAppendingPathComponent:@"metal_acc_nugrid.metallib"] ])
      if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) { path = candidate; break; }
    if (!path) throw std::runtime_error("metal_acc_nugrid.metallib not found next to executable");
    NSError *error = nil;
    library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if (!library) throw std::runtime_error("cannot load metal_acc_nugrid.metallib");
    if (verbose) std::printf("Metal device: %s\n", device.name.UTF8String);
  }
  id<MTLBuffer> buffer(size_t bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:std::max<size_t>(bytes, 4)
                                          options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("Metal ACC_NUGRID buffer allocation failed");
    std::memset(b.contents, 0, std::max<size_t>(bytes, 4));
    return b;
  }
  id<MTLComputePipelineState> pipeline(const char *name) {
    for (size_t i = 0; i < names.size(); ++i) if (names[i] == name) return pipelines[i];
    id<MTLFunction> f = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!f) throw std::runtime_error(std::string("missing ACC_NUGRID kernel ") + name);
    NSError *error = nil;
    id<MTLComputePipelineState> s = [device newComputePipelineStateWithFunction:f error:&error];
    if (!s) throw std::runtime_error(std::string("cannot create pipeline ") + name);
    names.emplace_back(name); pipelines.push_back(s);
    return s;
  }
  void dispatch(const char *name, const std::vector<id<MTLBuffer>> &buffers,
                const std::vector<const void *> &constants, const std::vector<size_t> &sizes,
                uint32_t count) {
    if (count == 0) return;
    id<MTLComputePipelineState> state = pipeline(name);
    @autoreleasepool {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
      [enc setComputePipelineState:state];
      for (NSUInteger i = 0; i < buffers.size(); ++i) [enc setBuffer:buffers[i] offset:0 atIndex:i];
      for (NSUInteger i = 0; i < constants.size(); ++i)
        [enc setBytes:constants[i] length:sizes[i] atIndex:buffers.size() + i];
      [enc dispatchThreadgroups:MTLSizeMake((count + 63) / 64, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [enc endEncoding]; [command commit]; [command waitUntilCompleted];
      if (command.status == MTLCommandBufferStatusError)
        throw std::runtime_error(command.error.localizedDescription.UTF8String);
    }
  }
};

} // namespace

void simulate(Fnames &filenames, Files *files, States &states, Pars &pars,
              Solver &solver, BoundCs &boundCs, Stage &stage, Arrays &arrays,
              int verbose) {
  @autoreleasepool {
  try {
    const int L = solver.L;
    if (L < 1 || L > 14) throw std::runtime_error("ACC_NUGRID needs L between 1 and 14");
    const int side = 1 << L;
    if (pars.xsz > side || pars.ysz > side)
      throw std::runtime_error("the raster (" + std::to_string(pars.xsz) + " x " +
          std::to_string(pars.ysz) + ") does not fit 2^L = " + std::to_string(side) +
          " cells; increase L (raster inputs fitting L >= 5 are needed with losses on CUDA)");
    const int xsz = pars.xsz, ysz = pars.ysz;
    const size_t rasterCells = size_t(xsz) * ysz;
    const uint32_t finestOffset = levelOffset(L);
    const uint32_t nodeCount = levelOffset(L + 1);
    const float nodata = static_cast<float>(pars.nodata_elevation);

    Runtime runtime(verbose);

    // get_modal_values: finest z0/z1x/z1y (DEM and its DG2 slope files), h
    // (start file) and n (nfile); padding: z0 = NoData, others 0.
    std::vector<float> z0(nodeCount, 0.0f), z1x(nodeCount, 0.0f), z1y(nodeCount, 0.0f);
    std::vector<float> h0(nodeCount, 0.0f), n0(nodeCount, 0.0f);
    for (uint32_t c = 0; c < uint32_t(side) * side; ++c) z0[finestOffset + c] = nodata;
    const std::string dem = filenames.demfilename;
    Raster rz = readAscii(dem, true, nodata), rx = readAscii(dem + "1x", false, nodata),
           ry = readAscii(dem + "1y", false, nodata);
    Raster rh, rn;
    const bool startfile = states.startfile == ON;
    const bool nonUniformN = filenames.nfilename[0] != 0;
    if (startfile) rh = readAscii(filenames.startfilename, true, 0.0);
    if (nonUniformN) rn = readAscii(filenames.nfilename, true, 0.0);
    auto put = [&](std::vector<float> &dst, const Raster &r) {
      for (int row = 0; row < r.ysz && row < side; ++row)
        for (int col = 0; col < r.xsz && col < side; ++col) {
          const uint32_t y = static_cast<uint32_t>(r.ysz - 1 - row);
          dst[finestOffset + (dilate(col) | (dilate(y) << 1))] =
              static_cast<float>(r.v[size_t(row) * r.xsz + col]);
        }
    };
    put(z0, rz); put(z1x, rx); put(z1y, ry);
    if (startfile) put(h0, rh);
    if (nonUniformN) put(n0, rn);

    Losses losses;
    const bool lossesEnabled = losses.setup(runtime.device, filenames, states, pars, &arrays,
                                            xsz, ysz, xsz, 0, 0, rasterCells, verbose);

    Side sides[4];
    readSides(sides, filenames, pars);
    // point sources: the 'P' lines of the bcifile, in order (LoadBCs keeps
    // the same order, so PS_TimeSeries[k] is the k-th source's series)
    struct Source { int x, y, type; float value; int leaf; };
    std::vector<Source> sources;
    if (filenames.bcifilename[0]) {
      FILE *fp = std::fopen(filenames.bcifilename, "r");
      char str[255];
      while (fp && std::fgets(str, sizeof(str), fp)) {
        char point = 0, type[16] = {0}, extra[64] = {0};
        float xs = 0, ys = 0;
        if (std::sscanf(str, "%c %f %f %15s %63s", &point, &xs, &ys, type, extra) < 4 || point != 'P')
          continue;
        Source s{static_cast<int>((xs - pars.blx) / pars.dx), static_cast<int>((ys - pars.bly) / pars.dx),
                 mwdg2::adaptiveBcType(type), std::strtof(extra, nullptr), -1};
        sources.push_back(s);
      }
      if (fp) std::fclose(fp);
    }

    // preflag_details (boundary cells, sources, gauges) and the loss preflag
    std::vector<uint8_t> sig(levelOffset(L), 0);
    auto flag = [&](int x, int y) {
      if (x < 0 || y < 0 || x >= side || y >= side) return;
      sig[levelOffset(L - 1) + ((dilate(x) | (dilate(y) << 1)) >> 2)] = 1;
    };
    for (int k = 0; k < 4; ++k) {
      if (sides[k].type == mwdg2::BC_CLOSED) continue;
      for (int c = sides[k].start; c <= sides[k].end; ++c) {
        if (k == 0) flag(c, ysz - 1);
        else if (k == 1) flag(xsz - 1, c);
        else if (k == 2) flag(c, 0);
        else flag(0, c);
      }
    }
    for (const auto &s : sources) flag(s.x, s.y);
    for (int i = 0; i < stage.Nstages; ++i)
      flag(static_cast<int>((stage.stage_loc_x[i] - pars.blx) / pars.dx),
           static_cast<int>((stage.stage_loc_y[i] - pars.bly) / pars.dx));
    if (lossesEnabled) {
      const float *rate = losses.uses_rate_grid() ? static_cast<const float *>(losses.rate_buffer().contents) : nullptr;
      const float *ks = losses.uses_greenampt() ? static_cast<const float *>(losses.ks_buffer().contents) : nullptr;
      const float *psi = losses.uses_greenampt() ? static_cast<const float *>(losses.psi_buffer().contents) : nullptr;
      const float *dth = losses.uses_greenampt() ? static_cast<const float *>(losses.dtheta_buffer().contents) : nullptr;
      const float *fmax = losses.uses_soil_depth() ? static_cast<const float *>(losses.fmax_buffer().contents) : nullptr;
      auto rasterIndex = [&](int x, int y) {
        if (x >= xsz || y >= ysz) return -1;
        const int idx = (ysz - 1 - y) * xsz + x;
        return std::fabs(static_cast<float>(arrays.DEM[idx]) - nodata) < 1e-6f ? -1 : idx;
      };
      for (uint32_t node = 0; node < levelOffset(L); ++node) {
        uint32_t level = 0;
        while (node >= levelOffset(level + 1)) ++level;
        const uint32_t code = node - levelOffset(level), span = 1u << (L - level);
        const int x0 = contract(code) * span, y0 = contract(code >> 1) * span;
        const int first = rasterIndex(x0, y0);
        bool different = false;
        for (int y = y0; y < y0 + int(span) && !different; ++y)
          for (int x = x0; x < x0 + int(span) && !different; ++x) {
            const int idx = rasterIndex(x, y);
            different = (idx < 0) != (first < 0);
            if (idx >= 0 && first >= 0) {
              if (rate) different |= rate[idx] != rate[first];
              if (ks) different |= ks[idx] != ks[first] || psi[idx] != psi[first] || dth[idx] != dth[first];
              if (fmax) different |= fmax[idx] != fmax[first];
            }
          }
        if (different) sig[node] = 1;
      }
    }

    // preflag_topo: multiwavelet encoding of the bed (and of h, n), node
    // significant if its largest bed detail / max(z, 1) reaches eps/2^(L-l)
    float maxZ = 1.0f;
    for (uint32_t c = 0; c < uint32_t(side) * side; ++c) maxZ = std::max(maxZ, z0[finestOffset + c]);
    const float eps = static_cast<float>(solver.epsilon);
    for (int level = L - 1; level >= 0; --level) {
      const float epsLocal = eps / float(1 << (L - level));
      for (uint32_t k = 0; k < (1u << (2 * level)); ++k) {
        const uint32_t node = levelOffset(level) + k, child = 4 * node + 1;
        float row[12];
        for (int r = 0; r < 12; ++r) {
          float sum = 0.0f;
          for (int c = 0; c < 4; ++c)
            sum += mwdg2_coefficient(r, c * 3) * z0[child + c] +
                   mwdg2_coefficient(r, c * 3 + 1) * z1x[child + c] +
                   mwdg2_coefficient(r, c * 3 + 2) * z1y[child + c];
          row[r] = 0.5f * sum;
        }
        z0[node] = row[0]; z1x[node] = row[1]; z1y[node] = row[2];
        float maxDetail = 0.0f;
        for (int r = 3; r < 12; ++r) maxDetail = std::max(maxDetail, std::fabs(row[r]));
        auto mean = [&](const std::vector<float> &v) {
          float sum = 0.0f;
          for (int c = 0; c < 4; ++c) sum += mwdg2_coefficient(0, c * 3) * v[child + c];
          return 0.5f * sum;
        };
        if (startfile) h0[node] = mean(h0);
        if (nonUniformN) n0[node] = mean(n0);
        if (maxDetail / maxZ >= epsLocal) sig[node] = 1;
      }
    }
    // get_reg_tree: a node is significant if one of its children is
    for (int level = L - 1; level >= 1; --level)
      for (uint32_t k = 0; k < (1u << (2 * level)); ++k)
        if (sig[levelOffset(level) + k]) sig[(levelOffset(level) + k - 1) / 4] = 1;

    // traverse_tree_of_sig_details + compaction: leaves in z order
    std::vector<Leaf> leaves;
    std::vector<uint32_t> leafNode;
    std::vector<int32_t> finestLeaf(size_t(side) * side, -1);   // row-major, y from south
    for (uint32_t code = 0; code < uint32_t(side) * side; ++code) {
      uint32_t level = 0, node = 0;
      while (level < uint32_t(L) && sig[node]) {
        ++level;
        node = levelOffset(level) + (code >> (2 * (L - level)));
      }
      if (leafNode.empty() || leafNode.back() != node) {
        const uint32_t local = node - levelOffset(level), span = 1u << (L - level);
        Leaf f{};
        f.h = startfile ? h0[node] : 0.0f;
        f.z = z0[node];
        f.n = n0[node];
        f.dx = static_cast<float>(pars.dx) * float(span);
        f.li = contract(local); f.lj = contract(local >> 1);
        f.x0 = f.li * span; f.y0 = f.lj * span; f.span = span; f.level = level;
        leaves.push_back(f);
        leafNode.push_back(node);
      }
      finestLeaf[size_t(contract(code >> 1)) * side + contract(code)] = int32_t(leaves.size() - 1);
    }
    const uint32_t leafCount = static_cast<uint32_t>(leaves.size());

    // find_nonuniform_neighbours: one record per leaf, direction and distinct
    // neighbour leaf; at the domain edge one per edge leaf (or per finest
    // cell of a prescribed boundary)
    std::vector<Record> records;
    std::vector<uint32_t> edgeRecords;
    for (uint32_t i = 0; i < leafCount; ++i) {
      Leaf &f = leaves[i];
      f.recStart = static_cast<uint32_t>(records.size());
      const int span = int(f.span);
      const int xsw = int(f.x0), ysw = int(f.y0), xne = xsw + span - 1, yne = ysw + span - 1;
      for (int d = DIR_N; d <= DIR_W; ++d) {
        int64_t previous = -5;
        for (int k = 0; k < span; ++k) {
          bool border; int x, y, coord; int code;
          if (d == DIR_N) { border = yne == ysz - 1 || yne == side - 1; x = xne - k; y = yne + 1; coord = x; code = -1; }
          else if (d == DIR_E) { border = xne == xsz - 1 || xne == side - 1; x = xne + 1; y = yne - k; coord = y; code = -2; }
          else if (d == DIR_S) { border = ysw == 0; x = xsw + k; y = ysw - 1; coord = x; code = -3; }
          else { border = xsw == 0; x = xsw - 1; y = ysw + k; coord = y; code = -4; }
          const Side &s = sides[d - 1];
          const bool bcBound = s.type != mwdg2::BC_CLOSED && s.bound(coord);
          const int64_t check = border ? code : int64_t(leafNode[finestLeaf[size_t(y) * side + x]]);
          const bool already = check == previous;
          previous = check;
          if (already && !(bcBound && border)) continue;
          Record r{};
          r.owner = i;
          r.dir = static_cast<uint32_t>(d);
          r.side = -1;
          if (border) {
            r.nghbr = -1;
            r.length = f.dx;
            r.side = d - 1;
            const int ownerCoord = (d == DIR_N || d == DIR_S) ? int(f.li) : int(f.lj);
            r.bound = (f.level == uint32_t(L) && sides[d - 1].type != mwdg2::BC_CLOSED &&
                       sides[d - 1].bound(ownerCoord)) ? 1u : 0u;
            edgeRecords.push_back(static_cast<uint32_t>(records.size()));
          } else {
            r.nghbr = finestLeaf[size_t(y) * side + x];
            r.length = leaves[r.nghbr].dx;
            const bool eastOrSouth = d == DIR_E || d == DIR_S;
            r.ownerFlag = eastOrSouth ? 1.0f : -1.0f;
            r.nghbrFlag = eastOrSouth ? -1.0f : 1.0f;
          }
          records.push_back(r);
        }
      }
      f.recCount = static_cast<uint32_t>(records.size()) - f.recStart;
    }
    // count_interfaces_per_neighbours / find_interfaces: the perpendicular
    // records of the owner and of the neighbour
    std::vector<uint32_t> loadIdx;
    auto perpendicular = [](uint32_t a, uint32_t b) {
      const bool aEW = a == DIR_E || a == DIR_W, bEW = b == DIR_E || b == DIR_W;
      return aEW != bEW;
    };
    for (auto &r : records) {
      r.itfStart = static_cast<uint32_t>(loadIdx.size());
      if (r.nghbr < 0) continue;
      float dxSum = 0.0f;
      for (uint32_t owner : {r.owner, static_cast<uint32_t>(r.nghbr)}) {
        const Leaf &f = leaves[owner];
        for (uint32_t k = f.recStart; k < f.recStart + f.recCount; ++k)
          if (perpendicular(r.dir, records[k].dir)) {
            loadIdx.push_back(k);
            dxSum += records[k].length;
          }
      }
      r.itfCount = static_cast<uint32_t>(loadIdx.size()) - r.itfStart;
      r.dxSum = dxSum;
    }
    for (auto &s : sources)
      if (s.x >= 0 && s.y >= 0 && s.x < side && s.y < side) {
        const int leaf = finestLeaf[size_t(s.y) * side + s.x];
        if (leaves[leaf].level == uint32_t(L)) s.leaf = leaf;
      }
    // raster cell -> leaf, for losses and output
    std::vector<int32_t> owner(rasterCells, -1);
    for (int row = 0; row < ysz; ++row)
      for (int col = 0; col < xsz; ++col)
        owner[size_t(row) * xsz + col] = finestLeaf[size_t(ysz - 1 - row) * side + col];

    if (verbose) {
      std::printf("Metal ACC_NUGRID: %u leaves (%d x %d finest), %zu neighbour records, %zu interfaces\n",
                  leafCount, side, side, records.size(), loadIdx.size());
      std::fflush(stdout);
    }

    // buffers
    id<MTLBuffer> leafBuf = runtime.buffer(leaves.size() * sizeof(Leaf));
    std::memcpy(leafBuf.contents, leaves.data(), leaves.size() * sizeof(Leaf));
    id<MTLBuffer> recBuf = runtime.buffer(records.size() * sizeof(Record));
    std::memcpy(recBuf.contents, records.data(), records.size() * sizeof(Record));
    id<MTLBuffer> qBuf = runtime.buffer(records.size() * sizeof(float));
    id<MTLBuffer> loadBuf = runtime.buffer(loadIdx.size() * sizeof(uint32_t));
    std::memcpy(loadBuf.contents, loadIdx.data(), loadIdx.size() * sizeof(uint32_t));
    id<MTLBuffer> qvolBuf = runtime.buffer(loadIdx.size() * sizeof(float));
    id<MTLBuffer> vRecBuf = runtime.buffer(records.size() * sizeof(float));
    id<MTLBuffer> vLeafBuf = runtime.buffer(leaves.size() * sizeof(float));
    id<MTLBuffer> cflBuf = runtime.buffer(leaves.size() * sizeof(float));
    id<MTLBuffer> ownerBuf = runtime.buffer(rasterCells * sizeof(int32_t));
    std::memcpy(ownerBuf.contents, owner.data(), rasterCells * sizeof(int32_t));
    id<MTLBuffer> depthBuf = runtime.buffer(rasterCells * sizeof(float));
    id<MTLBuffer> zeroQ = runtime.buffer(rasterCells * sizeof(float));
    id<MTLBuffer> demBuf = runtime.buffer(rasterCells * sizeof(float));
    for (size_t i = 0; i < rasterCells; ++i)
      static_cast<float *>(demBuf.contents)[i] = static_cast<float>(arrays.DEM[i]);
    id<MTLBuffer> removedBuf = runtime.buffer(leaves.size() * sizeof(float));
    const uint32_t srcCount = static_cast<uint32_t>(sources.size());
    id<MTLBuffer> srcLeafBuf = runtime.buffer(std::max<size_t>(1, srcCount) * sizeof(int32_t));
    id<MTLBuffer> srcTypeBuf = runtime.buffer(std::max<size_t>(1, srcCount) * sizeof(int32_t));
    id<MTLBuffer> srcValueBuf = runtime.buffer(std::max<size_t>(1, srcCount) * sizeof(float));
    for (uint32_t k = 0; k < srcCount; ++k) {
      static_cast<int32_t *>(srcLeafBuf.contents)[k] = sources[k].leaf;
      static_cast<int32_t *>(srcTypeBuf.contents)[k] = sources[k].type;
    }
    auto *leafData = static_cast<Leaf *>(leafBuf.contents);
    const float *qData = static_cast<const float *>(qBuf.contents);

    ::DynamicRain<std::allocator<NUMERIC_TYPE>> dynamicRain(filenames.dynamicrainfilename, verbose);
    id<MTLBuffer> rainBuf = nil;
    UniformRain uniformRain;
    if (states.rainfall == ON) uniformRain.load(filenames.rainfilename);

    auto project = [&]() {
      uint32_t cells = static_cast<uint32_t>(rasterCells);
      runtime.dispatch("nug_project", {leafBuf, ownerBuf, depthBuf}, {&cells}, {sizeof(cells)}, cells);
      const float *d = static_cast<const float *>(depthBuf.contents);
      for (size_t i = 0; i < rasterCells; ++i) arrays.H[i] = d[i];
    };
    if (arrays.maxH)
      for (size_t i = 0; i < rasterCells; ++i) {
        arrays.maxH[i] = 0; arrays.maxHtm[i] = 0; arrays.totalHtm[i] = 0; arrays.initHtm[i] = NULLVAL;
      }
    project();
    solver.vol1 = DomainVol(&states, &pars, nullptr, &arrays, nullptr);
    solver.Tstep = solver.InitTstep;
    solver.MinTstep = solver.InitTstep;
    double lossTotal = 0.0, integratedLossTime = 0.0;
    bool firstLossRecord = true;
    const float cellArea = static_cast<float>(pars.dx * pars.dy);
    if (verbose) { std::printf("\nStarting time steps: acceleration mode (Metal ACC_NUGRID)\n\n"); std::fflush(stdout); }

    while (solver.t < solver.Sim_Time) {
      @autoreleasepool {
      zero_flux_stats(&boundCs);
      if (pars.drain_nodata == ON)
        runtime.dispatch("nug_drain", {leafBuf}, {&leafCount, &nodata}, {sizeof(leafCount), sizeof(nodata)}, leafCount);
      const float previousDt = static_cast<float>(solver.Tstep);
      if (dynamicRain.enabled()) {
        dynamicRain.update_time(solver.t);
        const Geometry &g = dynamicRain.geometry();
        const size_t n = size_t(g.xsz) * g.ysz;
        if (!rainBuf) rainBuf = runtime.buffer(n * sizeof(float));
        auto *dst = static_cast<float *>(rainBuf.contents);
        const NUMERIC_TYPE *src = dynamicRain.data();
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
        int32_t geo[2] = {g.xsz, g.ysz};
        float f[4] = {previousDt, nodata, static_cast<float>(pars.tly - g.tly), static_cast<float>(g.dx)};
        float dyRain = static_cast<float>(g.dy);
        runtime.dispatch("nug_rain_dynamic", {leafBuf, rainBuf}, {&leafCount, geo, f, &dyRain},
                         {sizeof(leafCount), sizeof(geo), sizeof(f), sizeof(dyRain)}, leafCount);
      }
      if (states.rainfall == ON) {
        uniformRain.update(static_cast<float>(solver.t));
        float p[2] = {uniformRain.value * previousDt, nodata};
        runtime.dispatch("nug_rain_uniform", {leafBuf}, {&leafCount, p}, {sizeof(leafCount), sizeof(p)}, leafCount);
        double area = 0.0;
        for (uint32_t i = 0; i < leafCount; ++i)
          if (std::fabs(leafData[i].z - nodata) >= 1e-6f) area += double(leafData[i].dx) * leafData[i].dx;
        pars.RainTotalLoss += static_cast<NUMERIC_TYPE>(p[0] * area);
        boundCs.VolInMT += static_cast<NUMERIC_TYPE>(p[0] * area);
      }
      // calculate_dt / get_dt_CFL
      solver.Tstep = solver.InitTstep;
      float cflArgs[4] = {static_cast<float>(solver.InitTstep), static_cast<float>(solver.cfl),
                          static_cast<float>(solver.g), static_cast<float>(solver.DepthThresh)};
      runtime.dispatch("nug_cfl", {leafBuf, cflBuf}, {&leafCount, cflArgs}, {sizeof(leafCount), sizeof(cflArgs)}, leafCount);
      float dt = static_cast<float>(solver.InitTstep);
      const float *cfl = static_cast<const float *>(cflBuf.contents);
      for (uint32_t i = 0; i < leafCount; ++i) dt = std::min(dt, cfl[i]);
      solver.Tstep = dt;
      // point sources and boundary values at the step's start time
      const float now = static_cast<float>(solver.t);
      for (uint32_t k = 0; k < srcCount; ++k) {
        float v = sources[k].value;
        if ((sources[k].type == mwdg2::BC_HVAR || sources[k].type == mwdg2::BC_QVAR) &&
            boundCs.PS_TimeSeries && int(k) < boundCs.numPS && boundCs.PS_TimeSeries[k])
          v = static_cast<float>(InterpolateTimeSeries(boundCs.PS_TimeSeries[k], solver.t));
        static_cast<float *>(srcValueBuf.contents)[k] = v;
      }
      BcArgs bc{};
      for (int k = 0; k < 4; ++k) {
        if (sides[k].type == mwdg2::BC_HVAR || sides[k].type == mwdg2::BC_QVAR)
          sides[k].inlet = sides[k].series.at(now);
        bc.type[k] = sides[k].type;
        bc.value[k] = sides[k].inlet;
      }
      uint32_t recCount = static_cast<uint32_t>(records.size());
      uint32_t itfCount = static_cast<uint32_t>(loadIdx.size());
      runtime.dispatch("nug_load_qvol", {loadBuf, qBuf, qvolBuf}, {&itfCount}, {sizeof(itfCount)}, itfCount);
      float qArgs[4] = {dt, static_cast<float>(solver.g), static_cast<float>(solver.DepthThresh),
                        static_cast<float>(pars.FPn)};
      uint32_t nuN = nonUniformN ? 1u : 0u;
      runtime.dispatch("nug_compute_q", {leafBuf, recBuf, qBuf, qvolBuf, vRecBuf},
                       {&recCount, qArgs, &bc, &nuN},
                       {sizeof(recCount), sizeof(qArgs), sizeof(bc), sizeof(nuN)}, recCount);
      uint32_t hCount[2] = {leafCount, srcCount};
      float hArgs[4] = {dt, static_cast<float>(solver.DepthThresh), 0.0f, 0.0f};
      runtime.dispatch("nug_update_h", {leafBuf, recBuf, qBuf, srcLeafBuf, srcTypeBuf, srcValueBuf, vRecBuf, vLeafBuf},
                       {hCount, hArgs}, {sizeof(hCount), sizeof(hArgs)}, leafCount);
      // edge discharge (after the wet/dry limiting) for the mass balance
      double edgeIn = 0.0;
      for (uint32_t r : edgeRecords) {
        const uint32_t d = records[r].dir;
        edgeIn += ((d == DIR_N || d == DIR_W) ? 1.0 : -1.0) * qData[r];
      }
      if (edgeIn >= 0) { boundCs.Qin += edgeIn; boundCs.VolInMT += edgeIn * dt; }
      else { boundCs.Qout -= edgeIn; boundCs.VolOutMT -= edgeIn * dt; }
      if (lossesEnabled) {
        project();
        @autoreleasepool {
          id<MTLCommandBuffer> command = [runtime.queue commandBuffer];
          id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
          losses.encode(enc, depthBuf, zeroQ, zeroQ, false, 0, 0, 0.0f, demBuf, nodata,
                        static_cast<float>(solver.DepthThresh), dt, solver.t);
          [enc endEncoding]; [command commit]; [command waitUntilCompleted];
        }
        losses.collect();
        losses.take_interval_volume();
        uint32_t layout[4] = {leafCount, uint32_t(xsz), uint32_t(ysz), 0};
        runtime.dispatch("nug_restrict_losses", {leafBuf, depthBuf, demBuf, losses.residual_buffer(), removedBuf},
                         {layout, &nodata}, {sizeof(layout), sizeof(nodata)}, leafCount);
        double removed = 0.0;
        const float *rem = static_cast<const float *>(removedBuf.contents);
        for (uint32_t i = 0; i < leafCount; ++i) removed += rem[i];
        removed *= cellArea;
        lossTotal += removed;
        integratedLossTime += dt;
        boundCs.VolOutMT += static_cast<NUMERIC_TYPE>(removed);
        pars.InfilTotalLoss += static_cast<NUMERIC_TYPE>(removed);
      }

      if (solver.t > 0) solver.MinTstep = std::min<NUMERIC_TYPE>(solver.MinTstep, dt);
      const NUMERIC_TYPE tStart = solver.t;
      solver.t += dt;
      solver.itCount += 1;
      if (verbose) std::printf("t=%f\tdt=%f\n", double(tStart), double(dt));

      if (solver.t >= pars.MassTotal) {
        project();
        // update_max at the mass output times, as cuda acc_nugrid
        if (arrays.maxH)
          for (size_t i = 0; i < rasterCells; ++i)
            if (arrays.H[i] > solver.DepthThresh) {
              if (arrays.initHtm[i] == NULLVAL) arrays.initHtm[i] = solver.t / C(3600.0);
              arrays.totalHtm[i] += solver.Tstep / C(3600.0);
              if (arrays.H[i] > arrays.maxH[i]) { arrays.maxH[i] = arrays.H[i]; arrays.maxHtm[i] = solver.t / C(3600.0); }
            }
        if (lossesEnabled) {
          char path[512];
          std::snprintf(path, sizeof(path), "%s.loss", filenames.resrootname);
          FILE *fp = std::fopen(path, firstLossRecord ? "w" : "a");
          if (fp) {
            if (firstLossRecord) std::fprintf(fp, "time,integrated_time,loss_volume,water_volume\n");
            double water = 0.0;
            for (uint32_t i = 0; i < leafCount; ++i) {
              const Leaf &f = leafData[i];
              int valid = 0;
              for (uint32_t y = f.y0; y < f.y0 + f.span && y < uint32_t(ysz); ++y)
                for (uint32_t x = f.x0; x < f.x0 + f.span && x < uint32_t(xsz); ++x)
                  if (std::fabs(static_cast<float>(arrays.DEM[(ysz - 1 - y) * xsz + x]) - nodata) >= 1e-6f) ++valid;
              water += double(f.h) * valid * cellArea;
            }
            std::fprintf(fp, "%.17g,%.17g,%.17g,%.17g\n", double(solver.t), integratedLossTime, lossTotal, water);
            std::fclose(fp);
            firstLossRecord = false;
          }
        }
        if (states.save_stages == ON) {
          const float *vLeaf = static_cast<const float *>(vLeafBuf.contents);
          // through LISFLOOD's open .stage/.velocity files (lisflood.cpp)
          auto record = [&](FILE *fp, bool velocity) {
            if (!fp) return;
            std::fprintf(fp, "%12.3f", double(solver.t));
            for (int i = 0; i < stage.Nstages; ++i) {
              const int x = static_cast<int>((stage.stage_loc_x[i] - pars.blx) / pars.dx);
              const int y = static_cast<int>((stage.stage_loc_y[i] - pars.bly) / pars.dx);
              double v = 0.0;
              if (x >= 0 && y >= 0 && x < side && y < side) {
                const int leaf = finestLeaf[size_t(y) * side + x];
                v = velocity ? vLeaf[leaf] : leafData[leaf].h;
              }
              std::fprintf(fp, "%10.4f", v);
            }
            std::fputc('\n', fp);
            std::fflush(fp);
          };
          record(files ? files->stage_fp : nullptr, false);
          if (states.voutput_stage == ON) record(files ? files->vel_fp : nullptr, true);
        }
        update_mass_stats(&states, &pars, &solver, &boundCs, &arrays);
        write_mass_stats(files, &pars, &solver, &boundCs);
      }
      if (solver.t >= pars.SaveTotal) {
        project();
        if (states.save_vtk == ON) {  // write_soln_vtk.cu: the leaves as quads
          char path[800];
          std::snprintf(path, sizeof(path), "%s-%.4d.vtk", filenames.resrootname, pars.SaveNo);
          FILE *fp = std::fopen(path, "w");
          if (fp) {
            const int prec = pars.output_precision;
            const float *vLeaf = static_cast<const float *>(vLeafBuf.contents);
            std::fprintf(fp, "# vtk DataFile Version 3.0\nMultiresolution flow and topo data\nASCII\n\n"
                             "DATASET UNSTRUCTURED_GRID\nPOINTS %u float\n", leafCount * 4);
            for (uint32_t i = 0; i < leafCount; ++i) {
              const Leaf &f = leafData[i];
              const double c = f.dx, x = pars.blx + f.li * c, y = pars.bly + f.lj * c;
              std::fprintf(fp, "%.*f %.*f %.*f\n%.*f %.*f %.*f\n%.*f %.*f %.*f\n%.*f %.*f %.*f\n",
                           prec, x, prec, y, prec, 1.0, prec, x, prec, y + c, prec, 1.0,
                           prec, x + c, prec, y, prec, 1.0, prec, x + c, prec, y + c, prec, 1.0);
            }
            std::fprintf(fp, "\nCELLS %u %u\n", leafCount, leafCount * 5);
            for (uint32_t i = 0; i < leafCount; ++i)
              std::fprintf(fp, "4 %u %u %u %u\n", 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
            std::fprintf(fp, "\nCELL_TYPES %u\n", leafCount);
            for (uint32_t i = 0; i < leafCount; ++i) std::fprintf(fp, "8\n");
            std::fprintf(fp, "\nCELL_DATA %u\n\nSCALARS z0 float 1\nLOOKUP_TABLE default\n", leafCount);
            for (uint32_t i = 0; i < leafCount; ++i) std::fprintf(fp, "%.*f\n", prec, double(leafData[i].z));
            std::fprintf(fp, "\nSCALARS h float 1\nLOOKUP_TABLE default\n");
            for (uint32_t i = 0; i < leafCount; ++i) std::fprintf(fp, "%.*f\n", prec, double(leafData[i].h));
            std::fprintf(fp, "\nSCALARS v float 1\nLOOKUP_TABLE default\n");
            for (uint32_t i = 0; i < leafCount; ++i) std::fprintf(fp, "%.*f\n", prec, double(vLeaf[i]));
            std::fclose(fp);
          }
        }
        write_regular_output(&filenames, &solver, &states, &pars, &arrays, nullptr);
        pars.SaveTotal += pars.SaveInt;
        pars.SaveNo += 1;
      }
      }
    }
    project();
    write_max_field(&filenames, &states, &pars, &arrays);
    if (verbose) std::printf("\nFinished Metal ACC_NUGRID.\n\n");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ERROR: Metal ACC_NUGRID: %s\n", e.what());
    std::exit(1);
  }
  }
}

} } } // namespace lis::metal::acc_nugrid
#endif
