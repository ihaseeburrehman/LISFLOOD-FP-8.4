//
// LISFLOOD-FP Metal ACC Solver - Implementation
//
// Host side of the local-inertial ("acceleration") solver. Mirrors the stage
// ordering of lis::cuda::acc::Simulation::run in cuda/acc/cuda_acc_simulate.cu
// so the two backends can be compared step for step.
//

#ifdef __APPLE__

#include <string>
#include "metal_acc_simulate.h"
#include "../losses/metal_losses.h"
#include "../../rain/rain.h"
#include "../../rain/rain.tpp"
#include "../../swe/output.h"
#include "../../swe/stats.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#if defined(LISFLOOD_METAL_MPI)
#include <mpi.h>
#endif
#include <cmath>

namespace lis {
namespace metal {
namespace acc {

// Must match the structs in metal_acc_kernels.metal
struct MetalAccGeometry {
  int32_t xsz, ysz;
  float dx, dy, tly, blx, bly;
};
struct MetalAccPhysical {
  float g, manning;
};
struct MetalAccSolver {
  float dt, depth_thresh, max_hflow, theta, nodata_elev, max_dt, cfl;
};

struct SimulatorImpl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;

  id<MTLComputePipelineState> pDryWeight, pDryScale;
  id<MTLComputePipelineState> pDryIter, pFill, pDtCuda, pPointSrc;
  id<MTLComputePipelineState> pUpdateDt, pQx, pQy, pZeroEdge, pBC, pUpdateH,
      pStoreQold, pUniformRain, pDynRain, pDrainNodata, pUpdateMax,
      pBoundaryFlux;

  // H/DEM/manning are xsz*ysz; Q rasters are (xsz+1)*(ysz+1)
  id<MTLBuffer> H, DEM, manning;
  id<MTLBuffer> Qx, Qy, Qxold, Qyold, Vx, Vy;
  id<MTLBuffer> maxH, maxHtm, totalHtm, initHtm;
  id<MTLBuffer> dryW, dryW2;   // dry-check weights (ping-pong)
  id<MTLBuffer> changedBuf;    // one flag per dry-check iteration
  id<MTLBuffer> dtField;       // CUDA-compatible mode: per-cell running-min dt
  id<MTLBuffer> urainPart;     // uniform-rain partial sums
  id<MTLBuffer> psCell, psType, psValue, psQ;   // point sources (.bci P lines)
  int n_ps = 0;
  // 0: CPU-faithful (default) -- the stage order and arithmetic of
  //    iterateq.cpp, bit-identical to the CPU where the arithmetic allows.
  // 1: CUDA-compatible -- the order and the quirks of cuda/acc, for verifying
  //    the port against the CUDA solver only. LISFLOOD_ACC_REFERENCE=cuda.
  int acc_mode = 0;
  std::string manning_file;   // CUDA mode re-reads it raw (see uploadFields)
  double prev_dt = 0.0;       // CUDA mode: rain uses the previous step's dt
  int dry_iters_last = 0;
  id<MTLBuffer> geomBuf, physBuf, solvBuf, minDtBuf;
  // One float per threadgroup per counter; see store_tg_sum in the kernels.
  id<MTLBuffer> rainPart, drainPart, clipPart, bndInPart, bndOutPart;
  size_t n_tg_cells = 0, n_tg_edge = 0;
  static constexpr int kEdgeTG = 64;
  id<MTLBuffer> bcTypeBuf, bcValueBuf, rainBuf, rainGeomBuf;

  MetalAccGeometry geom{};
  MetalAccPhysical phys{};
  MetalAccSolver solv{};

  size_t n_cells = 0, n_q = 0;
  bool has_manning = false;
  bool use_friction = false;
  Losses losses;                // infiltration and evaporation (8.4)
  Fnames *fnames = nullptr;
  int verbose = 0;
  NUMERIC_TYPE time = 0;
  double last_rain_upload = -1.0;
  // Mass-balance accumulators, in double, drained into BoundCs by syncMass().
  double acc_rain = 0.0, acc_drain = 0.0, acc_clip = 0.0;
  double acc_vol_in = 0.0, acc_vol_out = 0.0;
  double last_qin = 0.0, last_qout = 0.0;   // edge only, last step
  double last_mass_sync_time = 0.0;

  Arrays *arrays = nullptr;
  BoundCs *boundCs = nullptr;
  Pars *pars = nullptr;
  ::Solver *solver = nullptr;
  States *states = nullptr;
  ::DynamicRain<std::allocator<NUMERIC_TYPE>> *dynamic_rain = nullptr;

  ~SimulatorImpl() { delete dynamic_rain; }

  id<MTLComputePipelineState> pipeline(const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn =
        [library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
      if (verbose) printf("Metal ACC: missing kernel '%s'\n", name);
      return nil;
    }
    id<MTLComputePipelineState> p =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!p) {
      printf("Metal ACC: pipeline '%s' failed: %s\n", name,
             [[err localizedDescription] UTF8String]);
      return nil;
    }
    // Every 2-D kernel is dispatched 16x16 = 256 threads per threadgroup. A
    // pipeline that cannot hold 256 (register or threadgroup-memory pressure)
    // must not be dispatched that way; without the validation layer Metal
    // does not report it.
    if (getenv("LISFLOOD_METAL_DEBUG"))
      printf("Metal ACC: %-24s maxThreads=%lu  execWidth=%lu  tgMem=%lu\n",
             name, (unsigned long)p.maxTotalThreadsPerThreadgroup,
             (unsigned long)p.threadExecutionWidth,
             (unsigned long)p.staticThreadgroupMemoryLength);
    if (p.maxTotalThreadsPerThreadgroup < 256) {
      printf("Metal ACC: pipeline '%s' allows only %lu threads per "
             "threadgroup (< 256)\n",
             name, (unsigned long)p.maxTotalThreadsPerThreadgroup);
      return nil;
    }
    return p;
  }

  bool init(Fnames &filenames, Pars &p, ::Solver &s, int verb) {
    verbose = verb;
    fnames = &filenames;
    manning_file = filenames.nfilename;
    dynamic_rain = new ::DynamicRain<std::allocator<NUMERIC_TYPE>>(
        filenames.dynamicrainfilename, verbose);

    device = MTLCreateSystemDefaultDevice();
    if (!device) return false;
    queue = [device newCommandQueue];
    if (!queue) return false;

    if (verbose) {
      printf("Metal ACC: GPU %s (unified memory: %s)\n",
             [[device name] UTF8String],
             device.hasUnifiedMemory ? "yes" : "no");
    }

    // Locate the shader library next to the executable or in the CWD.
    NSError *err = nil;
    NSString *path = nil;
    NSString *exeDir = [[[NSProcessInfo processInfo] arguments][0]
        stringByDeletingLastPathComponent];
    for (NSString *cand in @[
           [exeDir stringByAppendingPathComponent:@"metal_acc_kernels.metallib"],
           @"metal_acc_kernels.metallib"
         ]) {
      if ([[NSFileManager defaultManager] fileExistsAtPath:cand]) {
        path = cand;
        break;
      }
    }
    if (!path) {
      if (verbose) printf("Metal ACC: metal_acc_kernels.metallib not found\n");
      return false;
    }
    library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!library) return false;

    pUpdateDt = pipeline("acc_update_dt");
    pQx = pipeline("acc_floodplain_qx");
    pQy = pipeline("acc_floodplain_qy");
    pZeroEdge = pipeline("acc_zero_edge_flux");
    pBC = pipeline("acc_boundary_conditions");
    pDryWeight = pipeline("acc_drycheck_weight");
    pDryScale = pipeline("acc_drycheck_scale");
    pUpdateH = pipeline("acc_update_h");
    pStoreQold = pipeline("acc_store_qold");
    pUniformRain = pipeline("acc_uniform_rain");
    pDynRain = pipeline("acc_dynamic_rain");
    pDrainNodata = pipeline("acc_drain_nodata");
    pUpdateMax = pipeline("acc_update_max");
    pBoundaryFlux = pipeline("acc_boundary_flux");
    pDryIter = pipeline("acc_drycheck_cpu_iter");
    pFill = pipeline("acc_fill");
    pDtCuda = pipeline("acc_update_dt_cuda");
    pPointSrc = pipeline("acc_point_sources");
    const char *ref = getenv("LISFLOOD_ACC_REFERENCE");
    acc_mode = (ref && !strcmp(ref, "cuda")) ? 1 : 0;
    if (acc_mode == 1)
      printf("Metal ACC: CUDA-compatible verification mode "
             "(LISFLOOD_ACC_REFERENCE=cuda). Not for production runs.\n");
    if (!(pUpdateDt && pQx && pQy && pZeroEdge && pBC && pUpdateH && pStoreQold))
      return false;

    geom.xsz = p.xsz;
    geom.ysz = p.ysz;
    geom.dx = (float)p.dx;
    geom.dy = (float)p.dy;
    geom.tly = (float)p.tly;
    geom.blx = (float)p.blx;
    geom.bly = (float)p.bly;

    phys.g = (float)s.g;
    phys.manning = (float)p.FPn;

    solv.dt = (float)s.InitTstep;
    solv.depth_thresh = (float)s.DepthThresh;
    solv.max_hflow = (float)s.MaxHflow;
    solv.theta = (float)s.theta;
    solv.nodata_elev = (float)p.nodata_elevation;
    solv.max_dt = (float)s.InitTstep;
    solv.cfl = (float)s.cfl;

    n_cells = (size_t)geom.xsz * geom.ysz;
    n_q = (size_t)(geom.xsz + 1) * (geom.ysz + 1);

    MTLResourceOptions opt = MTLResourceStorageModeShared;
    auto mk = [&](size_t n) {
      id<MTLBuffer> b = [device newBufferWithLength:n * sizeof(float) options:opt];
      if (b) memset([b contents], 0, n * sizeof(float));
      return b;
    };
    H = mk(n_cells); DEM = mk(n_cells); manning = mk(n_cells);
    Qx = mk(n_q); Qy = mk(n_q); Qxold = mk(n_q); Qyold = mk(n_q);
    Vx = mk(n_q); Vy = mk(n_q);
    dryW = mk(n_cells); dryW2 = mk(n_cells); dtField = mk(n_cells);
    changedBuf = [device newBufferWithLength:8 * sizeof(uint32_t) options:opt];
    maxH = mk(n_cells); maxHtm = mk(n_cells);
    totalHtm = mk(n_cells); initHtm = mk(n_cells);
    if (!H || !DEM || !Qx || !Qy) return false;

    // initHtm starts at NULLVAL so the first wetting time is recorded
    float *ih = (float *)[initHtm contents];
    for (size_t k = 0; k < n_cells; k++) ih[k] = -9999.0f;

    geomBuf = [device newBufferWithBytes:&geom length:sizeof(geom) options:opt];
    physBuf = [device newBufferWithBytes:&phys length:sizeof(phys) options:opt];
    // Upload the solver constants NOW. step() refreshes this buffer, but only
    // after computeTimestep() has run -- so on the very first step the CFL
    // kernel used to read an all-zero block (cfl = 0 => dt_local = 0), the host
    // rejected that as "no valid dt" and fell back to max_dt = initial_tstep.
    // At 100 m that happens to be CFL-stable; at 10 m it is a 10 s step where
    // CFL allows 0.66 s, and the first step then created ~1.4e6 m^3 of water.
    solvBuf = [device newBufferWithBytes:&solv length:sizeof(solv) options:opt];
    minDtBuf = [device newBufferWithLength:sizeof(uint32_t) options:opt];
    n_tg_cells = (size_t)((geom.xsz + 15) / 16) * (size_t)((geom.ysz + 15) / 16);
    n_tg_edge = (size_t)((2 * geom.xsz + 2 * geom.ysz + kEdgeTG - 1) / kEdgeTG);
    rainPart = mk(n_tg_cells); drainPart = mk(n_tg_cells); clipPart = mk(n_tg_cells);
    urainPart = mk(n_tg_cells);
    bndInPart = mk(n_tg_edge); bndOutPart = mk(n_tg_edge);

    int n_bc = 2 * geom.xsz + 2 * geom.ysz;
    bcTypeBuf = [device newBufferWithLength:n_bc * sizeof(int32_t) options:opt];
    bcValueBuf = [device newBufferWithLength:n_bc * sizeof(float) options:opt];
    memset([bcTypeBuf contents], 0, n_bc * sizeof(int32_t));
    memset([bcValueBuf contents], 0, n_bc * sizeof(float));

    size_t nr = 1;
    if (dynamic_rain && dynamic_rain->enabled()) {
      const Geometry &rg = dynamic_rain->geometry();
      nr = (size_t)rg.xsz * rg.ysz;
      MetalAccGeometry mrg{};
      mrg.xsz = rg.xsz; mrg.ysz = rg.ysz;
      mrg.dx = (float)rg.dx; mrg.dy = (float)rg.dy;
      mrg.tly = (float)rg.tly; mrg.blx = (float)rg.blx; mrg.bly = (float)rg.bly;
      rainGeomBuf = [device newBufferWithBytes:&mrg length:sizeof(mrg) options:opt];
    } else {
      MetalAccGeometry z{};
      rainGeomBuf = [device newBufferWithBytes:&z length:sizeof(z) options:opt];
    }
    rainBuf = mk(nr);

    // point sources: the P lines of the .bci file
    n_ps = (boundCs && boundCs->numPS > 0) ? boundCs->numPS : 0;
    size_t nps = std::max(n_ps, 1);
    psCell = [device newBufferWithLength:nps * sizeof(int32_t) options:opt];
    psType = [device newBufferWithLength:nps * sizeof(int32_t) options:opt];
    psValue = mk(nps); psQ = mk(nps);
    for (int k = 0; k < n_ps; k++) {
      ((int32_t *)[psCell contents])[k] = boundCs->xpi[k] + boundCs->ypi[k] * geom.xsz;
      ((int32_t *)[psType contents])[k] = (int32_t)boundCs->PS_Ident[k];
    }
    return true;
  }

  // Point-source values at the start of the step, evaluated as UpdateH() does.
  void updatePointSourceValues() {
    float *v = (float *)[psValue contents];
    for (int k = 0; k < n_ps; k++) {
      ESourceType t = boundCs->PS_Ident[k];
      if (t == QVAR5 || t == HVAR3)
        v[k] = (float)InterpolateTimeSeries(boundCs->PS_TimeSeries[k], time);
      else
        v[k] = (float)boundCs->PS_Val[k];
      ((float *)[psQ contents])[k] = 0.0f;
    }
  }

  void dispatch2D(id<MTLComputeCommandEncoder> enc, int nx, int ny) {
    [enc dispatchThreads:MTLSizeMake(std::max(nx, 1), std::max(ny, 1), 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  }

  void uploadFields(NUMERIC_TYPE *h_H, NUMERIC_TYPE *h_DEM, NUMERIC_TYPE *h_M) {
    float *dH = (float *)[H contents];
    float *dZ = (float *)[DEM contents];
    float *dM = (float *)[manning contents];
    has_manning = (h_M != nullptr);
    for (size_t k = 0; k < n_cells; k++) {
      dZ[k] = (float)h_DEM[k];
      dH[k] = h_H ? (float)h_H[k] : 0.0f;
      dM[k] = h_M ? (float)h_M[k] : phys.manning;
    }
    use_friction = has_manning || phys.manning > 0.0f;
    if (fnames)
      losses.setup(device, *fnames, *states, *pars, arrays, geom.xsz, geom.ysz,
                   geom.xsz, 0, 0, n_cells, verbose);

    // CUDA-compatible mode: cuda/cuda_simulate.cu initialise_manning() reads the
    // Manning raster as it is, NoData values included (-9999 becomes n), where
    // the CPU replaces them with fpfric (input.cpp). On faces next to NoData that
    // changes the friction term enormously, so reproduce it here.
    if (acc_mode == 1 && has_manning && !manning_file.empty()) {
      FILE *f = fopen(manning_file.c_str(), "r");
      char key[64]; double v;
      bool ok = f != nullptr;
      for (int k = 0; ok && k < 6; k++) ok = fscanf(f, "%63s %lf", key, &v) == 2;
      for (size_t k = 0; ok && k < n_cells; k++) {
        float x; ok = fscanf(f, "%f", &x) == 1; dM[k] = x;
      }
      if (f) fclose(f);
      if (!ok) {
        printf("Metal ACC: could not re-read %s raw; CUDA mode keeps the CPU "
               "Manning values\n", manning_file.c_str());
        for (size_t k = 0; k < n_cells; k++) dM[k] = (float)h_M[k];
      }
    }

    if (boundCs && boundCs->BC_Ident) {
      int32_t *t = (int32_t *)[bcTypeBuf contents];
      for (int k = 0; k < 2 * geom.xsz + 2 * geom.ysz; k++)
        t[k] = (int32_t)boundCs->BC_Ident[k];
    }
  }

  // Refresh prescribed boundary values for this time (QVAR/HVAR interpolate).
  void updateBoundaryValues() {
    if (!boundCs || !boundCs->BC_Ident) return;
    float *v = (float *)[bcValueBuf contents];
    int n = 2 * geom.xsz + 2 * geom.ysz;
    for (int k = 0; k < n; k++) {
      switch (boundCs->BC_Ident[k]) {
      case HFIX2: v[k] = (float)boundCs->BC_Val[k]; break;
      case HVAR3:
        v[k] = (float)InterpolateTimeSeries(boundCs->BC_TimeSeries[k], time);
        break;
      case QFIX4: v[k] = (float)boundCs->BC_Val[k]; break;
      case QVAR5:
        v[k] = (float)InterpolateTimeSeries(boundCs->BC_TimeSeries[k], time);
        break;
      case FREE1:
      case FREE6: v[k] = (float)boundCs->BC_Val[k]; break;
      default: v[k] = 0.0f; break;
      }
    }
  }

  NUMERIC_TYPE computeTimestep() {
    if (!pUpdateDt) return solv.max_dt;
    memcpy([solvBuf contents], &solv, sizeof(solv));   // cfl, max_dt, thresholds
    float init = solv.max_dt;
    memcpy([minDtBuf contents], &init, sizeof(float));
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pUpdateDt];
      [enc setBuffer:H offset:0 atIndex:0];
      [enc setBuffer:geomBuf offset:0 atIndex:1];
      [enc setBuffer:physBuf offset:0 atIndex:2];
      [enc setBuffer:solvBuf offset:0 atIndex:3];
      [enc setBuffer:minDtBuf offset:0 atIndex:4];
      dispatch2D(enc, geom.xsz, geom.ysz);
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
    float dt;
    memcpy(&dt, [minDtBuf contents], sizeof(float));
    if (!(dt > 0.0f) || !std::isfinite(dt)) dt = solv.max_dt;
    return (NUMERIC_TYPE)std::min((double)dt, (double)solv.max_dt);
  }

  // ---- encoders for the individual stages -----------------------------------
  void encDynRain(id<MTLComputeCommandEncoder> enc) {
    [enc setComputePipelineState:pDynRain];
    [enc setBuffer:H offset:0 atIndex:0];
    [enc setBuffer:DEM offset:0 atIndex:1];
    [enc setBuffer:rainBuf offset:0 atIndex:2];
    [enc setBuffer:geomBuf offset:0 atIndex:3];
    [enc setBuffer:rainGeomBuf offset:0 atIndex:4];
    [enc setBuffer:solvBuf offset:0 atIndex:5];
    [enc setBuffer:rainPart offset:0 atIndex:6];
    dispatch2D(enc, geom.xsz, geom.ysz);
  }
  void encUniformRain(id<MTLComputeCommandEncoder> enc) {
    float rate = (float)InterpolateTimeSeries(arrays->rain, time);
    [enc setComputePipelineState:pUniformRain];
    [enc setBuffer:H offset:0 atIndex:0];
    [enc setBuffer:DEM offset:0 atIndex:1];
    [enc setBuffer:geomBuf offset:0 atIndex:2];
    [enc setBuffer:solvBuf offset:0 atIndex:3];
    [enc setBytes:&rate length:sizeof(float) atIndex:4];
    [enc setBuffer:urainPart offset:0 atIndex:5];
    dispatch2D(enc, geom.xsz, geom.ysz);
  }
  void encFluxesAndBoundaries(id<MTLComputeCommandEncoder> enc) {
    int fric = use_friction ? 1 : 0;
    int vout = (states && states->voutput == ON) ? 1 : 0;
    id<MTLComputePipelineState> ps[2] = {pQx, pQy};
    id<MTLBuffer> q[2] = {Qx, Qy}, v[2] = {Vx, Vy};
    for (int d = 0; d < 2; d++) {
      [enc setComputePipelineState:ps[d]];
      [enc setBuffer:H offset:0 atIndex:0];
      [enc setBuffer:DEM offset:0 atIndex:1];
      [enc setBuffer:manning offset:0 atIndex:2];
      [enc setBuffer:Qxold offset:0 atIndex:3];
      [enc setBuffer:Qyold offset:0 atIndex:4];
      [enc setBuffer:q[d] offset:0 atIndex:5];
      [enc setBuffer:v[d] offset:0 atIndex:6];
      [enc setBuffer:geomBuf offset:0 atIndex:7];
      [enc setBuffer:physBuf offset:0 atIndex:8];
      [enc setBuffer:solvBuf offset:0 atIndex:9];
      [enc setBytes:&fric length:sizeof(int) atIndex:10];
      [enc setBytes:&vout length:sizeof(int) atIndex:11];
      dispatch2D(enc, geom.xsz, geom.ysz);
    }
    int edge = std::max(geom.xsz, geom.ysz);
    [enc setComputePipelineState:pZeroEdge];
    [enc setBuffer:Qx offset:0 atIndex:0];
    [enc setBuffer:Qy offset:0 atIndex:1];
    [enc setBuffer:geomBuf offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(edge, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc setComputePipelineState:pBC];
    [enc setBuffer:H offset:0 atIndex:0];
    [enc setBuffer:DEM offset:0 atIndex:1];
    [enc setBuffer:manning offset:0 atIndex:2];
    [enc setBuffer:Qx offset:0 atIndex:3];
    [enc setBuffer:Qy offset:0 atIndex:4];
    [enc setBuffer:bcTypeBuf offset:0 atIndex:5];
    [enc setBuffer:bcValueBuf offset:0 atIndex:6];
    [enc setBuffer:geomBuf offset:0 atIndex:7];
    [enc setBuffer:physBuf offset:0 atIndex:8];
    [enc setBuffer:solvBuf offset:0 atIndex:9];
    [enc setBuffer:Qxold offset:0 atIndex:10];
    [enc setBuffer:Qyold offset:0 atIndex:11];
    [enc dispatchThreads:MTLSizeMake(2 * geom.xsz + 2 * geom.ysz, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  }
  void encDrain(id<MTLComputeCommandEncoder> enc) {
    [enc setComputePipelineState:pDrainNodata];
    [enc setBuffer:H offset:0 atIndex:0];
    [enc setBuffer:DEM offset:0 atIndex:1];
    [enc setBuffer:geomBuf offset:0 atIndex:2];
    [enc setBuffer:solvBuf offset:0 atIndex:3];
    [enc setBuffer:drainPart offset:0 atIndex:4];
    dispatch2D(enc, geom.xsz, geom.ysz);
  }
  void encDryIter(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> win,
                  id<MTLBuffer> wout, int flag) {
    [enc setComputePipelineState:pDryIter];
    [enc setBuffer:H offset:0 atIndex:0];
    [enc setBuffer:Qx offset:0 atIndex:1];
    [enc setBuffer:Qy offset:0 atIndex:2];
    [enc setBuffer:win offset:0 atIndex:3];
    [enc setBuffer:wout offset:0 atIndex:4];
    [enc setBuffer:geomBuf offset:0 atIndex:5];
    [enc setBuffer:solvBuf offset:0 atIndex:6];
    [enc setBuffer:changedBuf offset:flag * sizeof(uint32_t) atIndex:7];
    dispatch2D(enc, geom.xsz, geom.ysz);
  }
  void encPointSources(id<MTLComputeCommandEncoder> enc, int phase) {
    if (n_ps == 0 || !pPointSrc) return;
    [enc setComputePipelineState:pPointSrc];
    [enc setBuffer:H offset:0 atIndex:0];
    [enc setBuffer:DEM offset:0 atIndex:1];
    [enc setBuffer:psCell offset:0 atIndex:2];
    [enc setBuffer:psType offset:0 atIndex:3];
    [enc setBuffer:psValue offset:0 atIndex:4];
    [enc setBuffer:psQ offset:0 atIndex:5];
    [enc setBuffer:geomBuf offset:0 atIndex:6];
    [enc setBuffer:solvBuf offset:0 atIndex:7];
    [enc setBytes:&n_ps length:sizeof(int) atIndex:8];
    [enc setBytes:&phase length:sizeof(int) atIndex:9];
    [enc dispatchThreads:MTLSizeMake(n_ps, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  }
  void encBoundaryFluxStats(id<MTLComputeCommandEncoder> enc) {
    if (!pBoundaryFlux) return;
    [enc setComputePipelineState:pBoundaryFlux];
    [enc setBuffer:Qx offset:0 atIndex:0];
    [enc setBuffer:Qy offset:0 atIndex:1];
    [enc setBuffer:geomBuf offset:0 atIndex:2];
    [enc setBuffer:bndInPart offset:0 atIndex:3];
    [enc setBuffer:bndOutPart offset:0 atIndex:4];
    [enc dispatchThreads:MTLSizeMake(2 * geom.xsz + 2 * geom.ysz, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kEdgeTG, 1, 1)];
  }
  void encUpdateH(id<MTLComputeCommandEncoder> enc, int cuda_drycheck) {
    [enc setComputePipelineState:pUpdateH];
    [enc setBuffer:H offset:0 atIndex:0];
    [enc setBuffer:Qx offset:0 atIndex:1];
    [enc setBuffer:Qy offset:0 atIndex:2];
    [enc setBuffer:geomBuf offset:0 atIndex:3];
    [enc setBuffer:solvBuf offset:0 atIndex:4];
    [enc setBuffer:clipPart offset:0 atIndex:5];
    [enc setBytes:&cuda_drycheck length:sizeof(int) atIndex:6];
    dispatch2D(enc, geom.xsz, geom.ysz);
  }
  void encStoreQoldAndMax(id<MTLComputeCommandEncoder> enc) {
    [enc setComputePipelineState:pStoreQold];
    [enc setBuffer:Qx offset:0 atIndex:0];
    [enc setBuffer:Qy offset:0 atIndex:1];
    [enc setBuffer:Qxold offset:0 atIndex:2];
    [enc setBuffer:Qyold offset:0 atIndex:3];
    [enc setBuffer:geomBuf offset:0 atIndex:4];
    dispatch2D(enc, geom.xsz + 1, geom.ysz + 1);
    if (pUpdateMax) {
      float t_now = (float)time;
      [enc setComputePipelineState:pUpdateMax];
      [enc setBuffer:H offset:0 atIndex:0];
      [enc setBuffer:maxH offset:0 atIndex:1];
      [enc setBuffer:maxHtm offset:0 atIndex:2];
      [enc setBuffer:totalHtm offset:0 atIndex:3];
      [enc setBuffer:initHtm offset:0 atIndex:4];
      [enc setBuffer:geomBuf offset:0 atIndex:5];
      [enc setBuffer:solvBuf offset:0 atIndex:6];
      [enc setBytes:&t_now length:sizeof(float) atIndex:7];
      dispatch2D(enc, geom.xsz, geom.ysz);
    }
  }

  // Upload the rain grid when (and only when) the NetCDF reader moves to a new
  // frame -- update_time_SGC() says so -- rather than "at most once a second",
  // which would apply the old frame for up to a second past each boundary.
  bool refreshRain() {
    if (!(dynamic_rain && dynamic_rain->enabled() && pDynRain)) return false;
    bool changed = dynamic_rain->update_time_SGC(time);
    if (changed || last_rain_upload < 0) {
      const Geometry &rg = dynamic_rain->geometry();
      size_t nr = (size_t)rg.xsz * rg.ysz;
      NUMERIC_TYPE *src = dynamic_rain->data();
      float *dst = (float *)[rainBuf contents];
      for (size_t k = 0; k < nr; k++) dst[k] = (float)src[k];
      last_rain_upload = (double)time;
    }
    return true;
  }

  void commitAndWait(void (^body)(id<MTLComputeCommandEncoder>)) {
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      body(enc);
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
  }

  NUMERIC_TYPE step() { return acc_mode == 1 ? stepCuda() : stepCpu(); }

  // ---- default: the order of iterateq.cpp -------------------------------------
  //  CalcT -> FloodplainQ -> BCs -> drain_nodata -> DryCheck -> Rainfall
  //  -> dynamic rain -> UpdateH (Q point sources, continuity, H point sources)
  //  -> BoundaryFlux -> UpdateQs
  NUMERIC_TYPE stepCpu() {
    NUMERIC_TYPE dt = computeTimestep();
    solv.dt = (float)dt;
    solver->Tstep = dt;
    memcpy([solvBuf contents], &solv, sizeof(solv));
    updateBoundaryValues();
    updatePointSourceValues();
    const bool dyn = refreshRain();
    const bool uni = states && states->rainfall == ON && pUniformRain && arrays->rain;
    const bool drain = pars->drain_nodata == ON && pDrainNodata;
    const bool dry = states && states->drychecking == ON && pDryIter && pDryScale && pFill;

    // Pass 1: fluxes, boundaries, drain, and the first three dry-check sweeps.
    uint32_t *flags = (uint32_t *)[changedBuf contents];
    memset(flags, 0, 8 * sizeof(uint32_t));
    __block id<MTLBuffer> latest = dryW;
    commitAndWait(^(id<MTLComputeCommandEncoder> enc) {
      encFluxesAndBoundaries(enc);
      if (drain) encDrain(enc);
      if (dry) {
        float one = 1.0f; uint32_t n = (uint32_t)n_cells;
        [enc setComputePipelineState:pFill];
        [enc setBuffer:dryW offset:0 atIndex:0];
        [enc setBytes:&one length:sizeof(float) atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n_cells, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        encDryIter(enc, dryW, dryW2, 0);
        encDryIter(enc, dryW2, dryW, 1);
        encDryIter(enc, dryW, dryW2, 2);
        latest = dryW2;
      }
    });
    // Sweep until a sweep changes nothing; then `latest` is the exact
    // raster-order result (see acc_drycheck_cpu_iter).
    int iters = dry ? 3 : 0;
    while (dry && flags[2] != 0) {
      memset(flags, 0, 8 * sizeof(uint32_t));
      id<MTLBuffer> a = latest, b = (latest == dryW) ? dryW2 : dryW;
      commitAndWait(^(id<MTLComputeCommandEncoder> enc) {
        encDryIter(enc, a, b, 1);
        encDryIter(enc, b, a, 2);
      });
      latest = a; iters += 2;
      if (iters > 4096) { printf("Metal ACC: dry check did not converge\n"); break; }
    }
    dry_iters_last = iters;

    // Pass 2: apply the limiter, rain, point sources, continuity.
    id<MTLBuffer> W = latest;
    commitAndWait(^(id<MTLComputeCommandEncoder> enc) {
      if (dry) {
        [enc setComputePipelineState:pDryScale];
        [enc setBuffer:Qx offset:0 atIndex:0];
        [enc setBuffer:Qy offset:0 atIndex:1];
        [enc setBuffer:W offset:0 atIndex:2];
        [enc setBuffer:geomBuf offset:0 atIndex:3];
        dispatch2D(enc, geom.xsz + 1, geom.ysz + 1);
      }
      // infiltration and evaporation (8.4), where iterateq.cpp applies them
      losses.encode(enc, H, nil, nil, false, 0, 0, 0.0f, DEM, solv.nodata_elev,
                    solv.depth_thresh, dt, time);
      if (uni) encUniformRain(enc);
      if (dyn) encDynRain(enc);
      encPointSources(enc, 0);
      encBoundaryFluxStats(enc);
      encUpdateH(enc, 0);
      encPointSources(enc, 1);
      encStoreQoldAndMax(enc);
    });
    collectStats(dt, dyn, uni, drain);
    losses.collect();
    time += dt;
    return dt;
  }

  // ---- verification only: the order and quirks of cuda/acc ---------------------
  //  dynamic rain (previous dt) -> running-min dt -> fluxes -> boundaries
  //  -> drain -> uniform rain -> update_H with its one-sided limiter -> Qold
  NUMERIC_TYPE stepCuda() {
    updateBoundaryValues();
    updatePointSourceValues();
    const bool dyn = refreshRain();
    const bool uni = states && states->rainfall == ON && pUniformRain && arrays->rain;
    const bool drain = pars->drain_nodata == ON && pDrainNodata;
    const int drycheck = (states && states->drychecking == ON) ? 1 : 2;

    solv.dt = (float)prev_dt;                      // rain first, with the old dt
    memcpy([solvBuf contents], &solv, sizeof(solv));
    float init = solv.max_dt;
    memcpy([minDtBuf contents], &init, sizeof(float));
    commitAndWait(^(id<MTLComputeCommandEncoder> enc) {
      if (dyn) encDynRain(enc);
      [enc setComputePipelineState:pDtCuda];
      [enc setBuffer:H offset:0 atIndex:0];
      [enc setBuffer:dtField offset:0 atIndex:1];
      [enc setBuffer:geomBuf offset:0 atIndex:2];
      [enc setBuffer:physBuf offset:0 atIndex:3];
      [enc setBuffer:solvBuf offset:0 atIndex:4];
      [enc setBuffer:minDtBuf offset:0 atIndex:5];
      dispatch2D(enc, geom.xsz, geom.ysz);
    });
    const double rain_dyn = dyn ? sumPartials(rainPart, n_tg_cells) : 0.0;
    float dtf; memcpy(&dtf, [minDtBuf contents], sizeof(float));
    NUMERIC_TYPE dt = dtf;
    solv.dt = dtf;
    solver->Tstep = dt;
    memcpy([solvBuf contents], &solv, sizeof(solv));

    commitAndWait(^(id<MTLComputeCommandEncoder> enc) {
      encFluxesAndBoundaries(enc);
      if (drain) encDrain(enc);
      if (uni) encUniformRain(enc);
      encPointSources(enc, 0);
      encBoundaryFluxStats(enc);
      encUpdateH(enc, drycheck);
      encPointSources(enc, 1);
      encStoreQoldAndMax(enc);
      // infiltration and evaporation (8.4), after the step as in cuda/acc
      losses.encode(enc, H, nil, nil, false, 0, 0, 0.0f, DEM, solv.nodata_elev,
                    solv.depth_thresh, dt, time + dt);
    });
    losses.collect();
    acc_rain += rain_dyn;
    collectStats(dt, false, uni, drain);
    prev_dt = dt;
    time += dt;
    return dt;
  }

  // Fold this step's per-threadgroup partials into the double accumulators.
  // Only counters whose kernel ran this step are read.
  void collectStats(NUMERIC_TYPE dt, bool dyn, bool uni, bool drain) {
    if (dyn) acc_rain += sumPartials(rainPart, n_tg_cells);
    if (uni) acc_rain += sumPartials(urainPart, n_tg_cells);
    if (drain) acc_drain += sumPartials(drainPart, n_tg_cells);
    acc_clip += sumPartials(clipPart, n_tg_cells);
    double qin = 0.0, qout = 0.0;
    if (pBoundaryFlux) {
      qin = sumPartials(bndInPart, n_tg_edge);
      qout = sumPartials(bndOutPart, n_tg_edge);
    }
    // point sources count as inflow or outflow, as in BoundaryFlux()
    const float *q = (const float *)[psQ contents];
    for (int k = 0; k < n_ps; k++) {
      if (q[k] > 0.0f) qin += q[k]; else qout -= q[k];
    }
    last_qin = qin; last_qout = qout;
    acc_vol_in += qin * (double)dt;
    acc_vol_out += qout * (double)dt;
  }

  static double sumPartials(id<MTLBuffer> b, size_t n) {
    const float *v = (const float *)[b contents];
    double t = 0.0;
    for (size_t k = 0; k < n; k++) t += (double)v[k];
    return t;
  }

  // Hand the accumulated volumes to the CPU mass-balance bookkeeping. Called
  // once per mass interval, immediately before update_mass_stats().
  //
  // Volumes follow iterateq.cpp exactly: edge discharge integrated as Q*dt
  // per step, rain into VolInMT and the cumulative Rain column, NoData drain
  // into VolOutMT, and water created by the h < 0 clip booked nowhere, so that
  // it shows up in Verror as it does on the CPU.
  //
  // Qin/Qout do NOT follow the CPU ACC path, on purpose. There,
  // drain_nodata_water() adds the drained discharge to Qout and BoundaryFlux()
  // then overwrites Qout with edge faces only, so the drain is lost. On the
  // Alzette and Walferdange domains the FREE boundary lies entirely over NoData
  // and every drop leaves through the drain, so the CPU ACC Qout column reads
  // 0.000 for the whole run. Here Qout is (edge outflow + NoData drain) and Qin
  // is edge inflow, both averaged over the mass interval -- the convention of
  // the Metal FV1 backend and of CUDA, and the one that means "discharge
  // leaving the model".
  void syncMass() {
    const double elapsed = (double)time - last_mass_sync_time;
    if (boundCs) {
      boundCs->VolInMT += (NUMERIC_TYPE)(acc_vol_in + acc_rain);
      boundCs->VolOutMT += (NUMERIC_TYPE)(acc_vol_out + acc_drain);
      // infiltration and evaporation leave the domain (8.4)
      const double lost = losses.take_interval_volume();
      boundCs->VolOutMT += (NUMERIC_TYPE)lost;
      pars->InfilTotalLoss += (NUMERIC_TYPE)lost;
      pars->RainTotalLoss += (NUMERIC_TYPE)acc_rain;
      if (elapsed > 0.0) {
        boundCs->Qin = (NUMERIC_TYPE)(acc_vol_in / elapsed);
        boundCs->Qout = (NUMERIC_TYPE)((acc_vol_out + acc_drain) / elapsed);
      }
    }
    if (verbose) {
      printf("Metal ACC mass: t=%.1f  rain=%.4e  edge_in=%.4e  edge_out=%.4e  "
             "nodata_drain=%.4e  clip_created=%.4e m3\n",
             (double)time, acc_rain, acc_vol_in, acc_vol_out, acc_drain,
             acc_clip);
      fflush(stdout);
    }
    acc_rain = acc_drain = acc_clip = acc_vol_in = acc_vol_out = 0.0;
    last_mass_sync_time = (double)time;
  }

  void download(NUMERIC_TYPE *h_H, NUMERIC_TYPE *h_Qx, NUMERIC_TYPE *h_Qy) {
    const float *dH = (const float *)[H contents];
    for (size_t k = 0; k < n_cells; k++) h_H[k] = (NUMERIC_TYPE)dH[k];
    if (h_Qx && h_Qy) {
      const float *qx = (const float *)[Qx contents];
      const float *qy = (const float *)[Qy contents];
      for (size_t k = 0; k < n_q; k++) {
        h_Qx[k] = (NUMERIC_TYPE)qx[k];
        h_Qy[k] = (NUMERIC_TYPE)qy[k];
      }
    }
  }

  void downloadMax(NUMERIC_TYPE *mx, NUMERIC_TYPE *mxt, NUMERIC_TYPE *tot,
                   NUMERIC_TYPE *ini) {
    const float *a = (const float *)[maxH contents];
    const float *b = (const float *)[maxHtm contents];
    const float *c = (const float *)[totalHtm contents];
    const float *d = (const float *)[initHtm contents];
    for (size_t k = 0; k < n_cells; k++) {
      if (mx) mx[k] = (NUMERIC_TYPE)a[k];
      if (mxt) mxt[k] = (NUMERIC_TYPE)b[k];
      if (tot) tot[k] = (NUMERIC_TYPE)c[k];
      if (ini) ini[k] = (NUMERIC_TYPE)d[k];
    }
  }
};

// ------------------------------------------------------------- public API --

Simulator::Simulator(Fnames &filenames, Pars &pars, ::Solver &solver,
                     Arrays &arrays, States &states, BoundCs &boundCs,
                     int verbose) {
  impl = new SimulatorImpl();
  impl->arrays = &arrays;
  impl->boundCs = &boundCs;
  impl->pars = &pars;
  impl->solver = &solver;
  impl->states = &states;
  if (!impl->init(filenames, pars, solver, verbose)) {
    delete impl;
    impl = nullptr;
  }
}

Simulator::~Simulator() { delete impl; }

void Simulator::initialize(NUMERIC_TYPE *H, NUMERIC_TYPE *DEM,
                           NUMERIC_TYPE *Mann) {
  if (impl) impl->uploadFields(H, DEM, Mann);
}

NUMERIC_TYPE Simulator::step() { return impl ? impl->step() : 0; }

void Simulator::copy_to_host(NUMERIC_TYPE *H, NUMERIC_TYPE *Qx,
                             NUMERIC_TYPE *Qy) {
  if (impl) impl->download(H, Qx, Qy);
}

void Simulator::sync_mass() {
  if (impl) impl->syncMass();
}

void Simulator::copy_max_to_host(NUMERIC_TYPE *maxH, NUMERIC_TYPE *maxHtm,
                                 NUMERIC_TYPE *totalHtm,
                                 NUMERIC_TYPE *initHtm) {
  if (impl) impl->downloadMax(maxH, maxHtm, totalHtm, initHtm);
}

bool Simulator::is_available() {
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  return d != nil;
}

// ACC must NOT use swe/output.h's write_solution. That one writes Arrptr->HU and
// Arrptr->HV for .Qx/.Qy -- the FV1 momentum arrays, which allocate_swe_fields()
// only creates when Statesptr->fv1 is ON. Under `acceleration` they are null, so
// `qoutput` segfaults at the first save. ACC's fluxes live in Arrptr->Qx/Qy, and
// the writer that knows that is the global write_regular_output(), which is what
// the CPU ACC path in iterateq.cpp calls. It does not advance the save counters,
// so we do that here, exactly as iterateq.cpp does.
static void write_acc_output(Fnames *filenames, States *states, Pars *pars,
                             ::Solver *solver, Arrays *arrays) {
  write_regular_output(filenames, solver, states, pars, arrays, nullptr);
  pars->SaveTotal += pars->SaveInt;
  pars->SaveNo += 1;
}

void simulate(Fnames &filenames, Files *files, States &states, Pars &pars,
              ::Solver &solver, BoundCs &boundCs, Stage &stage, Arrays &arrays,
              int verbose) {
#if defined(LISFLOOD_METAL_MPI)
  if (states.cuda_mpi_gpus > 1) {
    fprintf(stderr,
            "ERROR: ACC solver does not support multi-GPU (staggered Qx/Qy grid "
            "prevents row decomposition). Use GPUs=1 or choose FV1/FV2/DG2.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
#endif
  Simulator sim(filenames, pars, solver, arrays, states, boundCs, verbose);
  if (!sim.is_valid()) {
    printf("ERROR: Metal ACC initialisation failed. Check that\n"
           "  1. metal_acc_kernels.metallib sits next to the executable\n"
           "  2. a Metal GPU is present\n");
    return;
  }

  sim.initialize(arrays.H, arrays.DEM, arrays.Manningsn);

  if (verbose) {
    printf("\nMetal ACC: %d x %d cells, dx=%.2f m, theta=%.3f\n", pars.xsz,
           pars.ysz, pars.dx, solver.theta);
    fflush(stdout);
  }

  time_t loop_start;
  time(&loop_start);
  solver.MinTstep = solver.InitTstep;
  solver.vol1 = DomainVol(&states, &pars, nullptr, &arrays, nullptr);

  while (solver.t < solver.Sim_Time) {
    zero_flux_stats(&boundCs);

    NUMERIC_TYPE dt = sim.step();
    if (dt <= 0) {
      printf("\nERROR: Metal ACC returned a non-positive timestep.\n");
      break;
    }
    if (solver.t > 0) solver.MinTstep = std::min(solver.MinTstep, dt);
    const NUMERIC_TYPE t_start = solver.t;
    solver.t += dt;
    solver.itCount++;

    // one line per step, as the CUDA solvers print (the 8.4 tests read it)
    if (verbose) printf("t=%f\tdt=%f\n", t_start, dt);

    if (solver.t >= pars.MassTotal) {
      sim.copy_to_host(arrays.H, arrays.Qx, arrays.Qy);
      sim.sync_mass();
      update_mass_stats(&states, &pars, &solver, &boundCs, &arrays);
      write_mass_stats(files, &pars, &solver, &boundCs);
      if (states.save_stages == ON) write_depth_samples(files, &pars, &solver, &stage, &arrays);
    }

    if (solver.t >= pars.SaveTotal) {
      sim.copy_to_host(arrays.H, arrays.Qx, arrays.Qy);
      write_acc_output(&filenames, &states, &pars, &solver, &arrays);
    }
  }

  sim.copy_to_host(arrays.H, arrays.Qx, arrays.Qy);
  sim.copy_max_to_host(arrays.maxH, arrays.maxHtm, arrays.totalHtm,
                       arrays.initHtm);
  // Only if a save is actually due. write_regular_output() has no
  // t >= SaveTotal guard of its own (the SWE write_solution does), so an
  // unconditional call here wrote a duplicate of the last save and left the
  // Metal run with one more .wd than the CPU run of the same .par.
  if (solver.t >= pars.SaveTotal)
    write_acc_output(&filenames, &states, &pars, &solver, &arrays);
  write_max_field(&filenames, &states, &pars, &arrays);

  time_t loop_end;
  time(&loop_end);
  if (verbose) {
    printf("\nloop time %lf\nFinished.\n\n", difftime(loop_end, loop_start));
  }
}

} // namespace acc
} // namespace metal
} // namespace lis

#endif // __APPLE__
