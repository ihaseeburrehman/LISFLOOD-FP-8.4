//
// LISFLOOD-FP Metal DG2 Solver - Implementation
// The Metal DG2 host with the DG2 time step of cuda/dg2 (nine coefficients per
// cell, friction at the Gauss points, Krivodonova slope limiter, two
// Runge-Kutta stages). HV keeps the CPU and CUDA sign (positive northward).
// Objective-C++ for Metal API access
//

#ifdef __APPLE__

#include "metal_dg2_simulate.h"
#include "../losses/metal_losses.h"
#include "../../rain/rain.h"
#include "../../rain/rain.tpp"
#include "../../swe/fv1.h"              // CPU FV1 solver functions
#include "../../swe/fv1/modifiedvars.h" // initialise_Zstar
#include "../../swe/output.h"           // write_mass_stats, write_solution
#include "../../swe/stats.h"            // update_mass_stats, DomainVol
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(LISFLOOD_METAL_MPI)
#include <mpi.h>
#endif

namespace lis {
namespace metal {
namespace dg2 {

// Threadgroup tile used by the flux-sharing sweeps (must match TG_X/TG_Y in
// metal_dg2_kernels.metal).
static const NSUInteger kTileX = 16;
static const NSUInteger kTileY = 16;

// Metal geometry structure (must match shader)
struct MetalGeometry {
  int32_t xsz;
  int32_t ysz;
  int32_t pitch;
  float dx;
  float dy;
  float tly;
  float blx;
  float bly;
  int32_t row_begin;        // first owned row (1 for single-GPU)
  int32_t row_end;          // one past last owned row (ysz+1 for single-GPU)
  int32_t owns_north_ghost; // 1 if rank 0
  int32_t owns_south_ghost; // 1 if last rank
};

// Metal physical params (must match shader)
struct MetalPhysicalParams {
  float g;
  float manning;
  float speed_thresh;
};

// Metal solver params (must match shader)
struct MetalSolverParams {
  float dt;
  float depth_thresh;
};

// Implementation class using PIMPL idiom
struct SimulatorImpl {
  // Metal objects
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLLibrary> library;

  // Compute pipelines
  id<MTLComputePipelineState> applyFrictionPipeline;
  id<MTLComputePipelineState> dg2ZstarPipeline;
  id<MTLComputePipelineState> dg2FrictionPipeline;
  id<MTLComputePipelineState> dg2MaxHPipeline;
  id<MTLComputePipelineState> dg2CopySlopesPipeline;
  id<MTLComputePipelineState> dg2LimitPipeline;
  id<MTLComputePipelineState> dg2ApplyLimitedPipeline;
  id<MTLComputePipelineState> dg2ZeroPerimeterPipeline;
  id<MTLComputePipelineState> dg2GhostPipeline;
  id<MTLComputePipelineState> dg2DtPipeline;
  id<MTLComputePipelineState> dg2ZeroDischargePipeline;
  id<MTLComputePipelineState> dg2DrainPipeline;
  id<MTLComputePipelineState> dg2FlowXPipeline;
  id<MTLComputePipelineState> dg2FlowYPipeline;
  // flow coefficients: 0 = Uold (the state), 1 = Uint, 2 = Ux; each buffer
  // holds 9 planes (H, H1x, H1y, HU, HU1x, HU1y, HV, HV1x, HV1y)
  id<MTLBuffer> U[3];
  id<MTLBuffer> slopeCopy;   // 6 planes: cuda dg2 Slopes
  id<MTLBuffer> limited;     // 6 planes: limited slopes, copied back
  id<MTLBuffer> maxBits;     // cuda dg2::maxH, as float bits
  bool limit_slopes = false;
  bool friction_on = false;
  float krivodonova_thresh = 10.0f;
  id<MTLComputePipelineState> computeZstarXPipeline;
  id<MTLComputePipelineState> computeZstarYPipeline;
  id<MTLComputePipelineState> updateFlowXPipeline;
  id<MTLComputePipelineState> updateFlowYPipeline;
  id<MTLComputePipelineState> computeDtPipeline;
  id<MTLComputePipelineState> zeroGhostCellsPipeline;
  id<MTLComputePipelineState> applyUniformRainPipeline;
  id<MTLComputePipelineState> applyRainPipeline; // For dynamic rain
  id<MTLComputePipelineState>
      updateGhostCellsPipeline; // Reflective ghost cells
  id<MTLComputePipelineState>
      drainNodataPipeline; // Drain water from NoData cells
  id<MTLComputePipelineState>
      accumulateOutflowPipeline; // Accumulate water leaving domain
  id<MTLComputePipelineState>
      updateMaxFieldPipeline; // Running per-cell maximum depth

  // Buffers for flow variables (double-buffered)
  id<MTLBuffer> H[2];   // the H planes of U[0] and U[1] (same buffers)
  id<MTLBuffer> maxH; // Running maximum depth, for the .max output

  // Buffers for grid data
  id<MTLBuffer> DEM;
  id<MTLBuffer> Zstar_x;
  id<MTLBuffer> Zstar_y;
  id<MTLBuffer> manning;
  id<MTLBuffer> dt_field;
  id<MTLBuffer> minDtBuffer; // For atomic reduction

  // Parameter buffers
  id<MTLBuffer> geomBuffer;
  id<MTLBuffer> physicsBuffer;
  id<MTLBuffer> solverBuffer;

  // Boundary flux buffers
  id<MTLBuffer> flux_west;
  id<MTLBuffer> flux_east;
  id<MTLBuffer> flux_north;
  id<MTLBuffer> flux_south;

  // Rain buffers
  id<MTLBuffer> rainDataBuffer;
  id<MTLBuffer> rainGeometryBuffer;

  // Boundary condition type buffer, plus the per-edge-cell prescribed value
  // (stage or unit discharge) refreshed every step for the time-varying types.
  id<MTLBuffer> boundaryTypesBuffer;
  id<MTLBuffer> boundaryValuesBuffer;

  // Simulation state
  int current_buffer; // 0 or 1 for double buffering
  MetalGeometry geom;
  MetalPhysicalParams physics;
  MetalSolverParams solverParams;
  NUMERIC_TYPE time;
  NUMERIC_TYPE cfl;
  NUMERIC_TYPE max_dt;
  NUMERIC_TYPE nodata;
  // The GPU works in single precision, so the DEM is stored relative to the
  // lowest valid elevation and NoData cells are given a pit a fixed distance
  // below that, rather than the raw nodata_elevation (typically -9999).
  // The FV1 scheme only ever uses elevation *differences* (Z*, ETA - Z*,
  // Zdagger differences), so the shift is exact; what it buys is conditioning.
  // At -9999 a float32 ulp is ~9.8e-4 m, i.e. the same size as DepthThresh, so
  // (h + Z) - Z* inside the NoData region lost the depth entirely and the
  // scheme silently leaked mass. See dem_offset/gpu_nodata use in copyToGPU().
  float dem_offset;
  float gpu_nodata;
  bool has_manning_map;
  bool is_first_step;
  Losses losses;                // infiltration and evaporation (8.4)
  Fnames *fnames = nullptr;
  std::string filenames_dem, filenames_start;
  int verbose;
  id<MTLBuffer> volOutBuffer; // Buffer for tracking outflow volume
  double last_mass_sync_time = 0.0;
  double vol_out_total;       // Host-side cumulative outflow (m3)
  double last_rain_time;      // Simulation time of last rain upload
  double current_rain_sum;    // Sum of current rain grid for sub-basin (m3/s)
  double
      interval_out_total; // Outflow accumulated since last mass balance check

  size_t buffer_size;
  size_t num_cells;

  // CPU data pointers
  Arrays *arrays;
  BoundCs *boundCs;
  Pars *pars;
  ::Solver *solver;
  States *states;

  // MPI partition
  int mpi_rank = 0;
  int mpi_size = 1;
  int row_begin = 1;
  int row_end = 0; // set in init

  // Dynamic Rain object
  ::DynamicRain<std::allocator<NUMERIC_TYPE>> *dynamic_rain;

  SimulatorImpl()
      : device(nil), commandQueue(nil), library(nil), current_buffer(0),
        time(0), cfl(0.7), max_dt(10.0), nodata(-9999), dem_offset(0.0f),
        gpu_nodata(-1000.0f), has_manning_map(false),
        is_first_step(true), verbose(0), volOutBuffer(nil), vol_out_total(0.0), last_rain_time(-1.0),
        current_rain_sum(0.0), interval_out_total(0.0), arrays(nullptr),
        boundCs(nullptr), pars(nullptr), solver(nullptr), states(nullptr),
        dynamic_rain(nullptr) {}

  ~SimulatorImpl() {
    if (dynamic_rain) {
      delete dynamic_rain;
      dynamic_rain = nullptr;
    }
  }

  bool init(Fnames &filenames, Pars &pars, ::Solver &solver, int verb,
            int rank = 0, int size = 1, int r_begin = 1, int r_end = 0) {
    verbose = verb;
    mpi_rank = rank;
    mpi_size = size;
    row_begin = r_begin;
    row_end = (r_end > r_begin) ? r_end : (pars.ysz + 1);
    fnames = &filenames;

    // Initialize Dynamic Rain
    // Note: DynamicRain constructor reads file header.
    dynamic_rain = new ::DynamicRain<std::allocator<NUMERIC_TYPE>>(
        filenames.dynamicrainfilename, verbose);

    // Get default Metal device
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      if (verbose)
        printf("Metal: No GPU device found\n");
      return false;
    }

    if (verbose) {
      printf("Metal: Using GPU: %s\n", [[device name] UTF8String]);
      printf("Metal: Unified Memory: %s\n",
             device.hasUnifiedMemory ? "Yes" : "No");
    }

    // Create command queue
    commandQueue = [device newCommandQueue];
    if (!commandQueue) {
      if (verbose)
        printf("Metal: Failed to create command queue\n");
      return false;
    }

    // Load shader library
    NSError *error = nil;

    // 1. Try bundle (portable for apps)
    NSString *libraryPath =
        [[NSBundle mainBundle] pathForResource:@"metal_dg2_kernels"
                                        ofType:@"metallib"];

    // 2. Try current directory
    if (!libraryPath) {
      if ([[NSFileManager defaultManager]
              fileExistsAtPath:@"metal_dg2_kernels.metallib"]) {
        libraryPath = @"metal_dg2_kernels.metallib";
      }
    }

    // 3. Try directory of executable
    if (!libraryPath) {
      NSString *executablePath = [[NSProcessInfo processInfo] arguments][0];
      NSString *executableDir =
          [executablePath stringByDeletingLastPathComponent];
      NSString *path = [executableDir
          stringByAppendingPathComponent:@"metal_dg2_kernels.metallib"];
      if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
        libraryPath = path;
      }
    }

    if (libraryPath) {
      if (verbose) printf("Metal: Shader library: %s\n", [libraryPath UTF8String]);
      library = [device newLibraryWithFile:libraryPath error:&error];
    }

    // 4. Compile from source if the source is found relative to the executable
    if (!library) {
      NSString *executablePath = [[NSProcessInfo processInfo] arguments][0];
      NSString *executableDir =
          [executablePath stringByDeletingLastPathComponent];
      // Go up from build/ to find metal/fv1/
      NSString *shaderPath =
          [executableDir stringByAppendingPathComponent:
                             @"../metal/dg2/metal_dg2_kernels.metal"];
      NSString *shaderSource =
          [NSString stringWithContentsOfFile:shaderPath
                                    encoding:NSUTF8StringEncoding
                                       error:nil];

      if (shaderSource) {
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        library = [device newLibraryWithSource:shaderSource
                                       options:options
                                         error:&error];
      }
    }

    if (!library) {
      if (verbose)
        printf("Metal: Failed to load shader library: %s\n",
               [[error localizedDescription] UTF8String]);
      return false;
    }

    // Create compute pipelines
    if (!createPipelines()) {
      return false;
    }

    // Set up geometry
    geom.xsz = pars.xsz;
    geom.ysz = pars.ysz;
    geom.pitch = pars.xsz + 2; // Add ghost cells
    geom.dx = (float)pars.dx;
    geom.dy = (float)pars.dy;
    geom.tly = (float)pars.tly;
    geom.blx = (float)pars.blx;
    geom.bly = (float)pars.bly;
    geom.row_begin = row_begin;
    geom.row_end = row_end;
    geom.owns_north_ghost = (mpi_rank == 0) ? 1 : 0;
    geom.owns_south_ghost = (mpi_rank == mpi_size - 1) ? 1 : 0;

    // Physical params
    physics.g = (float)solver.g;
    physics.manning = (float)pars.FPn;
    physics.speed_thresh = (float)solver.SpeedThresh;

    // Solver params
    solverParams.dt = (float)solver.InitTstep;
    solverParams.depth_thresh = (float)solver.DepthThresh;

    cfl = solver.cfl;
    limit_slopes = (pars.limit_slopes == ON);
    friction_on = (pars.FPn > 0.0) || (strlen(filenames.nfilename) > 0);
    krivodonova_thresh = (float)solver.krivodonova_threshold;
    filenames_dem = filenames.demfilename;
    filenames_start = filenames.startfilename;
    max_dt = solver.InitTstep;
    nodata = pars.nodata_elevation;

    // Calculate buffer sizes
    num_cells = (geom.xsz + 2) * (geom.ysz + 2);
    buffer_size = num_cells * sizeof(float);

    // Allocate buffers
    if (!allocateBuffers()) {
      return false;
    }

    // Allocate Rain Buffers
    if (dynamic_rain && dynamic_rain->enabled()) {
      const Geometry &rgeom = dynamic_rain->geometry();

      // Rain Data Buffer
      size_t rsize = rgeom.xsz * rgeom.ysz *
                     sizeof(NUMERIC_TYPE); // assuming standard float
      // Note: rain data is NUMERIC_TYPE (likely float or double).
      // If double, we might need conversion or kernels that handle double
      // (Metal supports float primarily, double on some). Ideally convert to
      // float if NUMERIC_TYPE is double. For now assume NUMERIC_TYPE is float
      // (as per user objective "single precision").

      rainDataBuffer =
          [device newBufferWithLength:rsize
                              options:MTLResourceStorageModeManaged];

      // Rain Geometry (MetalGeometry)
      MetalGeometry mRGeom;
      mRGeom.xsz = rgeom.xsz;
      mRGeom.ysz = rgeom.ysz;
      mRGeom.dx = (float)rgeom.dx;
      mRGeom.dy = (float)rgeom.dy;
      mRGeom.tly = (float)rgeom.tly;
      mRGeom.blx = (float)rgeom.blx;
      mRGeom.bly = (float)rgeom.bly;
      // Pitch not really used for rain grid usually linear, but let's set it
      mRGeom.pitch = rgeom.xsz;

      rainGeometryBuffer =
          [device newBufferWithBytes:&mRGeom
                              length:sizeof(MetalGeometry)
                             options:MTLResourceStorageModeShared];

      if (verbose) {
        printf("Metal: Dynamic Rain enabled. Grid: %d x %d\n", rgeom.xsz,
               rgeom.ysz);
      }
    }

    if (verbose) {
      printf("Metal: Grid size: %d x %d\n", geom.xsz, geom.ysz);
      printf("Metal: dx = %.2f m, dy = %.2f m\n", geom.dx, geom.dy);
      printf("Metal: g = %.2f m/s², CFL = %.2f, max_dt = %.2f s\n", physics.g,
             cfl, max_dt);
      printf("Metal: Buffer size: %.2f MB per field\n",
             buffer_size / (1024.0 * 1024.0));
    }

    return true;
  }

  bool createPipelines() {
    NSError *error = nil;

    auto createPipeline = [&](const char *name) -> id<MTLComputePipelineState> {
      id<MTLFunction> func =
          [library newFunctionWithName:[NSString stringWithUTF8String:name]];
      if (!func) {
        if (verbose)
          printf("Metal: Function '%s' not found\n", name);
        return nil;
      }
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:func error:&error];
      if (!pipeline && verbose) {
        printf("Metal: Failed to create pipeline for '%s'\n", name);
      }
      return pipeline;
    };

    applyFrictionPipeline = createPipeline("apply_friction");
    computeZstarXPipeline = createPipeline("compute_zstar_x");
    computeZstarYPipeline = createPipeline("compute_zstar_y");
    updateFlowXPipeline = createPipeline("update_flow_x");
    updateFlowYPipeline = createPipeline("update_flow_y");
    computeDtPipeline = createPipeline("compute_dt_per_cell");
    zeroGhostCellsPipeline = createPipeline("zero_ghost_cells");
    applyUniformRainPipeline = createPipeline("apply_uniform_rain");
    applyRainPipeline = createPipeline("apply_rain");
    updateGhostCellsPipeline = createPipeline("update_ghost_cells");
    drainNodataPipeline = createPipeline("drain_nodata_cells");
    accumulateOutflowPipeline = createPipeline("accumulate_outflow");
    updateMaxFieldPipeline = createPipeline("update_max_field");
    dg2ZstarPipeline = createPipeline("dg2_zstar");
    dg2FrictionPipeline = createPipeline("dg2_friction");
    dg2MaxHPipeline = createPipeline("dg2_max_h");
    dg2CopySlopesPipeline = createPipeline("dg2_copy_slopes");
    dg2LimitPipeline = createPipeline("dg2_limit_slopes");
    dg2ApplyLimitedPipeline = createPipeline("dg2_apply_limited");
    dg2ZeroPerimeterPipeline = createPipeline("dg2_zero_perimeter_slopes");
    dg2GhostPipeline = createPipeline("dg2_update_ghost_cells");
    dg2DtPipeline = createPipeline("dg2_compute_dt");
    dg2ZeroDischargePipeline = createPipeline("dg2_zero_discharge");
    dg2DrainPipeline = createPipeline("dg2_drain_nodata");
    dg2FlowXPipeline = createPipeline("dg2_flow_x");
    dg2FlowYPipeline = createPipeline("dg2_flow_y");

    return dg2ZstarPipeline && dg2FrictionPipeline && dg2MaxHPipeline &&
           dg2CopySlopesPipeline && dg2LimitPipeline && dg2ApplyLimitedPipeline &&
           dg2ZeroPerimeterPipeline && dg2GhostPipeline && dg2DtPipeline &&
           dg2ZeroDischargePipeline && dg2DrainPipeline && dg2FlowXPipeline &&
           dg2FlowYPipeline && applyRainPipeline && applyUniformRainPipeline &&
           updateMaxFieldPipeline;
  }

  bool allocateBuffers() {
    MTLResourceOptions options = MTLResourceStorageModeShared; // Unified memory

    for (int s = 0; s < 3; s++) {
      U[s] = [device newBufferWithLength:9 * buffer_size options:options];
      if (!U[s]) return false;
      memset([U[s] contents], 0, 9 * buffer_size);
    }
    H[0] = U[0];
    H[1] = U[1];
    slopeCopy = [device newBufferWithLength:6 * buffer_size options:options];
    limited = [device newBufferWithLength:6 * buffer_size options:options];
    maxBits = [device newBufferWithLength:sizeof(uint32_t) options:options];
    maxH = [device newBufferWithLength:buffer_size options:options];
    if (!slopeCopy || !limited || !maxBits || !maxH) return false;
    memset([slopeCopy contents], 0, 6 * buffer_size);
    memset([limited contents], 0, 6 * buffer_size);

    // DEM: 3 planes (Z, Z1x, Z1y)
    DEM = [device newBufferWithLength:3 * buffer_size options:options];
    Zstar_x = [device newBufferWithLength:buffer_size options:options];
    Zstar_y = [device newBufferWithLength:buffer_size options:options];
    minDtBuffer = [device newBufferWithLength:sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];
    manning = [device newBufferWithLength:buffer_size options:options];
    if (!DEM || !Zstar_x || !Zstar_y || !minDtBuffer || !manning) return false;
    memset([DEM contents], 0, 3 * buffer_size);
    memset([Zstar_x contents], 0, buffer_size);
    memset([Zstar_y contents], 0, buffer_size);
    memset([manning contents], 0, buffer_size);

    geomBuffer = [device newBufferWithBytes:&geom
                                     length:sizeof(MetalGeometry)
                                    options:options];
    physicsBuffer = [device newBufferWithBytes:&physics
                                        length:sizeof(MetalPhysicalParams)
                                       options:options];
    solverBuffer = [device newBufferWithBytes:&solverParams
                                       length:sizeof(MetalSolverParams)
                                      options:options];

    int num_bc_cells = 2 * geom.xsz + 2 * geom.ysz;
    boundaryTypesBuffer =
        [device newBufferWithLength:num_bc_cells * sizeof(int32_t)
                            options:options];
    boundaryValuesBuffer =
        [device newBufferWithLength:num_bc_cells * sizeof(float)
                            options:options];
    memset([boundaryTypesBuffer contents], 0, num_bc_cells * sizeof(int32_t));
    memset([boundaryValuesBuffer contents], 0, num_bc_cells * sizeof(float));

    volOutBuffer = [device newBufferWithLength:8 * sizeof(float)
                                       options:options];
    memset([volOutBuffer contents], 0, 8 * sizeof(float));

    return geomBuffer && physicsBuffer && solverBuffer && boundaryTypesBuffer &&
           boundaryValuesBuffer && volOutBuffer;
  }

  // An ESRI ASCII grid of the DEM's size (as cuda AsciiRaster::read: row 0 is
  // the north edge), values as written; false if the file cannot be read
  static bool read_ascii_grid(const char *path, int xsz, int ysz,
                              std::vector<double> &out) {
    FILE *f = fopen(path, "r");
    if (!f) return false;
    char key[64];
    double value;
    for (int k = 0; k < 6; k++) {
      long pos = ftell(f);
      if (fscanf(f, "%63s", key) != 1) { fclose(f); return false; }
      if (!isalpha((unsigned char)key[0])) { fseek(f, pos, SEEK_SET); break; }
      if (fscanf(f, "%lf", &value) != 1) { fclose(f); return false; }
    }
    out.assign((size_t)xsz * ysz, 0.0);
    for (size_t k = 0; k < out.size(); k++) {
      if (fscanf(f, "%lf", &out[k]) != 1) { fclose(f); return false; }
    }
    fclose(f);
    return true;
  }

  void copyToGPU(NUMERIC_TYPE *h_H, NUMERIC_TYPE *h_HU, NUMERIC_TYPE *h_HV,
                 NUMERIC_TYPE *h_DEM, NUMERIC_TYPE *h_Mann) {
    const int N = (int)num_cells;
    float *d_U = (float *)[U[0] contents];
    float *d_Z = (float *)[DEM contents];
    float *d_Mann = (float *)[manning contents];
    for (int s = 0; s < 3; s++) memset([U[s] contents], 0, 9 * buffer_size);
    memset([DEM contents], 0, 3 * buffer_size);
    memset([manning contents], 0, buffer_size);
    memset([maxH contents], 0, buffer_size);

    // Vertical datum: the lowest valid elevation (float32 conditioning). DG2
    // keeps NoData cells at nodata_elevation in this datum, as cuda/dg2 does:
    // their face elevations come from the DG2 slope files (which hold the
    // NoData value there), and remapping only the mean would change them.
    float nodata_val = (float)nodata;
    {
      double min_valid = 0.0;
      bool any_valid = false;
      for (int k = 0; k < geom.xsz * geom.ysz; k++) {
        double z = (double)h_DEM[k];
        if (fabs(z - (double)nodata) < 1.0) continue;
        if (!any_valid || z < min_valid) min_valid = z;
        any_valid = true;
      }
      dem_offset = any_valid ? (float)min_valid : 0.0f;
      gpu_nodata = (float)((double)nodata - (double)dem_offset);
      if (verbose)
        printf("Metal: DEM datum shifted by %.3f m; NoData at %.1f m\n",
               dem_offset, gpu_nodata);
    }

    std::vector<double> z1x, z1y;
    std::string base = filenames_dem;
    if (!read_ascii_grid((base + "1x").c_str(), geom.xsz, geom.ysz, z1x) ||
        !read_ascii_grid((base + "1y").c_str(), geom.xsz, geom.ysz, z1y)) {
      if (states->mwdg2 == ON) {
        // Adaptive input files contain finest-grid cell averages rather than
        // the preprocessed DG2 modal rasters.  Start their unresolved modes
        // at zero; the RKDG2 update evolves them immediately.  This is the
        // conservative projection of a piecewise-constant finest-grid DEM.
        z1x.assign((size_t)geom.xsz * geom.ysz, 0.0);
        z1y.assign((size_t)geom.xsz * geom.ysz, 0.0);
        if (verbose)
          printf("Metal MWDG2: DEM modal files absent; using conservative "
                 "zero-slope projection\n");
      } else {
        fprintf(stderr, "ERROR: Metal DG2 needs the DEM slope files %s1x and %s1y "
                        "(generateDG2DEM)\n", base.c_str(), base.c_str());
        std::exit(EXIT_FAILURE);
      }
    }
    std::vector<double> h1x, h1y;
    if (states->startfile == ON) {
      std::string sbase = filenames_start;
      if (!read_ascii_grid((sbase + "1x").c_str(), geom.xsz, geom.ysz, h1x) ||
          !read_ascii_grid((sbase + "1y").c_str(), geom.xsz, geom.ysz, h1y)) {
        if (states->mwdg2 == ON) {
          h1x.assign((size_t)geom.xsz * geom.ysz, 0.0);
          h1y.assign((size_t)geom.xsz * geom.ysz, 0.0);
          if (verbose)
            printf("Metal MWDG2: start modal files absent; using conservative "
                   "zero-slope projection\n");
        } else {
          fprintf(stderr, "ERROR: Metal DG2 needs the start file slopes %s1x and "
                          "%s1y (generateDG2start)\n", sbase.c_str(), sbase.c_str());
          std::exit(EXIT_FAILURE);
        }
      }
    }

    for (int j = 0; j < geom.ysz; j++) {
      for (int i = 0; i < geom.xsz; i++) {
        int k = j * geom.xsz + i;
        int c = (j + 1) * geom.pitch + (i + 1);
        float dem = (float)h_DEM[k];
        bool is_nodata = (fabs(dem - nodata_val) < 1.0f);
        d_Z[c] = is_nodata ? gpu_nodata : dem - dem_offset;
        d_Z[N + c] = (float)z1x[k];
        d_Z[2 * N + c] = (float)z1y[k];
        d_Mann[c] = h_Mann ? (float)h_Mann[k] : physics.manning;
        if (!is_nodata) {
          d_U[c] = (float)h_H[k];
          d_U[3 * N + c] = (float)h_HU[k];
          d_U[6 * N + c] = (float)h_HV[k];
          if (!h1x.empty()) {
            d_U[N + c] = (float)h1x[k];
            d_U[2 * N + c] = (float)h1y[k];
          }
        }
      }
    }
    has_manning_map = (h_Mann != nullptr);
    if (fnames)
      losses.setup(device, *fnames, *states, *pars, arrays, geom.xsz, geom.ysz,
                   geom.pitch, 1, 1, num_cells, verbose);

    // cuda dg2 Topography: zero_perimeter_slopes, then clamp_boundary_values
    for (int j = 1; j <= geom.ysz; j++) {
      for (int i : {1, geom.xsz}) {
        d_Z[N + j * geom.pitch + i] = 0.0f;
        d_Z[2 * N + j * geom.pitch + i] = 0.0f;
      }
    }
    for (int i = 1; i <= geom.xsz; i++) {
      for (int j : {1, geom.ysz}) {
        d_Z[N + j * geom.pitch + i] = 0.0f;
        d_Z[2 * N + j * geom.pitch + i] = 0.0f;
      }
    }
    for (int j = 1; j <= geom.ysz; j++) {
      d_Z[j * geom.pitch] = d_Z[j * geom.pitch + 1];
      d_Z[j * geom.pitch + geom.xsz + 1] = d_Z[j * geom.pitch + geom.xsz];
    }
    for (int i = 1; i <= geom.xsz; i++) {
      d_Z[i] = d_Z[geom.pitch + i];
      d_Z[(geom.ysz + 1) * geom.pitch + i] = d_Z[geom.ysz * geom.pitch + i];
    }

    if (boundCs && boundCs->BC_Ident) {
      int32_t *d_bc = (int32_t *)[boundaryTypesBuffer contents];
      int num_bc_cells = 2 * geom.xsz + 2 * geom.ysz;
      for (int i = 0; i < num_bc_cells; i++)
        d_bc[i] = (int32_t)boundCs->BC_Ident[i];
    }

    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      [encoder setComputePipelineState:dg2ZstarPipeline];
      [encoder setBuffer:DEM offset:0 atIndex:0];
      [encoder setBuffer:Zstar_x offset:0 atIndex:1];
      [encoder setBuffer:Zstar_y offset:0 atIndex:2];
      [encoder setBuffer:geomBuffer offset:0 atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(geom.xsz + 1, geom.ysz + 1, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
    }
  }

  void syncMassTotals() {
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];

      float *totals = (float *)[volOutBuffer contents];
      const double out_boundary = (double)totals[0];
      const double out_drain = (double)totals[2];
      const double out = out_boundary + out_drain;
      const double in = totals[1];
      const double uniform_rain = totals[3];
      const double elapsed = (double)time - last_mass_sync_time;

      if (verbose && elapsed > 0.0) {
        printf("Metal mass: t=%.1f  boundary_out=%.4e  nodata_drain=%.4e  "
               "boundary_in=%.4e\n",
               (double)time, out_boundary, out_drain, in);
        fflush(stdout);
      }
      if (boundCs) {
        boundCs->VolOutMT += (NUMERIC_TYPE)out;
        // infiltration and evaporation leave the domain (8.4)
        const double lost = losses.take_interval_volume();
        boundCs->VolOutMT += (NUMERIC_TYPE)lost;
        pars->InfilTotalLoss += (NUMERIC_TYPE)lost;
        boundCs->VolInMT += (NUMERIC_TYPE)(in + uniform_rain);
        pars->RainTotalLoss += (NUMERIC_TYPE)uniform_rain;
        if (elapsed > 0.0) {
          boundCs->Qout += (NUMERIC_TYPE)(out / elapsed);
          boundCs->Qin += (NUMERIC_TYPE)(in / elapsed);
        }
      }
      last_mass_sync_time = (double)time;

      // Zero the GPU counters for the next interval
      memset([volOutBuffer contents], 0, 8 * sizeof(float));
    }
  }

  void copyFromGPU(NUMERIC_TYPE *h_H, NUMERIC_TYPE *h_HU, NUMERIC_TYPE *h_HV) {
    // Sync mass balance first to ensure the final numbers are captured
    syncMassTotals();

    float *d_H = (float *)[U[0] contents];
    float *d_HU = d_H + 3 * num_cells;
    float *d_HV = d_H + 6 * num_cells;

    for (int j = 0; j < geom.ysz; j++) {
      for (int i = 0; i < geom.xsz; i++) {
        int cpu_idx = j * geom.xsz + i;
        int gpu_idx = (j + 1) * geom.pitch + (i + 1);

        h_H[cpu_idx] = (NUMERIC_TYPE)d_H[gpu_idx];
        h_HU[cpu_idx] = (NUMERIC_TYPE)d_HU[gpu_idx];
        h_HV[cpu_idx] = (NUMERIC_TYPE)d_HV[gpu_idx];
      }
    }
  }

  void copyMaxFromGPU(NUMERIC_TYPE *h_maxH) {
    if (h_maxH == nullptr)
      return;

    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
    }

    const float *d_maxH = (const float *)[maxH contents];
    for (int j = 0; j < geom.ysz; j++) {
      for (int i = 0; i < geom.xsz; i++) {
        h_maxH[j * geom.xsz + i] =
            (NUMERIC_TYPE)d_maxH[(j + 1) * geom.pitch + (i + 1)];
      }
    }
  }

#if defined(LISFLOOD_METAL_MPI)
  // Exchange halo rows with neighbouring MPI ranks for DG2 U buffer (9 planes).
  // U_buf is one of U[0]..U[2]; tags 46 (north->south) / 47 (south->north).
  void exchangeHalos(id<MTLBuffer> U_buf) {
    if (mpi_size <= 1) return;
    const int plane_size = geom.pitch * (geom.ysz + 2); // elements per plane
    const int n = 9 * geom.pitch;                        // floats per halo row
    static std::vector<float> send_buf;
    static std::vector<float> recv_buf;
    if ((int)send_buf.size() != 2 * n) { send_buf.resize(2 * n); recv_buf.resize(2 * n); }

    float *ptr = (float *)[U_buf contents];
    const int north_row = geom.row_begin;
    const int south_row = geom.row_end - 1;
    for (int p = 0; p < 9; ++p) {
      float *plane = ptr + p * plane_size;
      for (int col = 0; col < geom.pitch; ++col) {
        send_buf[p * geom.pitch + col]     = plane[north_row * geom.pitch + col];
        send_buf[n + p * geom.pitch + col] = plane[south_row * geom.pitch + col];
      }
    }

    const int north_rank = mpi_rank > 0          ? mpi_rank - 1 : MPI_PROC_NULL;
    const int south_rank = mpi_rank + 1 < mpi_size ? mpi_rank + 1 : MPI_PROC_NULL;
    MPI_Request requests[4];
    MPI_Irecv(recv_buf.data(),     n, MPI_FLOAT, south_rank, 46, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(recv_buf.data() + n, n, MPI_FLOAT, north_rank, 47, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(send_buf.data(),     n, MPI_FLOAT, north_rank, 46, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(send_buf.data() + n, n, MPI_FLOAT, south_rank, 47, MPI_COMM_WORLD, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    if (mpi_rank + 1 < mpi_size) {
      const int halo_south = geom.row_end;
      for (int p = 0; p < 9; ++p) {
        float *plane = ptr + p * plane_size;
        for (int col = 0; col < geom.pitch; ++col)
          plane[halo_south * geom.pitch + col] = recv_buf[p * geom.pitch + col];
      }
    }
    if (mpi_rank > 0) {
      const int halo_north = geom.row_begin - 1;
      for (int p = 0; p < 9; ++p) {
        float *plane = ptr + p * plane_size;
        for (int col = 0; col < geom.pitch; ++col)
          plane[halo_north * geom.pitch + col] = recv_buf[n + p * geom.pitch + col];
      }
    }
  }

  // Gather all ranks' owned rows into rank 0's U buffer (9 planes of Uold).
  void gatherRows(id<MTLBuffer> U_buf) {
    if (mpi_size <= 1) return;
    const int plane_size = geom.pitch * (geom.ysz + 2);
    const int first = geom.row_begin;
    const int last  = geom.row_end;
    std::vector<int> counts(mpi_size), displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
      int lo = 1 + (geom.ysz * r) / mpi_size;
      int hi = 1 + (geom.ysz * (r + 1)) / mpi_size;
      counts[r] = (hi - lo) * geom.xsz;
      displs[r] = (lo - 1) * geom.xsz;
    }
    float *ptr = (float *)[U_buf contents];
    for (int p = 0; p < 9; ++p) {
      float *plane = ptr + p * plane_size;
      std::vector<float> local((last - first) * geom.xsz);
      std::vector<float> global;
      if (mpi_rank == 0) global.resize(geom.ysz * geom.xsz);
      for (int j = first; j < last; ++j)
        std::memcpy(local.data() + (j - first) * geom.xsz,
                    plane + j * geom.pitch + 1, geom.xsz * sizeof(float));
      MPI_Gatherv(local.data(), (int)local.size(), MPI_FLOAT,
                  mpi_rank == 0 ? global.data() : nullptr, counts.data(),
                  displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
      if (mpi_rank == 0) {
        for (int j = 1; j <= geom.ysz; ++j)
          std::memcpy(plane + j * geom.pitch + 1,
                      global.data() + (j - 1) * geom.xsz, geom.xsz * sizeof(float));
      }
    }
  }
#else
  void exchangeHalos(id<MTLBuffer>) {}
  void gatherRows(id<MTLBuffer>) {}
#endif

  // Refresh the prescribed boundary values for this timestep. Stage boundaries
  // are shifted into the GPU's vertical datum so that (value - Z) is evaluated
  // consistently with the shifted DEM; discharge boundaries are in m2/s and
  // need no shift.
  void updateBoundaryValues() {
    if (!boundCs || !boundCs->BC_Ident || !boundaryValuesBuffer)
      return;
    float *d_val = (float *)[boundaryValuesBuffer contents];
    const int num_bc_cells = 2 * geom.xsz + 2 * geom.ysz;
    for (int i = 0; i < num_bc_cells; i++) {
      switch (boundCs->BC_Ident[i]) {
      case HFIX2:
        d_val[i] = (float)boundCs->BC_Val[i] - dem_offset;
        break;
      case HVAR3:
        d_val[i] = (float)InterpolateTimeSeries(boundCs->BC_TimeSeries[i], time) -
                   dem_offset;
        break;
      case QFIX4:
        d_val[i] = (float)boundCs->BC_Val[i];
        break;
      case QVAR5:
        d_val[i] =
            (float)InterpolateTimeSeries(boundCs->BC_TimeSeries[i], time);
        break;
      default:
        d_val[i] = 0.0f;
        break;
      }
    }
  }

  void applyPointSources() {
    if (!boundCs || boundCs->numPS <= 0)
      return;

    int xsz = geom.xsz;
    int ysz = geom.ysz;
    float dx = geom.dx;
    float dy = geom.dy;
    float dt = solverParams.dt;
    float cell_area = dx * dy;

    // Get direct pointer to GPU H buffer (unified memory)
    float *d_H = (float *)[H[current_buffer] contents];

    for (int i = 0; i < boundCs->numPS; i++) {
      int px = boundCs->xpi[i];
      int py = boundCs->ypi[i];

      if (px < 0 || px >= xsz || py < 0 || py >= ysz)
        continue;

      float val = 0.0f;
      bool is_q_type = false;
      if (boundCs->PS_Ident[i] == QVAR5) {
        val = (float)InterpolateTimeSeries(boundCs->PS_TimeSeries[i], time);
        is_q_type = true;
      } else if (boundCs->PS_Ident[i] == QFIX4) {
        val = (float)boundCs->PS_Val[i];
        is_q_type = true;
      } else if (boundCs->PS_Ident[i] == HVAR3) {
        float h_target =
            (float)InterpolateTimeSeries(boundCs->PS_TimeSeries[i], time);
        float z = arrays->DEM[py * xsz + px];
        float h_new = std::max(0.0f, h_target - z);
        float h_old = d_H[(py + 1) * geom.pitch + (px + 1)];
        val = (h_new - h_old) * cell_area / dt; // Calculate Q to match H
        is_q_type = false;                      // Already scaled to m3/s
      } else if (boundCs->PS_Ident[i] == HFIX2) {
        float h_target = (float)boundCs->PS_Val[i];
        float z = arrays->DEM[py * xsz + px];
        float h_new = std::max(0.0f, h_target - z);
        float h_old = d_H[(py + 1) * geom.pitch + (px + 1)];
        val = (h_new - h_old) * cell_area / dt; // Calculate Q to match H
        is_q_type = false;                      // Already scaled to m3/s
      }

      if (std::abs(val) > 1e-9f) {
        float Q_total = is_q_type ? (val * dx) : val;

        // GPU index (includes ghost cells)
        int gpu_idx = (py + 1) * geom.pitch + (px + 1);

        // Add volume: dH = Q_total * dt / Area
        float dH = (Q_total * dt) / cell_area;

        // Update GPU buffer directly (Unified memory)
        d_H[gpu_idx] += dH;

        // Update stats for mass balance (Qin is discharge m3/s)
        if (Q_total > 0) {
          boundCs->Qin += Q_total;
          boundCs->VolInMT += (NUMERIC_TYPE)(Q_total * dt);
        } else {
          boundCs->Qout -= Q_total;
          boundCs->VolOutMT -= (NUMERIC_TYPE)(Q_total * dt);
        }
      }
    }
  }

  // cuda dg2 update_ghost_cells on one coefficient set
  void encodeGhost(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> Uset) {
    [encoder setComputePipelineState:dg2GhostPipeline];
    [encoder setBuffer:Uset offset:0 atIndex:0];
    [encoder setBuffer:DEM offset:0 atIndex:1];
    [encoder setBuffer:Zstar_x offset:0 atIndex:2];
    [encoder setBuffer:Zstar_y offset:0 atIndex:3];
    [encoder setBuffer:geomBuffer offset:0 atIndex:4];
    [encoder setBuffer:solverBuffer offset:0 atIndex:5];
    [encoder setBuffer:boundaryTypesBuffer offset:0 atIndex:6];
    [encoder setBuffer:boundaryValuesBuffer offset:0 atIndex:7];
    [encoder dispatchThreads:MTLSizeMake(std::max(geom.xsz, geom.ysz), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  }

  // cuda order: ghost cells from the current state, then the time step over
  // every element including the ghost cells (cuda DynamicTimestep)
  NUMERIC_TYPE computeTimestepDg2() {
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      encodeGhost(encoder, U[0]);
      if (states->adaptive_ts == ON) {
        float max_dt_f = (float)max_dt;
        memcpy([minDtBuffer contents], &max_dt_f, sizeof(uint32_t));
        [encoder setComputePipelineState:dg2DtPipeline];
        [encoder setBuffer:U[0] offset:0 atIndex:0];
        [encoder setBuffer:geomBuffer offset:0 atIndex:1];
        [encoder setBuffer:physicsBuffer offset:0 atIndex:2];
        [encoder setBuffer:solverBuffer offset:0 atIndex:3];
        float cfl_val = (float)solver->cfl;
        [encoder setBytes:&cfl_val length:sizeof(float) atIndex:4];
        [encoder setBytes:&max_dt_f length:sizeof(float) atIndex:5];
        [encoder setBuffer:minDtBuffer offset:0 atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(geom.pitch, geom.ysz + 2, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      }
      [encoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
    }
    if (states->adaptive_ts != ON) return max_dt;
    float min_dt;
    memcpy(&min_dt, [minDtBuffer contents], sizeof(float));
    return (NUMERIC_TYPE)min_dt;
  }

  NUMERIC_TYPE step() {
    // cuda/dg2 order: dynamic and uniform rain and point sources with the
    // previous step's dt (cuda::dt, zero before the first step), then the
    // ghost cells and the new dt from the updated state
    int src = 0;
    updateBoundaryValues();
    NUMERIC_TYPE rain_dt = is_first_step ? 0.0 : (NUMERIC_TYPE)solverParams.dt;
    is_first_step = false;
    // --- APPLY DYNAMIC RAIN ON GPU ---
    if (dynamic_rain && dynamic_rain->enabled() && applyRainPipeline) {
      dynamic_rain->update_time(time);

      NUMERIC_TYPE *rdata = dynamic_rain->data();
      size_t num_rain_cells =
          dynamic_rain->geometry().xsz * dynamic_rain->geometry().ysz;
      float *d_rain = (float *)[rainDataBuffer contents];
      bool changed = last_rain_time < 0;
      for (size_t k = 0; k < num_rain_cells && !changed; ++k)
        changed = d_rain[k] != (float)rdata[k];
      if (changed) {
        for (size_t k = 0; k < num_rain_cells; ++k)
          d_rain[k] = (float)rdata[k];
        const MetalGeometry &rg = *(const MetalGeometry *)[rainGeometryBuffer contents];
        double sum = 0.0;
        for (int j = 0; j < geom.ysz; ++j) {
          for (int i = 0; i < geom.xsz; ++i) {
            if (std::fabs(arrays->DEM[j * geom.xsz + i] - nodata) < 1e-6) continue;
            int ri = (int)std::floor((geom.blx - rg.blx + i * geom.dx) / rg.dx);
            int rj = (int)std::floor((j * geom.dy - (geom.tly - rg.tly)) / rg.dy);
            if (ri >= 0 && ri < rg.xsz && rj >= 0 && rj < rg.ysz)
              sum += std::max(0.0f, d_rain[rj * rg.xsz + ri]);
          }
        }
        current_rain_sum = sum;
        last_rain_time = time;

        [rainDataBuffer
            didModifyRange:NSMakeRange(0, num_rain_cells * sizeof(float))];
      }

      // Update mass balance (CPU side is much faster for global sums)
      double interval_rain_vol =
          current_rain_sum * (double)pars->dA * (double)rain_dt;
      boundCs->VolInMT += (NUMERIC_TYPE)interval_rain_vol;
      pars->RainTotalLoss += (NUMERIC_TYPE)interval_rain_vol;

      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:applyRainPipeline];
        [encoder setBuffer:H[src] offset:0 atIndex:0];
        [encoder setBuffer:DEM offset:0 atIndex:1];
        [encoder setBuffer:rainDataBuffer offset:0 atIndex:2];
        [encoder setBuffer:geomBuffer offset:0 atIndex:3];
        [encoder setBuffer:rainGeometryBuffer offset:0 atIndex:4];
        float dt_f = (float)rain_dt;
        [encoder setBytes:&dt_f length:sizeof(float) atIndex:5];
        float nd = gpu_nodata;
        [encoder setBytes:&nd length:sizeof(float) atIndex:6];

        [encoder dispatchThreads:MTLSizeMake(geom.xsz, geom.ysz, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];  // point sources write H next, on the CPU
      }
    }

    // LoadRain has already converted the time series from mm/hour to m/s.
    if (states->rainfall == ON && rain_dt > 0) {
      float rate = (float)InterpolateTimeSeries(arrays->rain, time);
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:applyUniformRainPipeline];
        [encoder setBuffer:H[src] offset:0 atIndex:0];
        [encoder setBuffer:DEM offset:0 atIndex:1];
        [encoder setBuffer:geomBuffer offset:0 atIndex:2];
        [encoder setBytes:&rate length:sizeof(float) atIndex:3];
        float dt = (float)rain_dt;
        [encoder setBytes:&dt length:sizeof(float) atIndex:4];
        [encoder setBytes:&gpu_nodata length:sizeof(float) atIndex:5];
        [encoder setBuffer:volOutBuffer offset:0 atIndex:6];
        [encoder dispatchThreads:MTLSizeMake(geom.xsz, geom.row_end - geom.row_begin, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
      }
    }

    applyPointSources();

#if defined(LISFLOOD_METAL_MPI)
    exchangeHalos(U[0]);
#endif

    NUMERIC_TYPE adaptive_dt = computeTimestepDg2();

#if defined(LISFLOOD_METAL_MPI)
    if (mpi_size > 1) {
        float dt_f = (float)adaptive_dt;
        MPI_Allreduce(MPI_IN_PLACE, &dt_f, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        adaptive_dt = (NUMERIC_TYPE)dt_f;
    }
#endif
    solverParams.dt = (float)adaptive_dt;
    solver->Tstep = adaptive_dt;
    memcpy([solverBuffer contents], &solverParams, sizeof(MetalSolverParams));

    // --- one DG2 step, in the order of cuda dg2 Solver::update_flow_variables
    // RK stage 1: friction, ghost, limit, sweep1x, sweep1y, zero_discharge(Uint)
    const int part_rows = geom.row_end - geom.row_begin;
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      MTLSize gridSize = MTLSizeMake(geom.xsz, part_rows, 1);
      MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
      const NSUInteger N = num_cells;
      memset([maxBits contents], 0, sizeof(uint32_t));

      auto limit = [&](id<MTLBuffer> Uset) {
        if (!limit_slopes) return;
        [encoder setComputePipelineState:dg2CopySlopesPipeline];
        [encoder setBuffer:Uset offset:0 atIndex:0];
        [encoder setBuffer:DEM offset:0 atIndex:1];
        [encoder setBuffer:slopeCopy offset:0 atIndex:2];
        [encoder setBuffer:geomBuffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder setComputePipelineState:dg2LimitPipeline];
        [encoder setBuffer:Uset offset:0 atIndex:0];
        [encoder setBuffer:DEM offset:0 atIndex:1];
        [encoder setBuffer:slopeCopy offset:0 atIndex:2];
        [encoder setBuffer:geomBuffer offset:0 atIndex:3];
        struct { float kt; float unused; } dp = {krivodonova_thresh, 0.0f};
        [encoder setBytes:&dp length:sizeof(dp) atIndex:4];
        [encoder setBuffer:maxBits offset:0 atIndex:5];
        [encoder setBuffer:limited offset:0 atIndex:6];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder setComputePipelineState:dg2ApplyLimitedPipeline];
        [encoder setBuffer:Uset offset:0 atIndex:0];
        [encoder setBuffer:limited offset:0 atIndex:1];
        [encoder setBuffer:geomBuffer offset:0 atIndex:2];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      };
      auto zero_perimeter = [&](id<MTLBuffer> Uset) {
        [encoder setComputePipelineState:dg2ZeroPerimeterPipeline];
        [encoder setBuffer:Uset offset:0 atIndex:0];
        [encoder setBuffer:geomBuffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(std::max(geom.xsz, geom.ysz), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      };
      auto zero_discharge = [&](id<MTLBuffer> Uset) {
        [encoder setComputePipelineState:dg2ZeroDischargePipeline];
        [encoder setBuffer:Uset offset:0 atIndex:0];
        [encoder setBuffer:geomBuffer offset:0 atIndex:1];
        [encoder setBuffer:solverBuffer offset:0 atIndex:2];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      };
      // one sweep: L from In; Out = combine(mode, A, B, L)
      auto sweep = [&](bool x, id<MTLBuffer> In, id<MTLBuffer> A, id<MTLBuffer> B,
                       id<MTLBuffer> Out, int mode) {
        [encoder setComputePipelineState:(x ? dg2FlowXPipeline : dg2FlowYPipeline)];
        [encoder setBuffer:In offset:0 atIndex:0];
        [encoder setBuffer:A offset:0 atIndex:1];
        [encoder setBuffer:B offset:0 atIndex:2];
        [encoder setBuffer:Out offset:0 atIndex:3];
        [encoder setBuffer:DEM offset:0 atIndex:4];
        [encoder setBuffer:(x ? Zstar_x : Zstar_y) offset:0 atIndex:5];
        [encoder setBuffer:geomBuffer offset:0 atIndex:6];
        [encoder setBuffer:physicsBuffer offset:0 atIndex:7];
        [encoder setBuffer:solverBuffer offset:0 atIndex:8];
        [encoder setBuffer:boundaryTypesBuffer offset:0 atIndex:9];
        [encoder setBuffer:volOutBuffer offset:0 atIndex:10];
        struct { int mode; float mass_weight; } stage = {mode, 0.5f};
        [encoder setBytes:&stage length:sizeof(stage) atIndex:11];
        MTLSize tg = MTLSizeMake(kTileX, kTileY, 1);
        MTLSize groups = x
            ? MTLSizeMake((geom.xsz + 1 + (kTileX - 1) - 1) / (kTileX - 1),
                          ((NSUInteger)part_rows + kTileY - 1) / kTileY, 1)
            : MTLSizeMake((geom.xsz + kTileX - 1) / kTileX,
                          ((NSUInteger)part_rows + 1 + (kTileY - 1) - 1) / (kTileY - 1), 1);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:tg];
      };

      id<MTLBuffer> Uold = U[0], Uint = U[1], Ux = U[2];
      if (friction_on) {
        [encoder setComputePipelineState:dg2FrictionPipeline];
        [encoder setBuffer:Uold offset:0 atIndex:0];
        [encoder setBuffer:manning offset:0 atIndex:1];
        [encoder setBuffer:geomBuffer offset:0 atIndex:2];
        [encoder setBuffer:physicsBuffer offset:0 atIndex:3];
        [encoder setBuffer:solverBuffer offset:0 atIndex:4];
        int hm = has_manning_map ? 1 : 0;
        [encoder setBytes:&hm length:sizeof(int) atIndex:5];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      }
      if (limit_slopes) {
        [encoder setComputePipelineState:dg2MaxHPipeline];
        [encoder setBuffer:Uold offset:0 atIndex:0];
        [encoder setBuffer:geomBuffer offset:0 atIndex:1];
        [encoder setBuffer:maxBits offset:0 atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      encodeGhost(encoder, Uold);
      limit(Uold);
      zero_perimeter(Uold);
      sweep(true, Uold, Uold, Uold, Ux, 0);    // rk_stage1_x
      sweep(false, Uold, Ux, Ux, Uint, 0);     // rk_stage1_y
      zero_discharge(Uint);
      [encoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
    }

#if defined(LISFLOOD_METAL_MPI)
    exchangeHalos(U[1]);
#endif

    // RK stage 2: ghost, limit, sweep2x, sweep2y, drain, losses, max
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      MTLSize gridSize = MTLSizeMake(geom.xsz, part_rows, 1);
      MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
      const NSUInteger N = num_cells;

      auto limit = [&](id<MTLBuffer> Uset) {
        if (!limit_slopes) return;
        [encoder setComputePipelineState:dg2CopySlopesPipeline];
        [encoder setBuffer:Uset offset:0 atIndex:0];
        [encoder setBuffer:DEM offset:0 atIndex:1];
        [encoder setBuffer:slopeCopy offset:0 atIndex:2];
        [encoder setBuffer:geomBuffer offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(N, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder setComputePipelineState:dg2LimitPipeline];
        [encoder setBuffer:Uset offset:0 atIndex:0];
        [encoder setBuffer:DEM offset:0 atIndex:1];
        [encoder setBuffer:slopeCopy offset:0 atIndex:2];
        [encoder setBuffer:geomBuffer offset:0 atIndex:3];
        struct { float kt; float unused; } dp = {krivodonova_thresh, 0.0f};
        [encoder setBytes:&dp length:sizeof(dp) atIndex:4];
        [encoder setBuffer:maxBits offset:0 atIndex:5];
        [encoder setBuffer:limited offset:0 atIndex:6];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder setComputePipelineState:dg2ApplyLimitedPipeline];
        [encoder setBuffer:Uset offset:0 atIndex:0];
        [encoder setBuffer:limited offset:0 atIndex:1];
        [encoder setBuffer:geomBuffer offset:0 atIndex:2];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      };
      auto zero_perimeter = [&](id<MTLBuffer> Uset) {
        [encoder setComputePipelineState:dg2ZeroPerimeterPipeline];
        [encoder setBuffer:Uset offset:0 atIndex:0];
        [encoder setBuffer:geomBuffer offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(std::max(geom.xsz, geom.ysz), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      };
      auto zero_discharge = [&](id<MTLBuffer> Uset) {
        [encoder setComputePipelineState:dg2ZeroDischargePipeline];
        [encoder setBuffer:Uset offset:0 atIndex:0];
        [encoder setBuffer:geomBuffer offset:0 atIndex:1];
        [encoder setBuffer:solverBuffer offset:0 atIndex:2];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      };
      auto sweep = [&](bool x, id<MTLBuffer> In, id<MTLBuffer> A, id<MTLBuffer> B,
                       id<MTLBuffer> Out, int mode) {
        [encoder setComputePipelineState:(x ? dg2FlowXPipeline : dg2FlowYPipeline)];
        [encoder setBuffer:In offset:0 atIndex:0];
        [encoder setBuffer:A offset:0 atIndex:1];
        [encoder setBuffer:B offset:0 atIndex:2];
        [encoder setBuffer:Out offset:0 atIndex:3];
        [encoder setBuffer:DEM offset:0 atIndex:4];
        [encoder setBuffer:(x ? Zstar_x : Zstar_y) offset:0 atIndex:5];
        [encoder setBuffer:geomBuffer offset:0 atIndex:6];
        [encoder setBuffer:physicsBuffer offset:0 atIndex:7];
        [encoder setBuffer:solverBuffer offset:0 atIndex:8];
        [encoder setBuffer:boundaryTypesBuffer offset:0 atIndex:9];
        [encoder setBuffer:volOutBuffer offset:0 atIndex:10];
        struct { int mode; float mass_weight; } stage = {mode, 0.5f};
        [encoder setBytes:&stage length:sizeof(stage) atIndex:11];
        MTLSize tg = MTLSizeMake(kTileX, kTileY, 1);
        MTLSize groups = x
            ? MTLSizeMake((geom.xsz + 1 + (kTileX - 1) - 1) / (kTileX - 1),
                          ((NSUInteger)part_rows + kTileY - 1) / kTileY, 1)
            : MTLSizeMake((geom.xsz + kTileX - 1) / kTileX,
                          ((NSUInteger)part_rows + 1 + (kTileY - 1) - 1) / (kTileY - 1), 1);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:tg];
      };

      id<MTLBuffer> Uold = U[0], Uint = U[1], Ux = U[2];
      encodeGhost(encoder, Uint);
      limit(Uint);
      zero_perimeter(Uint);
      sweep(true, Uint, Uold, Uint, Ux, 1);    // rk_stage2_x
      sweep(false, Uint, Ux, Ux, Uold, 2);     // rk_stage2_y
      zero_discharge(Uold);

      if (pars->drain_nodata == ON) {
        [encoder setComputePipelineState:dg2DrainPipeline];
        [encoder setBuffer:Uold offset:0 atIndex:0];
        [encoder setBuffer:DEM offset:0 atIndex:1];
        [encoder setBuffer:geomBuffer offset:0 atIndex:2];
        float nd = gpu_nodata;
        [encoder setBytes:&nd length:sizeof(float) atIndex:3];
        [encoder setBuffer:volOutBuffer offset:0 atIndex:4];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      }

      // floodplain infiltration and evaporation (8.4; as cuda, after the step)
      losses.encode(encoder, Uold, nil, nil, false, 1, (int)N, 1.7320508075688772f,
                    DEM, gpu_nodata, solverParams.depth_thresh, adaptive_dt,
                    time + adaptive_dt);

      // running maximum depth (once per step, as the other solvers)
      [encoder setComputePipelineState:updateMaxFieldPipeline];
      [encoder setBuffer:Uold offset:0 atIndex:0];
      [encoder setBuffer:maxH offset:0 atIndex:1];
      [encoder setBuffer:geomBuffer offset:0 atIndex:2];
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

      [encoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
    }

    losses.collect();
    time += (NUMERIC_TYPE)adaptive_dt;
    return (NUMERIC_TYPE)adaptive_dt;
  }
};

Simulator::Simulator(Fnames &filenames, Pars &pars, ::Solver &solver,
                     Arrays &arrays, States &states, BoundCs &boundCs,
                     int verbose, int rank, int size, int r_begin, int r_end) {
  impl = new SimulatorImpl();
  impl->arrays = &arrays;
  impl->boundCs = &boundCs;
  impl->pars = &pars;
  impl->solver = &solver;
  impl->states = &states;

  if (!impl->init(filenames, pars, solver, verbose, rank, size, r_begin, r_end)) {
    delete impl;
    impl = nullptr;
  }
}

Simulator::~Simulator() {
  if (impl)
    delete impl;
}

void Simulator::initialize(NUMERIC_TYPE *H, NUMERIC_TYPE *HU, NUMERIC_TYPE *HV,
                           NUMERIC_TYPE *DEM, NUMERIC_TYPE *Mann) {
  if (impl) {
    impl->copyToGPU(H, HU, HV, DEM, Mann);
  }
}

NUMERIC_TYPE Simulator::step() {
  if (impl) {
    return impl->step();
  }
  return 0;
}

bool Simulator::is_available() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device != nil;
}

void Simulator::copy_to_host(NUMERIC_TYPE *H, NUMERIC_TYPE *HU,
                             NUMERIC_TYPE *HV) {
  if (impl) {
    impl->copyFromGPU(H, HU, HV);
  }
}

void Simulator::copy_max_to_host(NUMERIC_TYPE *maxH) {
  if (impl) {
    impl->copyMaxFromGPU(maxH);
  }
}

void Simulator::gather_rows() {
  if (impl) {
    impl->gatherRows(impl->U[0]);
  }
}

int Simulator::get_rank() const {
  return impl ? impl->mpi_rank : 0;
}

int Simulator::get_size() const {
  return impl ? impl->mpi_size : 1;
}

void simulate(Fnames &filenames, Files *files, States &states, Pars &pars,
              ::Solver &solver, BoundCs &boundCs, Stage &stage, Arrays &arrays,
              int verbose) {
  if (states.rainfallmask == ON || states.routing == ON) {
    fprintf(stderr, "ERROR: Metal DG2 does not support rainfallmask or rainfallrouting; "
                    "use unmasked rainfall or dynamicrainfile.\n");
    std::exit(EXIT_FAILURE);
  }
  if (states.rainfall == ON && !arrays.rain) {
    fprintf(stderr, "ERROR: Metal DG2 requires a loaded rainfall time series.\n");
    std::exit(EXIT_FAILURE);
  }

#if defined(LISFLOOD_METAL_MPI)
  int initialized = 0;
  MPI_Initialized(&initialized);
  const bool initialized_here = !initialized;
  if (initialized_here) MPI_Init(nullptr, nullptr);
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if (mpi_size != states.cuda_mpi_gpus) {
    fprintf(stderr,
            "ERROR: parameter GPUs %d requires launching exactly %d MPI ranks; "
            "got %d.\n",
            states.cuda_mpi_gpus, states.cuda_mpi_gpus, mpi_size);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  const int mpi_row_begin = 1 + (pars.ysz * mpi_rank) / mpi_size;
  const int mpi_row_end   = 1 + (pars.ysz * (mpi_rank + 1)) / mpi_size;
  if (mpi_row_end <= mpi_row_begin)
    throw std::runtime_error("More MPI ranks than DG2 rows");
  printf("Metal DG2 MPI rank %d/%d owns rows [%d,%d)\n",
         mpi_rank, mpi_size, mpi_row_begin, mpi_row_end);
  fflush(stdout);
#else
  const int mpi_rank      = 0;
  const int mpi_size      = 1;
  const int mpi_row_begin = 1;
  const int mpi_row_end   = pars.ysz + 1;
#endif

  Simulator sim(filenames, pars, solver, arrays, states, boundCs, verbose,
                mpi_rank, mpi_size, mpi_row_begin, mpi_row_end);

  if (!sim.is_valid()) {
    printf("ERROR: Metal GPU initialization failed! Check that:\n");
    printf("  1. Metal shader library (metal_dg2_kernels.metallib) exists\n");
    printf("  2. Metal GPU is available on this system\n");
    printf("  3. Single precision mode is enabled (NUMERIC_TYPE=float)\n");
#if defined(LISFLOOD_METAL_MPI)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    return;
  }

  sim.initialize(arrays.H, arrays.HU, arrays.HV, arrays.DEM, arrays.Manningsn);

  if (verbose && mpi_rank == 0) {
    printf("Metal DG2 GPU initialized successfully.\n");
    printf("Grid: %d x %d cells, %d rank(s)\n", pars.xsz, pars.ysz, mpi_size);
    printf("SaveInt: %.2f, MassInt: %.2f\n", pars.SaveInt, pars.MassInt);
    printf("Simulation time: %.2f seconds\n", solver.Sim_Time);
    printf("\nStarting GPU simulation...\n\n");
    fflush(stdout);
  }

  ::fv1::initialise_Zstar(&pars, &arrays);

  time_t loop_start;
  time(&loop_start);

  solver.MinTstep = solver.InitTstep;
  solver.vol1 = DomainVol(&states, &pars, nullptr, &arrays, nullptr);

  while (solver.t < solver.Sim_Time) {
    zero_flux_stats(&boundCs);

    NUMERIC_TYPE dt = sim.step();

    if (dt <= 0) {
      if (verbose && mpi_rank == 0)
        printf("\nERROR: Metal DG2 simulator returned zero timestep.\n");
      break;
    }

    if (solver.t > 0)
      solver.MinTstep = std::min(solver.MinTstep, dt);
    const NUMERIC_TYPE t_start = solver.t;
    solver.t += dt;
    solver.itCount += 1;

    if (verbose && mpi_rank == 0) printf("t=%f\tdt=%f\n", t_start, dt);

    // Mass balance output
    if (solver.t >= pars.MassTotal) {
      sim.gather_rows();
      if (mpi_rank == 0) {
        sim.copy_to_host(arrays.H, arrays.HU, arrays.HV);
        ::fv1::update_Hstar(&pars, &arrays);
        ::fv1::update_fluxes(&pars, &solver, &arrays);
        ::fv1::update_fluxes_on_boundaries(&pars, &solver, &boundCs, &arrays);

        if (states.gsection == ON && files->gau_fp) {
          fprintf(files->gau_fp, "%12.2" NUM_FMT "", solver.t);
          for (int i = 0; i < stage.Ngauges; i++) {
            NUMERIC_TYPE unit_discharge =
                CalcVirtualGauge(i, pars.xsz + 1, arrays.FHx, arrays.FHy, &stage);
            NUMERIC_TYPE discharge = 0.0;
            if (stage.gauge_dir[i] == North || stage.gauge_dir[i] == South)
              discharge = unit_discharge * pars.dx;
            else
              discharge = unit_discharge * pars.dy;
            fprintf(files->gau_fp, " %10.3" NUM_FMT "", discharge);
          }
          fprintf(files->gau_fp, "\n");
          fflush(files->gau_fp);
        }

        update_mass_stats(&states, &pars, &solver, &boundCs, &arrays);
        write_mass_stats(files, &pars, &solver, &boundCs);
      }
      pars.MassTotal += pars.MassInt;
    }

    // Solution output
    if (solver.t >= pars.SaveTotal) {
      sim.gather_rows();
      if (mpi_rank == 0) {
        sim.copy_to_host(arrays.H, arrays.HU, arrays.HV);
        if (states.voutput == ON) update_velocity(&pars, &solver, &arrays);
        write_solution(&filenames, &states, &pars, &solver, &arrays);
        pars.SaveNo += 1;
      }
      pars.SaveTotal += pars.SaveInt;
    }
  }

  // Final output
  sim.gather_rows();
  if (mpi_rank == 0) {
    sim.copy_to_host(arrays.H, arrays.HU, arrays.HV);
    if (states.voutput == ON) update_velocity(&pars, &solver, &arrays);
    write_solution(&filenames, &states, &pars, &solver, &arrays);
    sim.copy_max_to_host(arrays.maxH);
    write_max_field(&filenames, &states, &pars, &arrays);
  }

  time_t loop_end;
  time(&loop_end);
  double elapsed = difftime(loop_end, loop_start);

  if (verbose && mpi_rank == 0) {
    printf("\nloop time %lf\n", elapsed);
    printf("Finished.\n\n");
  }

#if defined(LISFLOOD_METAL_MPI)
  if (initialized_here) MPI_Finalize();
#endif
}

} // namespace dg2
} // namespace metal
} // namespace lis

#endif // __APPLE__
