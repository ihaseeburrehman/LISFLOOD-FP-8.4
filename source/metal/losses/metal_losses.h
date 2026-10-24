// LISFLOOD-FP Metal: floodplain infiltration and evaporation (8.4)
// Shared by the Metal FV1, FV2, DG2 and ACC solvers; the rules of
// cuda/cuda_losses (see metal_losses_kernels.metal). Objective-C++ only.
#pragma once
#ifdef __APPLE__
#include "../../lisflood.h"
#import <Metal/Metal.h>

namespace lis {
namespace metal {

class Losses {
public:
  /**
   * Reads the parameters (numbers or ESRI ASCII grids, as cuda::Losses) for
   * a raster of xsz x ysz cells stored with row pitch `pitch`, cell (i, j) at
   * (j + j0) * pitch + i + i0 (j = 0 is the north row), n_elements floats.
   * Returns false if no loss is asked for or the kernels cannot be loaded.
   */
  bool setup(id<MTLDevice> device, Fnames &filenames, States &states,
             Pars &pars, Arrays *arrays, int xsz, int ysz, int pitch, int i0,
             int j0, size_t n_elements, int verbose);

  bool enabled() const { return on; }

  /**
   * Encodes this step's losses. mode 0: U is the depth, HU/HV the discharges
   * (scaled with the depth when has_q); mode 1: U holds the nine DG2
   * coefficient planes of plane_n floats each (face: sqrt 3). t is the time
   * at which the evaporation rate is read. Call collect() once the command
   * buffer has completed.
   */
  void encode(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> U,
              id<MTLBuffer> HU, id<MTLBuffer> HV, bool has_q, int mode,
              int plane_n, float face, id<MTLBuffer> DEM, float nodata,
              float depth_thresh, double dt, double t);

  /** adds the volume removed by the last encoded step (m3) */
  void collect();

  /** volume removed since the previous call (m3) */
  double take_interval_volume();

  // Adaptive MWDG2 applies its per-raster nonlinear losses and then restricts
  // them to active leaves. These buffers preserve soil memory across mesh
  // regridding and provide the CUDA-style rounding compensation state.
  id<MTLBuffer> residual_buffer() const { return residual; }
  id<MTLBuffer> rate_buffer() const { return rate; }
  id<MTLBuffer> ks_buffer() const { return ks; }
  id<MTLBuffer> psi_buffer() const { return psi; }
  id<MTLBuffer> dtheta_buffer() const { return dtheta; }
  id<MTLBuffer> fmax_buffer() const { return fmax; }
  bool uses_rate_grid() const { return has_rate_grid; }
  bool uses_greenampt() const { return greenampt; }
  /** Green-Ampt or Horton: the ks/psi/dtheta buffers hold soil parameters */
  bool uses_soil_params() const { return greenampt || horton; }
  bool uses_soil_depth() const { return has_fmax; }

private:
  bool on = false;
  bool greenampt = false, horton = false, infiltration = false, evaporation = false;
  bool has_rate_grid = false, has_fmax = false;
  float uniform_rate = 0.0f;
  int xsz = 0, ysz = 0, pitch = 0, i0 = 0, j0 = 0, ngx = 0, ngy = 0;
  double dA = 0.0, interval = 0.0;
  TimeSeries *evap = nullptr;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> rate = nil, residual = nil, Fhi = nil, Flo = nil, ks = nil,
                psi = nil, dtheta = nil, fmax = nil, partial = nil;
};

} // namespace metal
} // namespace lis
#endif
