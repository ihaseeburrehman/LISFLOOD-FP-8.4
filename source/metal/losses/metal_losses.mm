// LISFLOOD-FP Metal: floodplain infiltration and evaporation (8.4)
#ifdef __APPLE__
#include "metal_losses.h"
#import <Foundation/Foundation.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lis {
namespace metal {

namespace {

struct LossParams {
  int32_t greenampt, has_rate_grid, has_fmax, has_q, mode;
  int32_t xsz, ysz, pitch, i0, j0, plane_n, ngx;
  float uniform_rate, evap_rate, dt, depth_thresh, nodata, face;
};

bool parse_number(const char *spec, double &value) {
  char *end = nullptr;
  value = strtod(spec, &end);
  if (end == spec) return false;
  while (*end == ' ' || *end == '\t') end++;
  return *end == '\0';
}

// A number for every cell, or an ESRI ASCII grid of the DEM's size (NoData
// and negative values: 0); values divided by divisor (as cuda::Losses)
bool read_parameter(const char *spec, int xsz, int ysz, double divisor,
                    const char *keyword, std::vector<float> &out) {
  out.assign((size_t)xsz * ysz, 0.0f);
  double number;
  if (parse_number(spec, number)) {
    float v = number > 0.0 ? (float)(number / divisor) : 0.0f;
    std::fill(out.begin(), out.end(), v);
    return true;
  }
  FILE *f = fopen(spec, "r");
  if (!f) {
    fprintf(stderr, "ERROR: %s: cannot open %s\n", keyword, spec);
    return false;
  }
  char key[64];
  double value, no_data = -9999.0;
  int ncols = -1, nrows = -1;
  for (int k = 0; k < 6; k++) {
    long pos = ftell(f);
    if (fscanf(f, "%63s", key) != 1) break;
    if (!isalpha((unsigned char)key[0])) { fseek(f, pos, SEEK_SET); break; }
    if (fscanf(f, "%lf", &value) != 1) break;
    if (!strcasecmp(key, "ncols")) ncols = (int)value;
    if (!strcasecmp(key, "nrows")) nrows = (int)value;
    if (!strcasecmp(key, "NODATA_value")) no_data = value;
  }
  if (ncols != xsz || nrows != ysz) {
    fprintf(stderr, "ERROR: %s: %s is %d x %d cells, the DEM %d x %d\n",
            keyword, spec, ncols, nrows, xsz, ysz);
    fclose(f);
    return false;
  }
  for (size_t k = 0; k < out.size(); k++) {
    if (fscanf(f, "%lf", &value) != 1) {
      fprintf(stderr, "ERROR: %s: %s ends early\n", keyword, spec);
      fclose(f);
      return false;
    }
    if (value == no_data || value <= 0.0) value = 0.0;
    out[k] = (float)(value / divisor);
  }
  fclose(f);
  return true;
}

} // namespace

bool Losses::setup(id<MTLDevice> device, Fnames &filenames, States &states,
                   Pars &pars, Arrays *arrays, int xsz_, int ysz_, int pitch_,
                   int i0_, int j0_, size_t n, int verbose) {
  infiltration = states.calc_infiltration == ON;
  greenampt = states.calc_greenampt == ON;
  horton = states.calc_horton == ON;
  evaporation = states.calc_evap == ON;
  if (!infiltration && !greenampt && !horton && !evaporation) return false;
  xsz = xsz_; ysz = ysz_; pitch = pitch_; i0 = i0_; j0 = j0_;
  dA = (double)pars.dx * (double)pars.dy;

  // kernels: metal_losses_kernels.metallib in the current directory or next
  // to the executable
  NSString *name = @"metal_losses_kernels.metallib";
  NSString *path = nil;
  if ([[NSFileManager defaultManager] fileExistsAtPath:name]) path = name;
  if (!path) {
    NSString *exe = [[NSProcessInfo processInfo] arguments][0];
    NSString *p = [[exe stringByDeletingLastPathComponent]
        stringByAppendingPathComponent:name];
    if ([[NSFileManager defaultManager] fileExistsAtPath:p]) path = p;
  }
  NSError *error = nil;
  id<MTLLibrary> library =
      path ? [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error] : nil;
  id<MTLFunction> fn = [library newFunctionWithName:@"apply_losses"];
  pipeline = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
  if (!pipeline) {
    fprintf(stderr, "ERROR: Metal: cannot load %s (infiltration and evaporation)\n",
            [name UTF8String]);
    std::exit(EXIT_FAILURE);
  }

  MTLResourceOptions opt = MTLResourceStorageModeShared;
  auto zeros = [&]() {
    id<MTLBuffer> b = [device newBufferWithLength:n * sizeof(float) options:opt];
    memset([b contents], 0, n * sizeof(float));
    return b;
  };
  auto upload = [&](const std::vector<float> &v) {
    id<MTLBuffer> b = zeros();
    float *d = (float *)[b contents];
    for (int j = 0; j < ysz; j++)
      for (int i = 0; i < xsz; i++)
        d[(size_t)(j + j0) * pitch + i + i0] = v[(size_t)j * xsz + i];
    return b;
  };
  auto fail = [&]() { std::exit(EXIT_FAILURE); };

  std::vector<float> g;
  if (infiltration) {
    if (strlen(filenames.infilfilename) > 0) {
      if (!read_parameter(filenames.infilfilename, xsz, ysz, 3600000.0, "infilfile", g)) fail();
      rate = upload(g);
      has_rate_grid = true;  // mm/h -> m/s
      if (verbose) printf("Infiltration: per-cell rates from %s\n", filenames.infilfilename);
    } else {
      uniform_rate = (float)pars.InfilRate;
      if (verbose) printf("Infiltration: uniform rate %.10f m/s\n", (double)uniform_rate);
    }
  }
  if (greenampt) {
    const char *missing = nullptr;
    if (!strlen(filenames.greenampt_ks)) missing = "greenampt_ks";
    else if (!strlen(filenames.greenampt_psi)) missing = "greenampt_psi";
    else if (!strlen(filenames.greenampt_porosity)) missing = "greenampt_porosity";
    else if (!strlen(filenames.greenampt_theta_init)) missing = "greenampt_theta_init";
    if (missing) {
      fprintf(stderr, "ERROR: infiltration greenampt needs %s (a number or a grid file)\n",
              missing);
      fail();
    }
    std::vector<float> k, s, por, th, dep;
    if (!read_parameter(filenames.greenampt_ks, xsz, ysz, 3600000.0, "greenampt_ks", k) ||
        !read_parameter(filenames.greenampt_psi, xsz, ysz, 1000.0, "greenampt_psi", s) ||
        !read_parameter(filenames.greenampt_porosity, xsz, ysz, 1.0, "greenampt_porosity", por) ||
        !read_parameter(filenames.greenampt_theta_init, xsz, ysz, 1.0, "greenampt_theta_init", th))
      fail();
    bool depth = strlen(filenames.greenampt_soil_depth) > 0;
    if (depth && !read_parameter(filenames.greenampt_soil_depth, xsz, ysz, 1.0,
                                 "greenampt_soil_depth", dep))
      fail();
    long wetter = 0;
    for (size_t c = 0; c < por.size(); c++) {
      float d = por[c] - th[c];
      if (d < 0.0f) { d = 0.0f; wetter++; }
      por[c] = d;
      if (depth) dep[c] *= d;
    }
    if (wetter > 0)
      fprintf(stderr, "WARNING: greenampt_theta_init > greenampt_porosity in %ld cells; "
                      "there the soil is taken as saturated\n", wetter);
    ks = upload(k); psi = upload(s); dtheta = upload(por);
    if (depth) { fmax = upload(dep); has_fmax = true; }
    Fhi = zeros(); Flo = zeros();
    if (verbose) printf("Infiltration: Green-Ampt%s\n", depth ? ", with soil depth" : "");
  }
  if (horton) {
    const char *missing = nullptr;
    if (!strlen(filenames.horton_f0)) missing = "horton_f0";
    else if (!strlen(filenames.horton_fc)) missing = "horton_fc";
    else if (!strlen(filenames.horton_k)) missing = "horton_k";
    if (missing) {
      fprintf(stderr, "ERROR: infiltration horton needs %s (a number or a grid file)\n",
              missing);
      fail();
    }
    std::vector<float> f0, fc, k;
    if (!read_parameter(filenames.horton_f0, xsz, ysz, 3600000.0, "horton_f0", f0) ||
        !read_parameter(filenames.horton_fc, xsz, ysz, 3600000.0, "horton_fc", fc) ||
        !read_parameter(filenames.horton_k, xsz, ysz, 3600.0, "horton_k", k))
      fail();  // mm/h -> m/s, 1/h -> 1/s
    long inverted = 0;
    for (size_t c = 0; c < f0.size(); c++)
      if (fc[c] > f0[c]) { f0[c] = fc[c]; inverted++; }
    if (inverted > 0)
      fprintf(stderr, "WARNING: horton_fc > horton_f0 in %ld cells; "
                      "there the capacity is constant at horton_fc\n", inverted);
    // the Green-Ampt buffers hold the Horton parameters (see the kernel)
    ks = upload(fc); psi = upload(f0); dtheta = upload(k);
    Fhi = zeros(); Flo = zeros();
    if (verbose) printf("Infiltration: Horton\n");
  }
  if (evaporation) {
    evap = arrays ? arrays->evap : nullptr;
    if (!evap) {
      Arrays a;
      a.evap = nullptr;
      LoadEvap(&filenames, &a, verbose);
      evap = a.evap;
    }
  }
  residual = zeros();
  // unused parameters still need a buffer to bind
  id<MTLBuffer> dummy = [device newBufferWithLength:sizeof(float) options:opt];
  if (!rate) rate = dummy;
  if (!Fhi) { Fhi = dummy; Flo = dummy; ks = dummy; psi = dummy; dtheta = dummy; }
  if (!fmax) fmax = dummy;
  ngx = (xsz + 15) / 16;
  ngy = (ysz + 15) / 16;
  partial = [device newBufferWithLength:(size_t)ngx * ngy * sizeof(float) options:opt];
  memset([partial contents], 0, (size_t)ngx * ngy * sizeof(float));
  on = true;
  return true;
}

void Losses::encode(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> U,
                    id<MTLBuffer> HU, id<MTLBuffer> HV, bool has_q, int mode,
                    int plane_n, float face, id<MTLBuffer> DEM, float nodata,
                    float depth_thresh, double dt, double t) {
  if (!on) return;
  LossParams p;
  p.greenampt = greenampt ? 1 : (horton ? 2 : 0);
  p.has_rate_grid = has_rate_grid ? 1 : 0;
  p.has_fmax = has_fmax ? 1 : 0;
  p.has_q = has_q ? 1 : 0;
  p.mode = mode;
  p.xsz = xsz; p.ysz = ysz; p.pitch = pitch; p.i0 = i0; p.j0 = j0;
  p.plane_n = plane_n;
  p.ngx = ngx;
  p.uniform_rate = infiltration ? uniform_rate : 0.0f;
  p.evap_rate = (evaporation && evap) ? (float)InterpolateTimeSeries(evap, (NUMERIC_TYPE)t) : 0.0f;
  p.dt = (float)dt;
  p.depth_thresh = depth_thresh;
  p.nodata = nodata;
  p.face = face;
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:U offset:0 atIndex:0];
  [enc setBuffer:(HU ? HU : U) offset:0 atIndex:1];
  [enc setBuffer:(HV ? HV : U) offset:0 atIndex:2];
  [enc setBuffer:DEM offset:0 atIndex:3];
  [enc setBytes:&p length:sizeof(p) atIndex:4];
  [enc setBuffer:rate offset:0 atIndex:5];
  [enc setBuffer:residual offset:0 atIndex:6];
  [enc setBuffer:Fhi offset:0 atIndex:7];
  [enc setBuffer:Flo offset:0 atIndex:8];
  [enc setBuffer:ks offset:0 atIndex:9];
  [enc setBuffer:psi offset:0 atIndex:10];
  [enc setBuffer:dtheta offset:0 atIndex:11];
  [enc setBuffer:fmax offset:0 atIndex:12];
  [enc setBuffer:partial offset:0 atIndex:13];
  [enc dispatchThreadgroups:MTLSizeMake(ngx, ngy, 1)
      threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

void Losses::collect() {
  if (!on) return;
  const float *v = (const float *)[partial contents];
  double sum = 0.0;
  for (int k = 0; k < ngx * ngy; k++) sum += (double)v[k];
  interval += sum * dA;
}

double Losses::take_interval_volume() {
  double v = interval;
  interval = 0.0;
  return v;
}

} // namespace metal
} // namespace lis
#endif
