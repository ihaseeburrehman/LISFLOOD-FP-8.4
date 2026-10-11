#pragma once
// Boundary conditions and point sources of the adaptive solvers (MWDG2,
// HWFV1), read and updated as cuda/adaptive does it (classes/Boundary.h,
// classes/PointSources.h): one segment per side (the first bcifile line for
// that side), cell range int((lower-origin)/dx) .. int((upper-origin)/dx)-1,
// cell index y counted from the south edge, and time series interpolated with
// the same row stepping.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace lis { namespace metal { namespace mwdg2 {

enum AdaptiveBcType { BC_CLOSED = 0, BC_FREE = 1, BC_HFIX = 2, BC_HVAR = 3, BC_QFIX = 4, BC_QVAR = 5 };
enum AdaptiveSideIndex { SIDE_N = 0, SIDE_E = 1, SIDE_S = 2, SIDE_W = 3 };

struct AdaptiveSeries {
  std::vector<float> time, value;
  int row = 0;
  // Boundary::update_inlet / PointSources::update_src
  float at(float now) {
    const int n = static_cast<int>(value.size());
    if (n == 0) return 0.0f;
    if (row - 1 < n && row + 1 < n) {
      float t1 = time[row], t2 = time[row + 1];
      if (now > t2 && row + 2 < n) { ++row; t1 = time[row]; t2 = time[row + 1]; }
      return value[row] + (value[row + 1] - value[row]) / (t2 - t1) * (now - t1);
    }
    return value[n - 1];
  }
};

struct AdaptiveSide {
  int type = BC_CLOSED;
  int start = 0, end = -1;    // cell indices along the side
  float inlet = 0.0f;
  AdaptiveSeries series;
  bool contains(int cell) const { return type != BC_CLOSED && cell >= start && cell <= end; }
};

struct AdaptivePointSource {
  int x = 0, y = 0;           // finest cell, y from the south edge
  int type = BC_HFIX;
  float value = 0.0f;
  AdaptiveSeries series;
};

inline int adaptiveBcType(const char *name) {
  if (!std::strncmp(name, "FREE", 4)) return BC_FREE;
  if (!std::strncmp(name, "HFIX", 4)) return BC_HFIX;
  if (!std::strncmp(name, "HVAR", 4)) return BC_HVAR;
  if (!std::strncmp(name, "QFIX", 4)) return BC_QFIX;
  if (!std::strncmp(name, "QVAR", 4)) return BC_QVAR;
  return BC_CLOSED;
}

// The named series in bdyfile: "name" line, then "<rows> <seconds|minutes|hours>",
// then rows of "value time".
inline AdaptiveSeries readAdaptiveSeries(const char *bdyfile, const char *name) {
  if (!bdyfile || !bdyfile[0])
    throw std::runtime_error(std::string("time series ") + name + " needs a bdyfile");
  FILE *fp = std::fopen(bdyfile, "r");
  if (!fp) throw std::runtime_error(std::string("cannot open bdyfile ") + bdyfile);
  char str[255], buf[64] = {0};
  const size_t len = std::strlen(name);
  while (std::strncmp(buf, name, len)) {
    if (!std::fgets(str, sizeof(str), fp)) {
      std::fclose(fp);
      throw std::runtime_error(std::string("time series ") + name + " not found in " + bdyfile);
    }
    buf[0] = 0;
    std::sscanf(str, "%63s", buf);
  }
  int rows = 0;
  buf[0] = 0;
  if (std::fgets(str, sizeof(str), fp)) std::sscanf(str, "%d %63s", &rows, buf);
  if (rows <= 0) { std::fclose(fp); throw std::runtime_error(std::string("empty time series ") + name); }
  const float mult = !std::strncmp(buf, "seconds", 7) ? 1.0f : !std::strncmp(buf, "minutes", 7) ? 60.0f : 3600.0f;
  AdaptiveSeries series;
  for (int i = 0; i < rows; ++i) {
    float v = 0, t = 0;
    if (!std::fgets(str, sizeof(str), fp)) break;
    std::sscanf(str, "%f %f", &v, &t);
    series.value.push_back(v);
    series.time.push_back(t * mult);
  }
  std::fclose(fp);
  return series;
}

struct AdaptiveBoundaries {
  AdaptiveSide side[4];
  std::vector<AdaptivePointSource> sources;

  bool any() const {
    for (const auto &s : side) if (s.type != BC_CLOSED) return true;
    return !sources.empty();
  }

  void read(const char *bcifile, const char *bdyfile, double xmin, double ymin,
            double cell, int xsz, int ysz) {
    if (!bcifile || !bcifile[0]) return;
    const char dirs[4] = {'N', 'E', 'S', 'W'};
    for (int k = 0; k < 4; ++k) {
      FILE *fp = std::fopen(bcifile, "r");
      if (!fp) throw std::runtime_error(std::string("cannot open bcifile ") + bcifile);
      char str[255];
      while (std::fgets(str, sizeof(str), fp)) {
        char dir = 0, type[16] = {0}, extra[64] = {0};
        float lower = 0, upper = 0;
        if (std::sscanf(str, "%c %f %f %15s %63s", &dir, &lower, &upper, type, extra) < 4 ||
            dir != dirs[k])
          continue;
        AdaptiveSide &s = side[k];
        s.type = adaptiveBcType(type);
        const double origin = (k == SIDE_N || k == SIDE_S) ? xmin : ymin;
        s.start = static_cast<int>((lower - origin) / cell);
        s.end = static_cast<int>((upper - origin) / cell) - 1;
        const int length = (k == SIDE_N || k == SIDE_S) ? xsz : ysz;
        if (s.start < 0) s.start = 0;
        if (s.end > length - 1) s.end = length - 1;
        if (s.type == BC_HFIX || s.type == BC_QFIX) s.inlet = std::strtof(extra, nullptr);
        if (s.type == BC_HVAR || s.type == BC_QVAR) s.series = readAdaptiveSeries(bdyfile, extra);
        break;
      }
      std::fclose(fp);
    }
    FILE *fp = std::fopen(bcifile, "r");
    char str[255];
    while (fp && std::fgets(str, sizeof(str), fp)) {
      char point = 0, type[16] = {0}, extra[64] = {0};
      float xs = 0, ys = 0;
      if (std::sscanf(str, "%c %f %f %15s %63s", &point, &xs, &ys, type, extra) < 4 || point != 'P')
        continue;
      AdaptivePointSource p;
      p.x = static_cast<int>((xs - xmin) / cell);
      p.y = static_cast<int>((ys - ymin) / cell);
      p.type = adaptiveBcType(type);
      if (p.type == BC_HFIX || p.type == BC_QFIX) p.value = std::strtof(extra, nullptr);
      else if (p.type == BC_HVAR || p.type == BC_QVAR) p.series = readAdaptiveSeries(bdyfile, extra);
      else continue;
      if (p.x < 0 || p.y < 0 || p.x >= xsz || p.y >= ysz) continue;
      sources.push_back(p);
    }
    if (fp) std::fclose(fp);
  }

  // Boundaries::update_all_inlets / PointSources::update_all_srcs
  void update(float now) {
    for (auto &s : side)
      if (s.type == BC_HVAR || s.type == BC_QVAR) s.inlet = s.series.at(now);
    for (auto &p : sources)
      if (p.type == BC_HVAR || p.type == BC_QVAR) p.value = p.series.at(now);
  }
};

} } } // namespace lis::metal::mwdg2
