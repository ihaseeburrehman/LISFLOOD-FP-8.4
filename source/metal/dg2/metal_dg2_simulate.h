//
// LISFLOOD-FP Metal DG2 Solver - Header
// C++ interface to Metal GPU compute
//

#pragma once

#ifdef __APPLE__

#include "../../geometry.h"
#include "../../lisflood.h"

namespace lis {
namespace metal {
namespace dg2 {

// Forward declaration of implementation
struct SimulatorImpl;

// Metal DG2 flow fields (mirroring CUDA structure)
struct Flow {
  NUMERIC_TYPE *H;
  NUMERIC_TYPE *HU;
  NUMERIC_TYPE *HV;
  NUMERIC_TYPE *maxH;
  NUMERIC_TYPE *totalHtm;
  NUMERIC_TYPE *maxHtm;
  NUMERIC_TYPE *initHtm;
};

// Main simulator class
class Simulator {
public:
  Simulator(Fnames &filenames, Pars &pars, Solver &solver, Arrays &arrays,
            States &states, BoundCs &boundCs, int verbose,
            int rank = 0, int size = 1, int r_begin = 1, int r_end = 0);

  ~Simulator();

  void initialize(NUMERIC_TYPE *H, NUMERIC_TYPE *HU, NUMERIC_TYPE *HV,
                  NUMERIC_TYPE *DEM, NUMERIC_TYPE *Mann);

  NUMERIC_TYPE step();

  void copy_to_host(NUMERIC_TYPE *H, NUMERIC_TYPE *HU, NUMERIC_TYPE *HV);

  void copy_max_to_host(NUMERIC_TYPE *maxH);

  void gather_rows();

  int get_rank() const;
  int get_size() const;

  NUMERIC_TYPE get_time() const;

  void apply_uniform_rain(NUMERIC_TYPE rain_rate);

  static bool is_available();

  bool is_valid() const { return impl != nullptr; }

private:
  SimulatorImpl *impl;
};

void simulate(Fnames &filenames, Files *files, States &states, Pars &pars,
              ::Solver &solver, BoundCs &boundCs, Stage &stage, Arrays &arrays,
              int verbose);

} // namespace dg2
} // namespace metal
} // namespace lis

#endif // __APPLE__
