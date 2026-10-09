//
// LISFLOOD-FP Metal ACC Solver - Header
// C++ interface to the Metal implementation of the local-inertial solver.
//

#pragma once

#ifdef __APPLE__

#include "../../geometry.h"
#include "../../lisflood.h"

namespace lis {
namespace metal {
namespace acc {

struct SimulatorImpl;

class Simulator {
public:
  Simulator(Fnames &filenames, Pars &pars, ::Solver &solver, Arrays &arrays,
            States &states, BoundCs &boundCs, int verbose);
  ~Simulator();

  void initialize(NUMERIC_TYPE *H, NUMERIC_TYPE *DEM, NUMERIC_TYPE *Mann);

  // Advance one timestep; returns the timestep actually taken.
  NUMERIC_TYPE step();

  // Copy H (and, when voutput is on, the velocities) back to the CPU arrays.
  void copy_to_host(NUMERIC_TYPE *H, NUMERIC_TYPE *Qx, NUMERIC_TYPE *Qy);

  // Fold the GPU mass counters into BoundCs/Pars. Call once per mass
  // interval, just before update_mass_stats().
  void sync_mass();

  // Copy the running maximum-depth bookkeeping back for the .max output.
  void copy_max_to_host(NUMERIC_TYPE *maxH, NUMERIC_TYPE *maxHtm,
                        NUMERIC_TYPE *totalHtm, NUMERIC_TYPE *initHtm);

  static bool is_available();
  bool is_valid() const { return impl != nullptr; }

private:
  SimulatorImpl *impl;
};

// Entry point used by metal_integrate.cpp
void simulate(Fnames &filenames, Files *files, States &states, Pars &pars,
              ::Solver &solver, BoundCs &boundCs, Stage &stage, Arrays &arrays,
              int verbose);

} // namespace acc
} // namespace metal
} // namespace lis

#endif // __APPLE__
