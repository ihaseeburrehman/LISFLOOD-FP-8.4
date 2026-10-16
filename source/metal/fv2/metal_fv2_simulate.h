//
// LISFLOOD-FP Metal FV2 Solver - Header
// C++ interface to Metal GPU compute
//

#pragma once

#ifdef __APPLE__

#include "../../geometry.h"
#include "../../lisflood.h"

namespace lis {
namespace metal {
namespace fv2 {

// Forward declaration of implementation
struct SimulatorImpl;

// Metal FV2 flow fields (mirroring CUDA structure)
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

  // Initialize from CPU data
  void initialize(NUMERIC_TYPE *H, NUMERIC_TYPE *HU, NUMERIC_TYPE *HV,
                  NUMERIC_TYPE *DEM, NUMERIC_TYPE *Mann);

  // Run one timestep
  NUMERIC_TYPE step();

  // Copy results back to CPU
  void copy_to_host(NUMERIC_TYPE *H, NUMERIC_TYPE *HU, NUMERIC_TYPE *HV);

  // Copy the running maximum depth field back to CPU (for the .max output)
  void copy_max_to_host(NUMERIC_TYPE *maxH);

  // Gather distributed rows to rank 0 before host copy / output
  void gather_rows();

  // MPI rank and size accessors
  int get_rank() const;
  int get_size() const;

  // Get current simulation time
  NUMERIC_TYPE get_time() const;

  // Update uniform rainfall
  void apply_uniform_rain(NUMERIC_TYPE rain_rate);

  // Check if Metal is available on this system
  static bool is_available();

  // Check if this simulator instance is valid (initialized successfully)
  bool is_valid() const { return impl != nullptr; }

private:
  SimulatorImpl *impl; // Pointer to implementation (PIMPL idiom)
};

// Entry point for LISFLOOD-FP integration
// Entry point for LISFLOOD-FP integration
void simulate(Fnames &filenames, Files *files, States &states, Pars &pars,
              ::Solver &solver, BoundCs &boundCs, Stage &stage, Arrays &arrays,
              int verbose);

} // namespace fv2
} // namespace metal
} // namespace lis

#endif // __APPLE__
