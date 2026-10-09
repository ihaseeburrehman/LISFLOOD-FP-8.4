// LISFLOOD-FP Metal GPU back end: entry points called from lisflood.cpp
#include "metal_integrate.h"
#include "fv1/metal_fv1_simulate.h"
#include "acc/metal_acc_simulate.h"
#include "fv2/metal_fv2_simulate.h"
#include "dg2/metal_dg2_simulate.h"
#include "../swe/fields.h"

bool metal_is_available() { return lis::metal::fv1::Simulator::is_available(); }

void run_metal_fv1(Fnames &filenames, Files *files, States &states, Pars &pars,
                   Solver &solver, BoundCs &boundCs, Stage &stage,
                   Arrays &arrays, int verbose) {
  if (verbose)
    printf("\nUsing the Metal GPU (Apple Silicon) for the FV1 solver\n\n");
  lis::metal::fv1::simulate(filenames, files, states, pars, solver, boundCs,
                            stage, arrays, verbose);
}

void run_metal_fv2(Fnames &filenames, Files *files, States &states, Pars &pars,
                   Solver &solver, BoundCs &boundCs, Stage &stage,
                   Arrays &arrays, int verbose) {
  if (verbose)
    printf("\nUsing the Metal GPU (Apple Silicon) for the FV2 solver\n\n");
  // the Metal host works on the FV1 arrays (HU, HV, Z*, fluxes), which the
  // CPU allocates only for fv1
  if (arrays.HU == nullptr) allocate_swe_fields(&pars, &arrays);
  lis::metal::fv2::simulate(filenames, files, states, pars, solver, boundCs,
                            stage, arrays, verbose);
}

void run_metal_dg2(Fnames &filenames, Files *files, States &states, Pars &pars,
                   Solver &solver, BoundCs &boundCs, Stage &stage,
                   Arrays &arrays, int verbose) {
  if (verbose)
    printf("\nUsing the Metal GPU (Apple Silicon) for the DG2 solver\n\n");
  if (arrays.HU == nullptr) allocate_swe_fields(&pars, &arrays);
  lis::metal::dg2::simulate(filenames, files, states, pars, solver, boundCs,
                            stage, arrays, verbose);
}

void run_metal_acc(Fnames &filenames, Files *files, States &states, Pars &pars,
                   Solver &solver, BoundCs &boundCs, Stage &stage,
                   Arrays &arrays, int verbose) {
  if (verbose)
    printf("\nUsing the Metal GPU (Apple Silicon) for the ACC solver\n\n");
  lis::metal::acc::simulate(filenames, files, states, pars, solver, boundCs,
                            stage, arrays, verbose);
}

void run_metal_hwfv1(Fnames &filenames, Files *files, States &states, Pars &pars,
                     Solver &solver, BoundCs &boundCs, Stage &stage,
                     Arrays &arrays, int verbose) {
  fprintf(stderr, "NOTICE: Metal HWFV1 uses the dense finest-level mesh; "
                  "epsilon does not compress the mesh.\n");
  if (verbose) {
    printf("\nUsing the Metal GPU for HWFV1 in dense finest-level mode\n");
    printf("  adaptive threshold epsilon=%g (mesh compression is disabled)\n\n",
           (double)solver.epsilon);
  }
  if (arrays.HU == nullptr) allocate_swe_fields(&pars, &arrays);
  lis::metal::fv1::simulate(filenames, files, states, pars, solver, boundCs,
                            stage, arrays, verbose);
}

void run_metal_mwdg2(Fnames &filenames, Files *files, States &states, Pars &pars,
                     Solver &solver, BoundCs &boundCs, Stage &stage,
                     Arrays &arrays, int verbose) {
  fprintf(stderr, "NOTICE: Metal MWDG2 uses the dense finest-level mesh; "
                  "epsilon does not compress the mesh.\n");
  if (verbose) {
    printf("\nUsing the Metal GPU for MWDG2 in dense finest-level mode\n");
    printf("  adaptive threshold epsilon=%g (mesh compression is disabled)\n\n",
           (double)solver.epsilon);
  }
  if (arrays.HU == nullptr) allocate_swe_fields(&pars, &arrays);
  lis::metal::dg2::simulate(filenames, files, states, pars, solver, boundCs,
                            stage, arrays, verbose);
}
