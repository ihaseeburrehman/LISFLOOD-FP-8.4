// LISFLOOD-FP Metal GPU back end: entry points called from lisflood.cpp
#pragma once
#include "../lisflood.h"

// true when the machine has a Metal GPU the FV1 solver can use
bool metal_is_available();

// run the FV1 solver on the Metal GPU
void run_metal_fv1(Fnames &filenames, Files *files, States &states, Pars &pars,
                   Solver &solver, BoundCs &boundCs, Stage &stage,
                   Arrays &arrays, int verbose);

// run the FV2 solver on the Metal GPU
void run_metal_fv2(Fnames &filenames, Files *files, States &states, Pars &pars,
                   Solver &solver, BoundCs &boundCs, Stage &stage,
                   Arrays &arrays, int verbose);

// run the DG2 solver on the Metal GPU
void run_metal_dg2(Fnames &filenames, Files *files, States &states, Pars &pars,
                   Solver &solver, BoundCs &boundCs, Stage &stage,
                   Arrays &arrays, int verbose);

// run the local-inertial (ACC) solver on the Metal GPU
void run_metal_acc(Fnames &filenames, Files *files, States &states, Pars &pars,
                   Solver &solver, BoundCs &boundCs, Stage &stage,
                   Arrays &arrays, int verbose);

// Run the adaptive solver equations at the finest level in dense Metal
// buffers.  This is the deterministic fallback used until a native Metal
// replacement for CUDA/CUB tree sorting and compaction is available.
void run_metal_hwfv1(Fnames &filenames, Files *files, States &states, Pars &pars,
                     Solver &solver, BoundCs &boundCs, Stage &stage,
                     Arrays &arrays, int verbose);
void run_metal_mwdg2(Fnames &filenames, Files *files, States &states, Pars &pars,
                     Solver &solver, BoundCs &boundCs, Stage &stage,
                     Arrays &arrays, int verbose);
void run_metal_acc_nugrid(Fnames &filenames, Files *files, States &states,
                          Pars &pars, Solver &solver, BoundCs &boundCs,
                          Stage &stage, Arrays &arrays, int verbose);
