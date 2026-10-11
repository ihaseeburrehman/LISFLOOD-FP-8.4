#pragma once
#include "../../lisflood.h"

namespace lis { namespace metal { namespace acc_nugrid {

/** ACC on the static non-uniform grid (cuda/acc_nugrid) on the Metal GPU. */
void simulate(Fnames &filenames, Files *files, States &states, Pars &pars,
              Solver &solver, BoundCs &boundCs, Stage &stage, Arrays &arrays,
              int verbose);

} } } // namespace lis::metal::acc_nugrid
