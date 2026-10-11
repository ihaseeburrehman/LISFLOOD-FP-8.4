#pragma once

#ifdef __APPLE__
#include "../../lisflood.h"

namespace lis { namespace metal { namespace mwdg2 {
void simulate(Fnames &filenames, Files *files, States &states, Pars &pars,
              Solver &solver, BoundCs &boundCs, Stage &stage,
              Arrays &arrays, int verbose);
} } }
#endif
