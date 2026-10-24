#!/usr/bin/env bash
# Build LISFLOOD-FP 8.4 in one command:   ./build.sh
#
#   macOS (Apple Silicon): Metal back end, all seven GPU solvers.
#   Linux + nvcc:           CUDA back end, all seven GPU solvers.
#   Multi-GPU (MPI) is switched on when an MPI compiler is found.
#
# Optional settings (environment variables):
#   BUILD=build        build directory
#   PRECISION=float    float (default, as the GPU solvers use) or double
#   MPI=auto           auto | on | off
#   CUDA_ARCH=         CUDA architecture, e.g. 80 for A100 (default: detected, else 80)
#   JOBS=8             parallel compile jobs
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-$HERE/build}
PRECISION=${PRECISION:-float}
MPI=${MPI:-auto}
JOBS=${JOBS:-8}

case "$PRECISION" in
  float)  CONFIG=$HERE/config.float.cmake ;;
  double) CONFIG=$HERE/source/config.default.cmake ;;
  *) echo "PRECISION must be float or double" >&2; exit 1 ;;
esac

ARGS=(-DCMAKE_BUILD_TYPE=Release -D_CONFIG="$CONFIG")

if [ "$(uname)" = Darwin ]; then
  BACKEND=Metal
  OMP=$(brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  ARGS+=(-DOpenMP_ROOT="$OMP"
         -DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I$OMP/include"
         -DOpenMP_C_FLAGS="-Xclang -fopenmp -I$OMP/include"
         -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_C_LIB_NAMES=omp
         -DOpenMP_omp_LIBRARY="$OMP/lib/libomp.dylib")
  # MacPorts MPICH is preferred: Open MPI 5 hangs at start-up on macOS.
  MPICXX=$(command -v mpicxx-mpich-mp || command -v mpicxx || true)
  MPI_FLAG=LISFLOOD_ENABLE_METAL_MPI
else
  command -v nvcc >/dev/null || { echo "nvcc not found: load CUDA first" >&2; exit 1; }
  BACKEND=CUDA
  if [ -z "${CUDA_ARCH:-}" ]; then
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | head -1 | tr -d '.' || true)
    CUDA_ARCH=${CUDA_ARCH:-80}
  fi
  ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
  MPICXX=$(command -v mpicxx || true)
  MPI_FLAG=LISFLOOD_ENABLE_CUDA_MPI
fi

if [ "$MPI" = on ] || { [ "$MPI" = auto ] && [ -n "$MPICXX" ]; }; then
  [ -n "$MPICXX" ] || { echo "MPI=on but no MPI compiler found" >&2; exit 1; }
  ARGS+=(-D$MPI_FLAG=ON -DMPI_CXX_COMPILER="$MPICXX")
  MPI_STATE="on ($MPICXX)"
else
  ARGS+=(-D$MPI_FLAG=OFF)
  MPI_STATE=off
fi

echo "LISFLOOD-FP 8.4: $BACKEND, $PRECISION precision, MPI $MPI_STATE -> $BUILD"
cmake -S "$HERE/source" -B "$BUILD" "${ARGS[@]}"
cmake --build "$BUILD" -j "$JOBS" --target lisflood generateDG2DEM generateDG2start
echo "Built $BUILD/lisflood"
