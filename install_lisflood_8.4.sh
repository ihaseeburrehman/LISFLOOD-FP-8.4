#!/usr/bin/env bash
# Install LISFLOOD-FP 8.4 on Linux with an NVIDIA GPU (CUDA) or on a Mac with
# Apple Silicon (Metal).
#
#   ./install_lisflood_8.4.sh --check    only list what is found and what is missing
#   ./install_lisflood_8.4.sh            check, then ask before building and testing
#   ./install_lisflood_8.4.sh --yes      check, build and test without asking
#
# Options
#   --precision float|double   numeric precision (default float)
#   --mpi auto|on|off          multi-GPU support (default auto: on if MPI is found)
#   --cuda-arch N              NVIDIA architecture, e.g. 80 A100, 90 H100
#                              (default: read from the GPU, else 80)
#   --build DIR                build folder (default build, or build_double)
#   --prefix DIR               also copy the programs to DIR/bin
#   --jobs N                   parallel compile jobs (default: all cores)
#   --no-test                  skip the test run of example/example.par
#
# Libraries in non-standard places: point to the folder that holds their
# bin/, include/ and lib/, for example
#   NETCDF_DIR=$HOME/opt/netcdf CUDA_HOME=/usr/local/cuda-12.6 ./install_lisflood_8.4.sh
# Variables: CUDA_HOME, NETCDF_DIR, NUMA_DIR, MPI_HOME. On an HPC system
# "module load" the compilers and libraries first (see README.md).
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MODE=ask; PRECISION=float; MPI=auto; CUDA_ARCH=""; BUILD=""; PREFIX=""; TEST=yes
JOBS=$( (nproc || sysctl -n hw.ncpu) 2>/dev/null || echo 4)

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE=check ;;
    --yes|-y) MODE=yes ;;
    --precision) PRECISION=${2:?}; shift ;;
    --mpi) MPI=${2:?}; shift ;;
    --cuda-arch) CUDA_ARCH=${2:?}; shift ;;
    --build) BUILD=${2:?}; shift ;;
    --prefix) PREFIX=${2:?}; shift ;;
    --jobs) JOBS=${2:?}; shift ;;
    --no-test) TEST=no ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1"; usage 1 ;;
  esac
  shift
done
case "$PRECISION" in float|double) ;; *) echo "--precision must be float or double"; exit 1 ;; esac
case "$MPI" in auto|on|off) ;; *) echo "--mpi must be auto, on or off"; exit 1 ;; esac
[ -n "$BUILD" ] || { BUILD=$HERE/build; [ "$PRECISION" = double ] && BUILD=$HERE/build_double; }
case "$BUILD" in /*) ;; *) BUILD=$PWD/$BUILD ;; esac

# Folders given by the user come first in every search.
for v in CUDA_HOME MPI_HOME; do
  d=${!v:-}; [ -n "$d" ] && [ -d "$d/bin" ] && PATH=$d/bin:$PATH
done
[ -z "${CUDA_HOME:-}" ] && [ -x /usr/local/cuda/bin/nvcc ] && ! command -v nvcc >/dev/null && PATH=/usr/local/cuda/bin:$PATH
export PATH

OS=$(uname)
MISSING=0; HINTS=()
ok()   { printf "  %-10s ok       %s\n" "$1" "$2"; }
miss() { printf "  %-10s MISSING  %s\n" "$1" "$2"; MISSING=$((MISSING + 1)); HINTS+=("$3"); }
warn() { printf "  %-10s note     %s\n" "$1" "$2"; [ -n "${3:-}" ] && HINTS+=("$3"); }
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# Package manager, for the install hints.
if   [ "$OS" = Darwin ];            then PKG="brew install"
elif command -v apt-get >/dev/null; then PKG="sudo apt-get install"
elif command -v dnf >/dev/null;     then PKG="sudo dnf install"
elif command -v yum >/dev/null;     then PKG="sudo yum install"
elif command -v zypper >/dev/null;  then PKG="sudo zypper install"
else PKG="(your package manager) install"; fi
pkg() { # pkg apt-name dnf-name brew-name
  case "$PKG" in
    *apt-get*) echo "$PKG $1" ;; *dnf*|*yum*|*zypper*) echo "$PKG $2" ;;
    brew*) echo "$PKG $3" ;; *) echo "$PKG $1" ;; esac
}
HAVE_MODULES=no; { command -v module >/dev/null || [ -n "${MODULESHOME:-}" ]; } && HAVE_MODULES=yes
mod_hint() { [ $HAVE_MODULES = yes ] && echo " (on an HPC system try: module avail $1)"; }

# find_lib NAME HEADER LIB [DIRS...]: prefix that holds include/HEADER and lib*/libLIB.*
find_lib() {
  local h=$2 l=$3; shift 3
  for p in "$@" /usr /usr/local /opt/homebrew /opt/local; do
    [ -n "$p" ] && [ -f "$p/include/$h" ] || continue
    for f in "$p"/lib*/lib"$l".* "$p"/lib/*/lib"$l".*; do
      [ -e "$f" ] && { echo "$p"; return 0; }
    done
  done
  return 1
}

echo "LISFLOOD-FP 8.4 installer: checking this machine ($OS $(uname -m))"
echo

# --- build tools ---------------------------------------------------------------
if command -v cmake >/dev/null; then
  v=$(cmake --version | head -1 | awk '{print $3}')
  if ver_ge "$v" 3.13; then ok cmake "$v"; else miss cmake "$v is too old, need 3.13 or newer" "$(pkg cmake cmake cmake), or: python3 -m pip install --user cmake$(mod_hint cmake)"; fi
else
  miss cmake "not found (need 3.13 or newer)" "$(pkg cmake cmake cmake), or: python3 -m pip install --user cmake$(mod_hint cmake)"
fi

if CXX_BIN=$(command -v "${CXX:-c++}" || command -v g++ || command -v clang++); then
  ok "C++" "$("$CXX_BIN" --version | head -1)"
else
  miss "C++" "no C++ compiler found" "$(pkg g++ gcc-c++ llvm)$( [ "$OS" = Darwin ] && echo ' or: xcode-select --install')"
fi

# --- GPU back end --------------------------------------------------------------
GPU=no
if [ "$OS" = Darwin ]; then
  BACKEND=Metal
  if [ "$(uname -m)" = arm64 ]; then ok "Mac" "Apple Silicon"; GPU=yes
  else warn "Mac" "Intel Mac: the Metal solvers are built and tested on Apple Silicon only" ""; fi
  if xcrun -sdk macosx -f metal >/dev/null 2>&1 && xcrun -sdk macosx metal -v >/dev/null 2>&1; then
    ok Metal "$(xcrun -sdk macosx metal -v 2>&1 | head -1)"
  else
    miss Metal "Metal shader compiler not available" \
      "Install Xcode from the App Store, then: sudo xcode-select -s /Applications/Xcode.app; with Xcode 26 or newer also: xcodebuild -downloadComponent MetalToolchain"
  fi
  OMP=$(brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  if [ -f "$OMP/include/omp.h" ]; then ok OpenMP "libomp in $OMP"
  else miss OpenMP "libomp not found" "brew install libomp"; fi
else
  BACKEND=CUDA
  if command -v nvcc >/dev/null; then
    v=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    if ver_ge "$v" 11.4; then ok CUDA "nvcc $v ($(command -v nvcc))"
    else miss CUDA "nvcc $v is too old, need 11.4 or newer" "Install a newer CUDA toolkit: https://developer.nvidia.com/cuda-downloads$(mod_hint cuda)"; fi
  else
    miss CUDA "nvcc not found" "Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads), or set CUDA_HOME=/path/to/cuda$(mod_hint cuda)"
  fi
  if command -v nvidia-smi >/dev/null && g=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) && [ -n "$g" ]; then
    GPU=yes
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    [ -z "$CUDA_ARCH" ] && [ -n "$cc" ] && CUDA_ARCH=$cc
    ok GPU "$g${cc:+ (architecture $cc)}"
  else
    warn GPU "no NVIDIA GPU visible here (a login node?): the build works, the test run is skipped" \
      "No GPU here: the build uses --cuda-arch ${CUDA_ARCH:-80}; pass the architecture of the GPUs you will run on (80 A100, 90 H100, 86 RTX 30xx, 89 RTX 40xx)."
  fi
  if p=$(find_lib numa numa.h numa "${NUMA_DIR:-}"); then ok NUMA "$p"
  elif ldconfig -p 2>/dev/null | grep -q 'libnuma\.so ' && [ -f /usr/include/numa.h ]; then ok NUMA /usr
  else miss NUMA "libnuma (numa.h) not found" "$(pkg libnuma-dev numactl-devel libnuma), or set NUMA_DIR=/path/to/numa"; fi
  if echo 'int main(){return 0;}' | "${CXX_BIN:-c++}" -fopenmp -x c++ - -o /dev/null >/dev/null 2>&1; then ok OpenMP "compiler supports -fopenmp"
  else warn OpenMP "not found: only the CPU solvers use it, the GPU solvers do not need it" ""; fi
fi

# --- NetCDF ------------------------------------------------------------------
NC=""
if [ -n "${NETCDF_DIR:-}" ]; then NC=$(find_lib netcdf netcdf.h netcdf "$NETCDF_DIR"); fi
if [ -z "$NC" ] && command -v nc-config >/dev/null; then NC=$(find_lib netcdf netcdf.h netcdf "$(nc-config --prefix 2>/dev/null)"); fi
if [ -z "$NC" ] && command -v pkg-config >/dev/null; then NC=$(find_lib netcdf netcdf.h netcdf "$(pkg-config --variable=prefix netcdf 2>/dev/null)"); fi
[ -z "$NC" ] && [ "$OS" = Darwin ] && NC=$(find_lib netcdf netcdf.h netcdf "$(brew --prefix netcdf 2>/dev/null)")
[ -z "$NC" ] && NC=$(find_lib netcdf netcdf.h netcdf)
if [ -n "$NC" ]; then
  ok NetCDF "$NC$(command -v nc-config >/dev/null && echo " (version $(nc-config --version 2>/dev/null | awk '{print $2}'))")"
else
  miss NetCDF "NetCDF-C library and netcdf.h not found" "$(pkg libnetcdf-dev netcdf-devel netcdf), or set NETCDF_DIR=/path/to/netcdf$(mod_hint netcdf)"
fi

# --- MPI (optional: several GPUs) --------------------------------------------
MPICXX=""
if [ "$OS" = Darwin ]; then
  MPICXX=$(command -v mpicxx-mpich-mp || true)
  if [ -n "$MPICXX" ]; then ok MPI "MPICH ($MPICXX)"
  elif command -v mpicxx >/dev/null; then
    warn MPI "only $(command -v mpicxx) found; Open MPI 5 hangs at start-up on macOS, so multi-rank is left off" \
      "Multi-rank on a Mac (correctness checks only, all ranks share one GPU): sudo port install mpich"
    [ "$MPI" = auto ] && MPI=off
  else warn MPI "not found: one GPU only (fine on a Mac)" ""; [ "$MPI" = auto ] && MPI=off; fi
else
  MPICXX=$(command -v mpicxx || true)
  if [ -n "$MPICXX" ]; then ok MPI "$MPICXX"
  else
    warn MPI "not found: the build runs on one GPU; install MPI to run fv1, fv2 and dg2 on several GPUs" \
      "Several GPUs: $(pkg libopenmpi-dev openmpi-devel open-mpi), or set MPI_HOME=/path/to/mpi$(mod_hint openmpi)"
    [ "$MPI" = auto ] && MPI=off
  fi
fi
if [ "$MPI" = on ] && [ -z "$MPICXX" ]; then miss MPI "--mpi on, but no MPI compiler found" "Install MPI or set MPI_HOME"; fi

command -v python3 >/dev/null && ok python3 "$(python3 --version 2>&1) (only for the test scripts in tests_8.4)" \
  || warn python3 "not found: only the test scripts in tests_8.4 need it" ""

# --- summary -----------------------------------------------------------------
echo
if [ ${#HINTS[@]} -gt 0 ]; then
  [ $MISSING -gt 0 ] && echo "To install what is missing:" || echo "Notes:"
  for h in "${HINTS[@]}"; do echo "  - $h"; done
  if [ $MISSING -gt 0 ] && [ "$OS" != Darwin ]; then
    echo "  Without administrator rights: ask your system administrators, or get the libraries"
    echo "  from conda, e.g. conda install -c conda-forge libnetcdf, then NETCDF_DIR=\$CONDA_PREFIX."
  fi
  echo
fi
echo "Plan: $BACKEND back end, $PRECISION precision, MPI $MPI$( [ "$BACKEND" = CUDA ] && echo ", architecture ${CUDA_ARCH:-80}") -> $BUILD"
[ -n "$PREFIX" ] && echo "      then copy the programs to $PREFIX/bin"
if [ $MISSING -gt 0 ]; then
  echo "$MISSING required item(s) missing: install them, then run this script again."
  exit 1
fi
[ "$MODE" = check ] && { echo "Everything needed is present. Run ./install_lisflood_8.4.sh to build."; exit 0; }
if [ "$MODE" = ask ]; then
  if [ -t 0 ]; then
    read -r -p "Build LISFLOOD-FP 8.4 now? [y/N] " a
    case "$a" in y|Y|yes|YES) ;; *) echo "Nothing built."; exit 0 ;; esac
  else
    echo "Not a terminal: run with --yes to build."; exit 0
  fi
fi

# --- build ---------------------------------------------------------------------
CPP=""
for d in "${NETCDF_DIR:-}" "${NUMA_DIR:-}" "${MPI_HOME:-}" "$NC"; do [ -n "$d" ] && CPP=${CPP:+$CPP:}$d; done
[ -n "$CPP" ] && export CMAKE_PREFIX_PATH=$CPP${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
[ -n "${CUDA_HOME:-}" ] && export CUDACXX=$CUDA_HOME/bin/nvcc
[ -n "$NC" ] && export PKG_CONFIG_PATH=$NC/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
echo
echo "Building (log: $BUILD.log) ..."
mkdir -p "$(dirname "$BUILD")"
if ! env PRECISION=$PRECISION MPI=$MPI BUILD="$BUILD" JOBS=$JOBS ${CUDA_ARCH:+CUDA_ARCH=$CUDA_ARCH} \
     "$HERE/build.sh" > "$BUILD.log" 2>&1; then
  echo "BUILD FAILED. The first errors:"
  grep -m 15 -i "error" "$BUILD.log" || tail -30 "$BUILD.log"
  echo "Full log: $BUILD.log"
  exit 1
fi
"$BUILD/lisflood" -version 2>&1 | grep -m1 "LISFLOOD-FP" | sed 's/^/Built: /'

# --- test ----------------------------------------------------------------------
if [ $TEST = yes ] && [ $GPU = yes ]; then
  T=$BUILD/example_test; rm -rf "$T"; cp -R "$HERE/example" "$T"
  echo "Testing with example/example.par (fv1 on the GPU, 3 h storm) ..."
  if (cd "$T" && "$BUILD/lisflood" example.par > run.log 2>&1) &&
     awk 'END {exit !($1 >= 10800 && $6 > 0 && $9 > 0)}' "$T/results/example.mass"; then
    echo "TEST PASSED: $(awk 'END {printf "t = %.0f s, volume %.0f m3, outflow %.2f m3/s", $1, $6, $9}' "$T/results/example.mass")"
  else
    echo "TEST FAILED: see $T/run.log"; exit 1
  fi
elif [ $TEST = yes ]; then
  echo "Test skipped: no GPU on this machine. Run it on a GPU machine with:"
  echo "  cd example && $BUILD/lisflood example.par"
fi

# --- install -------------------------------------------------------------------
BIN=$BUILD
if [ -n "$PREFIX" ]; then
  mkdir -p "$PREFIX/bin" && cp "$BUILD"/lisflood "$BUILD"/generateDG2DEM "$BUILD"/generateDG2start "$PREFIX/bin/" &&
  { ls "$BUILD"/*.metallib >/dev/null 2>&1 && cp "$BUILD"/*.metallib "$PREFIX/bin/" || true; } &&
  BIN=$PREFIX/bin && echo "Copied the programs to $BIN"
fi

echo
echo "Done. To run a case:"
echo "  $BIN/lisflood case.par"
[ "$MPI" != off ] && [ "$BACKEND" = CUDA ] && echo "  mpirun -np 4 $BIN/lisflood case.par      # with 'GPUs 4' in case.par (fv1, fv2, dg2)"
echo "Start from example/example.par: every keyword in it is explained."
echo "To call it as 'lisflood', add to your shell start-up file:  export PATH=\"$BIN:\$PATH\""
