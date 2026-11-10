#!/bin/bash
# Build SERGHEI-SWE and TRITON (CUDA, A100, MPI, double precision): sbatch build.sbatch [serghei|triton|both]
#SBATCH --job-name=om_build
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --gres=gpu:ga100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/ec/res4/scratch/lux0804/othermodels/logs/%x-%j.out
set -eo pipefail
unset LD_LIBRARY_PATH
module load prgenv/gnu gcc/8.5.0 cuda/12.6 hpcx-openmpi/2.14.0-cuda cmake/3.28.3 python3
W=/ec/res4/scratch/lux0804/othermodels; S=$W/src; WHAT=${1:-both}
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
echo "gcc $(gcc -dumpversion), nvcc $(nvcc --version | tail -1), mpicxx $(which mpicxx)"
if [ $WHAT != triton ]; then
  cd $S/serghei
  git rev-parse HEAD
  # PnetCDF from the release tarball (the git submodule needs autoconf >= 2.70)
  PN=$W/pnetcdf
  if [ ! -f $PN/lib/libpnetcdf.a ] && [ ! -f $PN/lib/libpnetcdf.so ]; then
    (cd $S && rm -rf pnetcdf-1.13.0 && tar xzf pnetcdf-1.13.0.tar.gz && cd pnetcdf-1.13.0 && \
     CC=mpicc CXX=mpicxx ./configure --prefix=$PN --disable-fortran > $W/logs/pnetcdf.log 2>&1 && \
     make -j 16 >> $W/logs/pnetcdf.log 2>&1 && make install >> $W/logs/pnetcdf.log 2>&1) || { tail -20 $W/logs/pnetcdf.log; exit 1; }
  fi
  export PNETCDF=$PN
  export OMPI_CXX=$S/serghei/kokkos/bin/nvcc_wrapper
  # Kokkos 5 needs C++20: nvcc uses gcc 12 as host compiler (MPI from the gcc 8.5 module)
  export NVCC_WRAPPER_DEFAULT_COMPILER=/usr/local/apps/gcc/12.2.0/bin/g++
  export LD_LIBRARY_PATH=/usr/local/apps/gcc/12.2.0/lib64:$LD_LIBRARY_PATH
  # Kokkos 5.1 (the SERGHEI submodule), CUDA for the A100
  KK=$W/kokkos_serghei
  if [ ! -f $KK/lib64/cmake/Kokkos/KokkosConfig.cmake ] && [ ! -f $KK/lib/cmake/Kokkos/KokkosConfig.cmake ]; then
    rm -rf $W/build_kokkos_serghei
    cmake -S $S/serghei/kokkos -B $W/build_kokkos_serghei -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$KK \
      -DCMAKE_CXX_COMPILER=$S/serghei/kokkos/bin/nvcc_wrapper -DCMAKE_CXX_STANDARD=20 \
      -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_SERIAL=ON > $W/logs/kokkos_cmake.log 2>&1 || { tail -30 $W/logs/kokkos_cmake.log; exit 1; }
    cmake --build $W/build_kokkos_serghei -j 16 --target install > $W/logs/kokkos_make.log 2>&1 || { grep -i error $W/logs/kokkos_make.log | head -20; exit 1; }
  fi
  rm -rf $W/build_serghei${SERGHEI_SUFFIX:-}
  cmake -S . -B $W/build_serghei${SERGHEI_SUFFIX:-} ${SERGHEI_EXTRA_CMAKE:-} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx \
    -DKokkos_ROOT=$KK -DSERGHEI_NETCDF_FORCING=ON \
    -DPnetCDF_ROOT=$PN -DPnetCDF_C_PATH=$PN > $W/logs/serghei_cmake.log 2>&1 || { tail -40 $W/logs/serghei_cmake.log; exit 1; }
  grep -iE "netcdf|kokkos.*cuda|precision|FP32" $W/logs/serghei_cmake.log | head -20
  cmake --build $W/build_serghei${SERGHEI_SUFFIX:-} -j 16 > $W/logs/serghei_make.log 2>&1 || { grep -iE "error" $W/logs/serghei_make.log | head -30; exit 1; }
  ls -la $W/build_serghei${SERGHEI_SUFFIX:-}/bin/serghei
  unset OMPI_CXX
  export NVCC_WRAPPER_DEFAULT_COMPILER=g++
fi
if [ $WHAT != serghei ]; then
  cd $S/triton
  git rev-parse HEAD
  export OMPI_CXX=$S/triton/external/kokkos/bin/nvcc_wrapper
  export TRITON_BACKEND=CUDA TRITON_ARCH=AMPERE80 TRITON_COMPILER=mpicxx TRITON_COMPILER_FLAGS="-DACTIVE_GPU=1" TRITON_RUN_COMMAND="mpirun -n 1"
  rm -rf $W/build_triton${TRITON_SUFFIX:-}
  cmake -S . -B $W/build_triton${TRITON_SUFFIX:-} -DCMAKE_BUILD_TYPE=Release ${TRITON_EXTRA_CMAKE:-} > $W/logs/triton_cmake.log 2>&1 || { tail -40 $W/logs/triton_cmake.log; exit 1; }
  grep -E "TRITON_|Kokkos.*CUDA|ARCH" $W/logs/triton_cmake.log | head -12
  cmake --build $W/build_triton${TRITON_SUFFIX:-} -j 16 > $W/logs/triton_make.log 2>&1 || { grep -iE "error" $W/logs/triton_make.log | head -30; exit 1; }
  ls -la $W/build_triton${TRITON_SUFFIX:-}/triton* | head
fi
