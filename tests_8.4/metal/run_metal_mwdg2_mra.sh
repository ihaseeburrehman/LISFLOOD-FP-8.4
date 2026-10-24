#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
WORK=${1:-"$ROOT/build/metal_mwdg2_mra_test"}
mkdir -p "$WORK"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_flow.metal" -o "$WORK/mra.air"
xcrun -sdk macosx metallib "$WORK/mra.air" -o "$WORK/mra.metallib"
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  "$ROOT/tests_8.4/metal/metal_mwdg2_mra.mm" -o "$WORK/test"
"$WORK/test" "$WORK/mra.metallib"
