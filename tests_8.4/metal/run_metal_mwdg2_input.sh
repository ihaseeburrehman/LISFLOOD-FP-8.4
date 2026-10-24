#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
WORK=${1:-"$ROOT/build/metal_mwdg2_input_test"}
mkdir -p "$WORK"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_input.metal" -o "$WORK/input.air"
xcrun -sdk macosx metallib "$WORK/input.air" -o "$WORK/input.metallib"
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  "$ROOT/tests_8.4/metal/metal_mwdg2_input.mm" -o "$WORK/test"
"$WORK/test" "$WORK/input.metallib"
