#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WORK=${1:-"$ROOT/build/metal_mwdg2_selection_test"}
mkdir -p "$WORK"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_selection.metal" -o "$WORK/selection.air"
xcrun -sdk macosx metallib "$WORK/selection.air" -o "$WORK/selection.metallib"
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  "$ROOT/tests_8.4/metal_mwdg2_selection.mm" -o "$WORK/test"
"$WORK/test" "$WORK/selection.metallib"
