#!/bin/sh
# Standalone adaptive MWDG2 end-to-end mini-case. This does not modify or
# dispatch through the LISFLOOD-FP application.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WORK=\${1:-"$ROOT/build/metal_mwdg2_adaptive_minicase"}
mkdir -p "$WORK"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_flow.metal" -o "$WORK/flow.air"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_input.metal" -o "$WORK/input.air"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_selection.metal" -o "$WORK/selection.air"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_minicase.metal" -o "$WORK/minicase.air"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_tree_primitives.metal" -o "$WORK/tree.air"
xcrun -sdk macosx metallib \
  "$WORK/flow.air" "$WORK/input.air" "$WORK/selection.air" \
  "$WORK/minicase.air" "$WORK/tree.air" -o "$WORK/adaptive.metallib"
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  "$ROOT/tests_8.4/metal_mwdg2_adaptive_minicase.mm" -o "$WORK/test"
"$WORK/test" "$WORK/adaptive.metallib"
