#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
WORK=${1:-"$ROOT/build/metal_mwdg2_live_step"}
mkdir -p "$WORK"
xcrun -sdk macosx metal -fno-fast-math -I "$ROOT/source/metal/adaptive" -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_flow.metal" -o "$WORK/flow.air"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_selection.metal" -o "$WORK/selection.air"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_minicase.metal" -o "$WORK/reconstruct.air"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_tree_primitives.metal" -o "$WORK/tree.air"
xcrun -sdk macosx metallib "$WORK/flow.air" "$WORK/selection.air" \
  "$WORK/reconstruct.air" "$WORK/tree.air" -o "$WORK/combined.metallib"
xcrun -sdk macosx clang++ -std=c++14 -fobjc-arc \
  -framework Foundation -framework Metal \
  "$ROOT/tests_8.4/metal/metal_mwdg2_live_step.mm" \
  "$ROOT/source/metal/adaptive/metal_mwdg2_mesh.mm" \
  "$ROOT/source/metal/adaptive/metal_mwdg2_scan.mm" \
  -o "$WORK/test"
"$WORK/test" "$WORK/combined.metallib"
