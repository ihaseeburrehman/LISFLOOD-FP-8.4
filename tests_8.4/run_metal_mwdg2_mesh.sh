#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WORK=${1:-"$ROOT/build/metal_mwdg2_mesh_test"}
mkdir -p "$WORK"
xcrun -sdk macosx metal -I "$ROOT/source/metal/adaptive" -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_mra.metal" -o "$WORK/mra.air"
xcrun -sdk macosx metal -c \
  "$ROOT/source/metal/adaptive/metal_mwdg2_selection.metal" -o "$WORK/selection.air"
xcrun -sdk macosx metal -c \
  "$ROOT/source/metal/adaptive/metal_tree_primitives.metal" -o "$WORK/tree.air"
xcrun -sdk macosx metallib "$WORK/mra.air" "$WORK/selection.air" \
  "$WORK/tree.air" -o "$WORK/mwdg2.metallib"
xcrun -sdk macosx clang++ -std=c++14 -fobjc-arc \
  -framework Foundation -framework Metal \
  "$ROOT/tests_8.4/metal_mwdg2_mesh.mm" \
  "$ROOT/source/metal/adaptive/metal_mwdg2_mesh.mm" \
  "$ROOT/source/metal/adaptive/metal_mwdg2_scan.mm" \
  -o "$WORK/test"
"$WORK/test" "$WORK/mwdg2.metallib"
