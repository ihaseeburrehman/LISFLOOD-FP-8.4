#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WORK=${1:-"$ROOT/build/metal_mwdg2_scan_test"}
mkdir -p "$WORK"
xcrun -sdk macosx metal -c \
  "$ROOT/source/metal/adaptive/metal_tree_primitives.metal" \
  -o "$WORK/tree.air"
xcrun -sdk macosx metallib "$WORK/tree.air" -o "$WORK/tree.metallib"
xcrun -sdk macosx clang++ -std=c++17 -fobjc-arc \
  -framework Foundation -framework Metal \
  "$ROOT/tests_8.4/metal_mwdg2_scan.mm" \
  "$ROOT/source/metal/adaptive/metal_mwdg2_scan.mm" \
  -o "$WORK/test"
"$WORK/test" "$WORK/tree.metallib"
