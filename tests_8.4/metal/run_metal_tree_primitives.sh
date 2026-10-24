#!/bin/sh
# Native GPU primitive tests; this does not certify an adaptive flow solver.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
WORK=${1:-"$ROOT/build/metal_tree_test"}
mkdir -p "$WORK"
xcrun -sdk macosx metal -fno-fast-math -c \
  "$ROOT/source/metal/adaptive/metal_tree_primitives.metal" -o "$WORK/tree.air"
xcrun -sdk macosx metallib "$WORK/tree.air" -o "$WORK/tree.metallib"
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  "$ROOT/tests_8.4/metal/metal_tree_primitives.mm" -o "$WORK/test"
"$WORK/test" "$WORK/tree.metallib"
