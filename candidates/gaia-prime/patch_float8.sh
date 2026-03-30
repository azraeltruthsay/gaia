#!/bin/bash
# Patch Float8_e8m0fnu references that are absent in PyTorch 2.7.0a0 (NGC 25.03).
# Replaces with Float8_e4m3fn so the code compiles. MXFP4 codepath is unreachable.
set -e
TARGET="${1:-.}"
find "$TARGET" -type f \( -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.cuh' \) -print0 | \
    xargs -0 grep -l 'Float8_e8m0fnu\|kFloat8_e8m0fnu\|float8_e8m0fnu' 2>/dev/null | \
    while read -r f; do
        sed -i 's/Float8_e8m0fnu/Float8_e4m3fn/g;s/kFloat8_e8m0fnu/kFloat8_e4m3fn/g;s/float8_e8m0fnu/float8_e4m3fn/g' "$f"
        echo "[patch_float8] Patched: $f"
    done
echo "[patch_float8] Done patching $TARGET"
