#!/bin/bash
# cmake wrapper that intercepts cmake invocations during vLLM's build.
#
# Problem: vLLM's cmake uses FetchContent to download qutlass, which
# contains Float8_e8m0fnu references absent from NGC PyTorch 2.7.0a0.
# FetchContent populates sources into .deps/ and compilation happens
# within the same cmake invocation, so we can't patch after the fact.
#
# Solution: On the first cmake invocation (configure), we let it run.
# If it fails due to the known Float8 issue, we patch .deps/ and
# re-run cmake with the same arguments so compilation succeeds.
#
# Setup (in Dockerfile):
#   mv /usr/local/bin/cmake /usr/local/bin/cmake.real
#   ln -s /app/cmake_wrapper.sh /usr/local/bin/cmake

REAL_CMAKE=/usr/local/bin/cmake.real
PATCH_SCRIPT=/app/patch_float8.sh
VLLM_DIR=/tmp/vllm

if [ ! -x "$REAL_CMAKE" ]; then
    echo "[cmake_wrapper] ERROR: Real cmake not found at $REAL_CMAKE" >&2
    exit 1
fi

# Run the real cmake
"$REAL_CMAKE" "$@"
EXIT_CODE=$?

# If cmake failed and fetched deps exist, patch them and retry once
DEPS_DIR="$VLLM_DIR/.deps"
if [ $EXIT_CODE -ne 0 ] && [ -d "$DEPS_DIR" ] && [ -x "$PATCH_SCRIPT" ]; then
    echo "[cmake_wrapper] cmake failed (exit $EXIT_CODE). Patching fetched deps and retrying..."
    "$PATCH_SCRIPT" "$DEPS_DIR"
    "$REAL_CMAKE" "$@"
    EXIT_CODE=$?
elif [ -d "$DEPS_DIR" ] && [ -x "$PATCH_SCRIPT" ]; then
    # Even on success, patch deps for any subsequent cmake calls (e.g. --build)
    "$PATCH_SCRIPT" "$DEPS_DIR"
fi

exit $EXIT_CODE
