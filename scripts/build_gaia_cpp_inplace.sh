#!/bin/sh
# build_gaia_cpp_inplace.sh — compile gaia_cpp (with all-token hidden-state
# capture) directly against gaia-core's installed llama headers + libllama,
# and deploy it into the running container's gaia_engine package.
#
# Why: the .so copied from gaia-prime is an OLDER build that lacks
# set_capture_all_tokens — all-token capture (~seq_len× more samples, needed for
# robust co-activation / synapse mapping, GAIA_Project-72q) requires the current
# gaia_cpp.cpp source. gaia-core already has g++/cmake, the full ggml/llama
# headers (site-packages/include) and libllama.so, so a direct g++ compile works
# WITHOUT rebuilding llama.cpp.
#
# Run INSIDE gaia-core:
#   docker compose exec -T gaia-core sh /gaia/GAIA_Project/scripts/build_gaia_cpp_inplace.sh
#
# NOTE: this is a runtime deploy (lost on image rebuild). The durable fix is to
# add this build stage to gaia-core's Dockerfile (the persistent-gaia_cpp task).
set -e

SRC=/gaia/GAIA_Project/gaia-engine/gaia_engine/cpp
INC=/usr/local/lib/python3.11/site-packages/include
DEST=/usr/local/lib/python3.11/site-packages/gaia_engine/cpp

python -m pybind11 --includes >/dev/null 2>&1 || pip install --quiet "pybind11[global]>=2.13"

PYINC=$(python -m pybind11 --includes)
EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "Compiling gaia_cpp$EXT (all-token) ..."
g++ -O3 -shared -fPIC -std=c++17 $PYINC -I"$INC" -I"$SRC" \
    "$SRC/gaia_cpp.cpp" -L/usr/local/lib -lllama -Wl,-rpath,/usr/local/lib \
    -o "/tmp/gaia_cpp$EXT"

cp "/tmp/gaia_cpp$EXT" "$DEST/gaia_cpp$EXT"
echo "Deployed → $DEST/gaia_cpp$EXT"
python -c "from gaia_engine.cpp import gaia_cpp; b=gaia_cpp.LlamaCppBackend('/models/core.gguf',0,[0],512); print('all-token capable:', hasattr(b,'set_capture_all_tokens'))" 2>/dev/null | grep all-token
