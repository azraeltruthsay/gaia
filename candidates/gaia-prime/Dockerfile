# gaia-prime-candidate: Custom vLLM inference server for GAIA
# RTX 5080 Blackwell (sm_120 / compute capability 12.0) compatible build
#
# Uses NGC PyTorch base image which includes CUDA 12.8+ and PyTorch
# pre-compiled with Blackwell kernel support. vLLM is built from source
# against this PyTorch to ensure sm_120 compatibility.
#
# Note: vLLM v0.15.x references Float8_e8m0fnu which is absent from the
# PyTorch 2.7.0a0 pre-release in NGC 25.03. We globally replace it with
# Float8_e4m3fn so the code compiles. The MXFP4 codepath is unreachable
# at runtime since the PyTorch dtype doesn't exist to produce matching
# tensors. A cmake hook patches fetched dependencies (qutlass) too.

FROM nvcr.io/nvidia/pytorch:25.03-py3

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Blackwell / sm_120 build configuration
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV VLLM_FLASH_ATTN_VERSION=2

ARG VLLM_VERSION=v0.15.1
ARG MAX_JOBS=4

# Install additional system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libxcb1 \
    libx11-6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Float8 compatibility patch scripts
COPY patch_float8.sh cmake_wrapper.sh /app/

# Remove NGC constraint files that pin packaging==23.2 (conflicts with vLLM)
RUN rm -f /opt/nvidia/pip_constraints.txt /opt/nvidia/entrypoint.d/50-pip-constraints.sh \
    && unset PIP_CONSTRAINT || true \
    && pip install --no-cache-dir "packaging>=24.2"

ENV PIP_CONSTRAINT=""

# Clone vLLM and apply Float8_e8m0fnu compatibility patches.
# The cmake wrapper intercepts cmake calls during pip install to patch
# fetched dependencies (qutlass etc.) after FetchContent downloads them.
RUN git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /tmp/vllm \
    && cd /tmp/vllm \
    && python use_existing_torch.py \
    && pip install --no-cache-dir -r requirements/build.txt \
    && pip install --no-cache-dir setuptools_scm \
    # Patch local vLLM source files
    && /app/patch_float8.sh /tmp/vllm \
    # Install cmake wrapper: move real cmake aside, symlink wrapper in its place
    && mv /usr/local/bin/cmake /usr/local/bin/cmake.real \
    && ln -s /app/cmake_wrapper.sh /usr/local/bin/cmake \
    && MAX_JOBS=${MAX_JOBS} pip install --no-cache-dir --no-build-isolation -e . \
    # Restore real cmake after build
    && rm -f /usr/local/bin/cmake \
    && mv /usr/local/bin/cmake.real /usr/local/bin/cmake \
    && cd /app \
    && rm -rf /tmp/vllm/.git

# Fix numpy ABI mismatch: vLLM pulls numpy 2.x but torch in NGC 25.03 was
# compiled against numpy 1.x. vLLM's logits processor calls .numpy() on
# tensors which crashes at runtime. Must be done AFTER vLLM install since
# vLLM's dependencies pull numpy 2.x and override any earlier pin.
RUN pip install --no-cache-dir "numpy<2"

# LMCache: hierarchical KV cache persistence (GPU → CPU → disk)
# Enables KV replay on wake instead of full re-prefill.
# lmcache's unpinned deps pull transformers>=5 and numpy 2.x — re-pin BOTH:
#   - transformers==4.57.6 (last 4.x): transformers 5 hard-breaks vLLM 0.15's
#     VL model paths at runtime (Qwen2VLImageProcessor.max_pixels removed;
#     strict config attr access kills `config.tie_word_embeddings`). An earlier
#     note here said the opposite — that 4.57.6 crashes on Qwen3-VL's tokenizer
#     (`'list' object has no attribute 'keys'`). Root cause of THAT is the
#     MODEL's tokenizer_config.json carrying a transformers-5-style
#     `extra_special_tokens` flat list; solved by the *-tf4shadow model dir
#     (see docker-compose.candidate.yml gaia-prime-candidate comments), so the
#     4.x pin is safe. lmcache 0.5.0's metadata wants transformers>=5.4 (pip
#     warns) but its runtime is verified working under 4.57.6; its optional
#     c_ops need CUDA 13 and fall back to torch ops on this CUDA 12.8 image.
#   - numpy<2 re-breaks otherwise (ABI pin above).
# (GAIA_Project-foh)
RUN pip install --no-cache-dir lmcache \
    && pip install --no-cache-dir "transformers==4.57.6" "numpy<2"

# Defensive forward-compat patch: vLLM reads image_processor.max_pixels /
# .min_pixels as bare attributes (removed in transformers 5); route through
# getattr with the size-dict fallback vLLM already half-uses. No-op under
# transformers 4.x (attrs exist), prevents an instant startup crash if the
# transformers pin ever drifts to 5.x again. vLLM is an editable install, so
# patching the /tmp/vllm source is live. (GAIA_Project-foh)
RUN sed -i \
      -e 's/min_pixels=image_processor.min_pixels,/min_pixels=getattr(image_processor, "min_pixels", None) or image_processor.size["shortest_edge"],/' \
      -e 's/max_pixels=image_processor.max_pixels,/max_pixels=getattr(image_processor, "max_pixels", None) or image_processor.size["longest_edge"],/' \
      -e 's/image_processor.max_pixels or image_processor.size\["longest_edge"\]/getattr(image_processor, "max_pixels", None) or image_processor.size["longest_edge"]/' \
      /tmp/vllm/vllm/model_executor/models/qwen2_vl.py \
    && grep -c 'getattr(image_processor' /tmp/vllm/vllm/model_executor/models/qwen2_vl.py

# Create non-root user for container execution (host UID mapping).
# NGC's 25.03 base image ships a default `ubuntu` user/group already at
# 1000 — remove it first so `gaia` can take that UID/GID (volume
# permissions on the host expect 1000). (GAIA_Project-foh)
RUN (userdel -r ubuntu 2>/dev/null || true) \
    && (groupdel ubuntu 2>/dev/null || true) \
    && groupadd -g 1000 gaia && useradd -u 1000 -g gaia -m -s /bin/bash gaia

# Set ownership for non-root execution
RUN chown -R gaia:gaia /app

# Expose service port
EXPOSE 7777

# Health check - Note: This is overridden by docker-compose healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:7777/health || exit 1

# Run as non-root user
USER gaia

# Default command includes LoRA and sleep mode flags for adapter + GPU handoff support.
# Sleep mode enables /sleep and /wake_up endpoints for VRAM↔RAM hot-swap.
# Overridden by docker-compose.candidate.yml command in production.
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/models/Claude", \
     "--host", "0.0.0.0", \
     "--port", "7777", \
     "--gpu-memory-utilization", "0.65", \
     "--max-model-len", "8192", \
     "--max-num-seqs", "4", \
     "--trust-remote-code", \
     "--dtype", "auto", \
     "--enable-lora", \
     "--max-loras", "4", \
     "--max-lora-rank", "64", \
     "--enable-sleep-mode"]
