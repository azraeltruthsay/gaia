# GAIA Core Service GPU Configuration Blueprint

## Role of `gaia-core` in GPU Utilization

The `gaia-core` service is responsible for the primary cognitive processes of GAIA, which heavily rely on Large Language Models (LLMs). When configured with `gpu_prime` or `prime` models, `gaia-core` directly loads and serves vLLM models using GPU resources. Its successful operation demonstrates a working pattern for Dockerized GPU access within the GAIA ecosystem.

## Key Configuration for GPU Access

The successful GPU utilization by `gaia-core` is a result of a combination of its Dockerfile base image, Docker Compose `deploy` configuration, and specific environment variables.

### 1. `gaia-core/Dockerfile`

The `Dockerfile` for `gaia-core` is built upon an NVIDIA CUDA base image, which pre-installs the necessary CUDA toolkit and drivers *within the container environment*. This is crucial for applications inside the container to interact with the host's GPU.

```dockerfile
# gaia-core: The Brain - Cognitive loop and reasoning engine
# GPU-enabled container for model inference

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# ... (other Python and system dependencies installation) ...

# Install Python dependencies, including GPU dependencies (e.g., vLLM, PyTorch)
RUN pip install --no-cache-dir -r requirements.txt -r requirements-gpu.txt

# ... (rest of the Dockerfile) ...
```

**Key takeaway**: By using `FROM nvidia/cuda:12.4.0-devel-ubuntu22.04`, `gaia-core` ensures that its container environment is natively equipped with compatible CUDA libraries for the host GPU.

### 2. `docker-compose.yml` (for `gaia-core` service)

The Docker Compose configuration for `gaia-core` explicitly tells the Docker daemon to allocate and make available GPU resources to the container. This is achieved through the `deploy` section, specifically `resources.reservations.devices`.

```yaml
  gaia-core:
    # ... (other service configurations) ...

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all # Requests all available NVIDIA GPUs
              capabilities: [gpu] # Ensures the container has GPU capabilities

    environment:
      # ... (other environment variables) ...

      # GPU settings
      - N_GPU_LAYERS=${N_GPU_LAYERS:-8} # Specific to Llama.cpp, but can affect vLLM behavior
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} # Explicitly selects GPU device 0
      - VLLM_DISABLE_CUSTOM_CUDA_MODULES=1 # vLLM specific setting
      - GAIA_VLLM_SAFE_MODE=0 # Custom GAIA safety setting for VRAM
      - GAIA_VLLM_WORKER_METHOD=spawn # Force spawn for vLLM to avoid CUDA fork issues
      - VLLM_WORKER_MULTIPROC_METHOD=spawn # vLLM specific setting for multiprocess workers
      - TORCH_COMPILE_DISABLE=1 # Workaround for potential segfaults with PyTorch/vLLM

    # ... (rest of the service configuration) ...
```

**Key takeaways**:
*   `deploy.resources.reservations.devices`: This is the standard way to expose host GPUs to Docker containers using the NVIDIA Container Toolkit.
*   `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}`: This environment variable is often crucial for CUDA applications to correctly identify and use specific GPUs when multiple are present, or to explicitly target the default GPU.
*   `VLLM_WORKER_MULTIPROC_METHOD=spawn`: This addresses potential issues with vLLM's multiprocessing and CUDA.

## Comparison with `gaia-prime-candidate`

The `gaia-prime-candidate` service, as defined in `docker-compose.candidate.yml`, also uses the `deploy.resources.reservations.devices` section, but with `count: 1` instead of `all`. It also sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

The primary difference that likely contributes to the `Error 803` is that `gaia-prime-candidate`'s Docker image (`vllm/vllm-openai:latest`) is external. While it's expected to be CUDA-enabled, its internal CUDA toolkit version might mismatch the host's NVIDIA drivers, or it might implicitly require additional environment setup that `gaia-core` explicitly provides through its `nvidia/cuda` base image and detailed environment variables.

The `Error 803: system has unsupported display driver / cuda driver combination` points to a fundamental incompatibility between the CUDA libraries within the `vllm/vllm-openai:latest` image and the host's NVIDIA drivers. This can occur if the host driver is too old/new for the CUDA version compiled into the Docker image.

## Troubleshooting Implications

Given that `gaia-core` successfully initializes CUDA, the host's NVIDIA Container Toolkit is likely working to some extent. The problem with `gaia-prime-candidate` strongly suggests a version mismatch between the host's NVIDIA drivers and the CUDA toolkit *baked into the `vllm/vllm-openai:latest` image*.

**Next steps should focus on:**
1.  Determining the CUDA version within the `vllm/vllm-openai:latest` image.
2.  Comparing it against the host's `nvidia-smi` output and `gaia-core`'s known working CUDA 12.4.0 environment.
3.  Adjusting the `vllm/vllm-openai` image tag if a compatibility issue is identified.
