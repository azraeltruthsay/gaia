# Dev Journal - February 1, 2026

## Subject: Debugging the Modular GAIA Stack Startup

### Summary

This week was dedicated to bringing the new modular, service-oriented GAIA architecture online. The process was fraught with a series of cascading failures that took us deep into the build system, dependency management, and low-level model execution. After a lengthy and complex debug cycle, all services are now stable, and the primary test script (`new_gaia_test.sh`) completes successfully.

### Key Issues & Resolutions

The initial attempt to start the services failed immediately due to the `gaia-mcp` container being unhealthy. This kicked off a long chain of problem discovery and resolution:

1.  **Module Not Found Errors**: The first issue was a `ModuleNotFoundError` for `gaia_common.integrations`. This was a simple fix: the new `integrations` directory was missing an `__init__.py` file, preventing it from being recognized as a Python package.

2.  **Stale Docker Builds**: The fix for the missing module didn't seem to work at first. We discovered that the `new_gaia_start.sh` script was not using the `--build` flag with `docker compose up`, so the container images were never being updated with the code changes.

3.  **Dependency Management Hell (`pip-compile`)**: Once builds were running, we found that core dependencies like `discord.py`, `llama-cpp-python`, and `sentence-transformers` were missing from the `gaia-core` container. The project convention is to use `pip-compile` on `pyproject.toml` to generate `requirements.txt`. However, this toolchain proved to be broken within the containerized build environment.
    *   **Resolution**: After multiple failed attempts to use `pip-compile`, we resorted to manually constructing a stable `requirements.txt` file containing all necessary dependencies.

4.  **`llama-cpp-python` Build Failure**: With a correct `requirements.txt`, the build failed again. `llama-cpp-python` requires a C++ compiler, which was not present in the minimal `gaia-mcp` and `gaia-core` Docker images.
    *   **Resolution**: We added the `build-essential` package to the `apt-get install` commands in both Dockerfiles.

5.  **The `vLLM` Segfault**: After fixing the build, the test run would crash with a segmentation fault when loading the `gpu_prime` model via `vLLM`. The crash was traced to `torch.compile` (Dynamo).
    *   **Resolution**: We disabled `torch.compile` globally by adding the environment variable `TORCH_COMPILE_DISABLE=1` to the `gaia-core` service in `docker-compose.yml`, which stabilized the `vLLM` engine.

6.  **Model Context Length Exceeded**: With `vLLM` stable, the RAG pipeline started working, but the resulting prompt (including the retrieved knowledge) was too long for the model's default configuration (2954 > 2048 tokens).
    *   **Resolution**: We increased the `max_model_len` for `gpu_prime` to `8192` in `gaia_constants.json` to handle large, RAG-enhanced prompts.

7.  **CUDA Out of Memory**: The final hurdle was a `CUDA out of memory` error. The system was attempting to load both the large `gpu_prime` model and the `sentence-transformers` embedding model onto the same GPU, which exhausted the VRAM.
    *   **Resolution**: We forced the `sentence-transformers` model to run on the CPU by modifying `vector_indexer.py` to load it with `device='cpu'`. This separated the workloads, leaving the GPU dedicated to the `gpu_prime` model.

### Current Status

With all the above fixes, the `new_gaia_test.sh` script now consistently runs to completion, producing the expected RAG-enhanced response for "Rupert Roads" from the `gpu_prime` model without crashing or timing out. The system correctly identifies the user's intent, uses the RAG pipeline to retrieve relevant documents, and generates a correct, context-aware response. The modular GAIA stack is now stable and fully functional.