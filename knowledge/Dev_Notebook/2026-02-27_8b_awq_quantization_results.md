# 8B AWQ Quantization & Model Comparison Results
**Date:** 2026-02-27

## Quantization

**Source:** `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` (BF16, 16 GB, 36 layers)
**Output:** `Qwen3-8B-abliterated-AWQ` (W4A16 GEMM, 5.7 GB)
**Tool:** AutoAWQ 0.2.9 + transformers 4.51.3
**Time:** 8.6 minutes on RTX 5080 (layer-by-layer calibration, 13.5s/layer)
**Config:** zero_point=True, group_size=128, w_bit=4, version=GEMM

### Failed approaches (llm-compressor)
- llm-compressor 0.9.0.2's SequentialPipeline OOM'd on 16GB GPU
- Root cause: `dispatch_for_sequential()` uses accelerate hooks that move entire model to GPU
- `disable_offloading()` context manager prevents CPU offload during calibration
- 4 attempts failed (device_map variations, monkey-patching) — all OOM at 13-14 GiB

### AutoAWQ workaround
- AutoAWQ deprecated but does true layer-by-layer quantization natively
- Required pinning `transformers==4.51.3` (PytorchGELUTanh removed in 4.52+)
- vLLM auto-converts AWQ GEMM → AWQ Marlin at load time for faster inference

## vLLM Deployment

- **VRAM:** 12.5 GiB (model: 7.0 GiB + KV cache: ~5.5 GiB)
- **KV cache slots:** 1,607 (cross-layer, 36 layers × 8 KV heads × 128 dim)
- **Quantization kernel:** awq_marlin (auto-detected by vLLM)
- **Context window:** 8,192 tokens (configurable up to 40,960)

## Comparison Battery (16 tests, 6 categories)

| Metric | 4B-heretic | 8B-abliterated-AWQ |
|--------|------------|-------------------|
| Total time | 532.9s | 445.8s (**-16%**) |
| Avg per test | 33.3s | 27.9s |
| Loop detections | 7/16 | 6/16 |
| Substantive responses | 9/16 | 10/16 |

### 8B wins
- **Short story:** 4B hallucinated Wikipedia; 8B wrote original atmospheric story
- **Technical knowledge:** 4B looped; 8B gave full mutex/semaphore/spinlock explanation (2,897 chars)
- **D&D mechanics:** Observer flagged 4B's incorrect grapple rule; 8B was accurate

### Both models (system-level issues)
- Tests 11-16 (persona/epistemic/safety) hit loop detection on both — observer pipeline issue, not model
- Structured output: Both triggered MCP tool routing instead of answering directly
- Poetry: Both hallucinated existing poems

### Conclusion
8B AWQ is a clear upgrade: faster inference AND better quality. Deployed as new default in docker-compose.yml.

## Warm Pool Change
- Removed `Qwen3-4B-Instruct-2507-heretic` from `/mnt/gaia_warm_pool`
- Copied `Qwen3-8B-abliterated-AWQ` (5.7 GB) to warm pool
- Updated `docker-compose.yml` defaults: `PRIME_MODEL_PATH` and `PRIME_MODEL`
