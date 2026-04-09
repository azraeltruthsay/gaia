Run a GAIA LoRA training session: $ARGUMENTS

## Overview

This skill handles the full training lifecycle — GPU preparation, training execution, adapter verification, merge, and system restoration. Follow EVERY step. Skipping steps causes OOM, stale state, or broken deployments.

## Pre-Training Checklist (MANDATORY)

1. **Disable restart policies** (prevents containers from respawning and stealing GPU):
   ```bash
   docker update --restart=no gaia-core gaia-nano gaia-audio gaia-prime
   ```

2. **Stop GPU containers**:
   ```bash
   docker stop gaia-core gaia-nano gaia-audio gaia-prime
   ```

3. **Wait 15-30 seconds**, then verify GPU is clear:
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
   ```
   Must show <1GB used. Do NOT proceed until confirmed.

4. **ThreadPoolExecutor patch** — the adaptive training pipeline handles this internally. If running manual training, ensure the monkey-patch is applied before any model imports.

5. Verify gaia-study is running: `curl -s http://localhost:8766/health`

## Training Execution

### Via Adaptive Pipeline (Preferred)
```bash
curl -s -X POST http://localhost:8766/study/adaptive-train \
  -H 'Content-Type: application/json' \
  -d '{
    "adapter_name": "<NAME>",
    "base_model": "<MODEL_PATH>",
    "train_data_path": "<CURRICULUM_PATH>",
    "max_phases": 3,
    "pass_threshold": 0.7,
    "rank": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "out_proj"],
    "max_steps_phase1": 200,
    "max_steps_repair": 100,
    "training_timeout": 1800
  }'
```

### Monitor
```bash
curl -s http://localhost:8766/study/adaptive-train/status | python3 -m json.tool
```

### Critical Training Rules
- **BitsAndBytes NF4 ONLY** — never quanto (wraps only 25% of layers)
- **All 9 target modules** — attention + MLP + linear attention. MLP is where identity lives. Linear attention is 75% of Qwen3.5 layers.
- **Train data format**: JSON array of `{"instruction": "...", "output": "..."}` objects
- For 9B models: may need `training_timeout: 3600` and `max_steps_phase1: 300`

## Post-Training Checklist (MANDATORY)

1. **Verify adapter** — check layer coverage in the output adapter directory
2. **Merge into base** (if needed):
   ```python
   from peft import PeftModel
   model = PeftModel.from_pretrained(base_model, adapter_path)
   merged = model.merge_and_unload()
   merged.save_pretrained(output_dir)
   ```
3. **Verify merge changed weights** — md5sum comparison of base vs merged safetensors
4. **Test identity** — direct engine call asking "Who are you?"
5. **Update symlinks** — `/models/core` → new merged dir, etc.
6. **Re-enable restart and start containers**:
   ```bash
   docker update --restart=unless-stopped gaia-core gaia-nano gaia-audio gaia-prime
   docker start gaia-core gaia-nano gaia-audio gaia-prime
   ```
7. **Verify system restored** — health checks on all tiers, consciousness matrix ok:true

## Known Failure Modes
- **quanto fallback**: logs show "quanto int4" → STOP, BnB path wasn't taken
- **Containers respawning**: forgot to disable restart → OOM during training
- **Attention-only targets**: identity doesn't bake without MLP modules
- **transformers 5.3.0 OOM**: ThreadPoolExecutor parallel loading spike → monkey-patch required
