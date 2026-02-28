"""
Quantize huihui-ai/Huihui-Qwen3-8B-abliterated-v2 to AWQ 4-bit
using AutoAWQ.  AutoAWQ does true layer-by-layer GPU calibration,
keeping only one layer on the GPU at a time — safe for 16 GB cards.

Run in a container with GPU access:
  docker run --gpus all --ipc=host \
    -v /gaia/GAIA_Project/gaia-models:/models \
    --rm <image> \
    bash -c "pip install autoawq && python3 /models/staging/quantize_awq.py"
"""

import os
import time

MODEL_DIR = "/models/staging/Huihui-Qwen3-8B-abliterated-v2"
OUTPUT_DIR = "/models/staging/Huihui-Qwen3-8B-abliterated-v2-AWQ"


def main():
    t0 = time.time()

    print("[1/4] Loading model for AWQ quantization...")
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model = AutoAWQForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("[2/4] Running AWQ quantization (W4, group_size=128)...")
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    model.quantize(tokenizer, quant_config=quant_config)

    print(f"[3/4] Saving to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_quantized(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    elapsed = time.time() - t0

    # Show stats
    total = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if os.path.isfile(os.path.join(OUTPUT_DIR, f))
    )
    print(f"\n[4/4] Done!")
    print(f"  Output size: {total / (1024**3):.2f} GB")
    print(f"  Elapsed: {elapsed / 60:.1f} minutes")
    print(f"  Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
