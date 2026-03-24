"""
merge_and_requantize.py — Identity Baking & Quantization Pipeline

Produces deployment-ready models from bf16 base weights + optional LoRA adapters.

Pipeline stages:
  1. (Optional) Merge LoRA adapter into bf16 base weights
  2. AWQ quantization (GPU) — for Prime (vLLM serving)
  3. GGUF conversion + quantization (CPU) — for Core/Nano (llama_cpp serving)

Usage (inside gaia-study container):
  # Quantize all three tiers from base bf16 weights (no adapter merge)
  python -m gaia_study.merge_and_requantize \
      --prime-base /models/Qwen3.5-4B-Abliterated \
      --core-base  /models/Qwen3.5-4B-Abliterated \
      --nano-base  /models/Qwen3.5-0.8B-Abliterated \
      --output-dir /models

  # Merge adapter first, then quantize (identity baking)
  python -m gaia_study.merge_and_requantize \
      --prime-base /models/Qwen3.5-4B-Abliterated \
      --adapter /models/lora_adapters/tier1_global/gaia_persona_v1 \
      --output-dir /models

  # Single tier only
  python -m gaia_study.merge_and_requantize \
      --nano-base /models/Qwen3.5-0.8B-Abliterated \
      --output-dir /models --only nano
"""

import argparse
import gc
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── AutoAWQ compatibility shim ────────────────────────────────────────────
# AutoAWQ 0.2.9 imports PytorchGELUTanh which was removed in transformers 4.52+.
# Patch it back as an alias before awq is imported anywhere.
try:
    import transformers.activations as _act
    if not hasattr(_act, "PytorchGELUTanh"):
        _act.PytorchGELUTanh = _act.GELUActivation
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GAIA.Requantize")

try:
    from gaia_common.utils.memory_guard import require_memory
except ImportError:
    # Fallback: allow running outside the GAIA container stack
    require_memory = None  # type: ignore[assignment]

# ── Constants ────────────────────────────────────────────────────────────────

GGUF_CONVERT_SCRIPT = "/opt/llama.cpp/convert_hf_to_gguf.py"
LLAMA_QUANTIZE_BIN = "/usr/local/bin/llama-quantize"

# Default quantization types per tier
PRIME_QUANT = "gptq"       # GPTQ for vLLM serving (supports hybrid architectures)
CORE_QUANT = "Q4_K_M"     # GGUF 4-bit for CPU inference
NANO_QUANT = "Q8_0"       # GGUF 8-bit for fast CPU triage



# ── LoRA Merge ───────────────────────────────────────────────────────────────

def merge_adapter(base_path: str, adapter_path: str, output_path: str) -> str:
    """Merge a LoRA adapter into bf16 base weights.

    Returns the path to the merged model (either output_path or base_path
    if no adapter is provided).
    """
    if not adapter_path:
        logger.info("No adapter specified — using base model directly")
        return base_path

    adapter_config = Path(adapter_path) / "adapter_config.json"
    if not adapter_config.exists():
        # Fallback: search checkpoints for the latest adapter_config.json
        checkpoint_configs = sorted(
            Path(adapter_path).glob("checkpoints/*/adapter_config.json")
        )
        if checkpoint_configs:
            adapter_path = str(checkpoint_configs[-1].parent)
            adapter_config = checkpoint_configs[-1]
            logger.info("Adapter config not at top level — using checkpoint: %s", adapter_path)
        else:
            logger.warning("Adapter config not found at %s or checkpoints — skipping merge", adapter_config)
            return base_path

    # Pre-flight memory check: estimate from model file sizes
    # bf16 merge loads model to CPU RAM. Safetensors are already bf16, so
    # memory needed ≈ file size + ~30% overhead for tokenizer/merge ops.
    if require_memory is not None:
        base_p = Path(base_path)
        total_bytes = sum(f.stat().st_size for f in base_p.glob("*.safetensors"))
        if total_bytes == 0:
            total_bytes = sum(f.stat().st_size for f in base_p.glob("*.bin"))
        estimated_mb = max(int(total_bytes / (1024 * 1024) * 1.4), 4000)
        logger.info("Estimated memory for merge: %d MB (model files: %.1f GB)",
                     estimated_mb, total_bytes / (1024**3))
        require_memory(needed_mb=estimated_mb, label="LoRA merge (bf16 model load)")

    logger.info("═══ Merging LoRA adapter into base weights ═══")
    logger.info("  Base:    %s", base_path)
    logger.info("  Adapter: %s", adapter_path)
    logger.info("  Output:  %s", output_path)

    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    t0 = time.time()

    # Load base model in bf16 on CPU (uses ~8GB system RAM for 4B)
    # Use AutoModelForImageTextToText for multimodal models to preserve the
    # vision encoder through the merge. AutoModelForCausalLM strips it.
    logger.info("Loading base model (bf16, CPU)...")
    load_kwargs = dict(
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    try:
        cfg = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
        has_vision = hasattr(cfg, "vision_config") and cfg.vision_config is not None
    except Exception:
        has_vision = False

    if has_vision:
        logger.info("Detected multimodal model — loading with AutoModelForImageTextToText to preserve vision encoder")
        base_model = AutoModelForImageTextToText.from_pretrained(base_path, **load_kwargs)
    else:
        logger.info("Text-only model — loading with AutoModelForCausalLM")
        base_model = AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)

    # Load and merge adapter
    logger.info("Loading and merging adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()

    # Save merged model
    logger.info("Saving merged model to %s ...", output_path)
    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(output_path, safe_serialization=True)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # Copy any extra config files (generation_config, chat_template, etc.)
    for fname in ["generation_config.json", "chat_template.jinja"]:
        src = Path(base_path) / fname
        if src.exists():
            shutil.copy2(src, Path(output_path) / fname)

    elapsed = time.time() - t0
    logger.info("Merge complete in %.1fs", elapsed)

    # Free memory
    del merged, model, base_model
    gc.collect()

    return output_path


# ── GPTQ Quantization ────────────────────────────────────────────────────────

# GPTQ calibration config
GPTQ_BITS = 4
GPTQ_GROUP_SIZE = 128
GPTQ_CALIB_SAMPLES = 128
GPTQ_CALIB_SEQ_LEN = 512


def _register_qwen3_5():
    """Register a custom gptqmodel definition for Qwen3.5 (multimodal + text).

    Qwen3.5 is a multimodal model (Qwen3_5ForConditionalGeneration) with:
      - Vision encoder (Qwen3_5VisionModel) at model.visual
      - Text model (Qwen3_5TextModel) at model.language_model
      - Hybrid attention: linear_attn (24/32 layers) + self_attn (8/32 layers)
      - Dense MLP (gate_proj, up_proj, down_proj) — NOT MoE
      - linear_attn uses in_proj_qkv + in_proj_z (DeltaNet, not fused QKV)

    The critical issue: gptqmodel's default BaseQModel uses AutoModelForCausalLM,
    which creates Qwen3_5ForCausalLM. That class passes the composite Qwen3_5Config
    directly to Qwen3_5TextModel, which expects Qwen3_5TextConfig (with layer_types).
    This causes: AttributeError: 'Qwen3_5Config' has no attribute 'layer_types'.

    Fix: use AutoModelForImageTextToText as loader, which creates
    Qwen3_5ForConditionalGeneration → Qwen3_5Model, which correctly splits
    config.text_config and config.vision_config to their respective sub-models.

    Registered under both 'qwen3_5' (top-level model_type) and 'qwen3_5_text'
    (text sub-model model_type) so gptqmodel finds it regardless of config path.
    """
    try:
        from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
        from gptqmodel.models.base import BaseQModel
        from gptqmodel.utils.model import MODALITY

        if "qwen3_5" in MODEL_MAP:
            return  # Already registered

        from transformers import AutoModelForImageTextToText

        class Qwen3_5GPTQ(BaseQModel):
            """GPTQ definition for Qwen3.5 multimodal (hybrid attn + dense MLP)."""

            # Use multimodal loader to get Qwen3_5ForConditionalGeneration
            # (not ForCausalLM which breaks on nested text_config)
            loader = AutoModelForImageTextToText

            layer_modules_strict = False
            pre_lm_head_norm_module = "model.language_model.norm"
            require_trust_remote_code = True
            # Processor loading requires torchvision (for video processor).
            # We only calibrate with text, so skip processor auto-load.
            # If multimodal calibration is needed later, install torchvision
            # and set require_load_processor = True.
            require_load_processor = False

            modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

            module_tree = [
                "model",
                "language_model",
                "layers",
                "#",
                {
                    "input_layernorm": ("input_layernorm:!",),
                    # Linear attention (DeltaNet, 24/32 layers)
                    "linear_attn": (
                        "in_proj_qkv",
                        "in_proj_z",
                        "in_proj_a:!",  # tiny: (hidden_size, num_v_heads=32)
                        "in_proj_b:!",  # tiny: (hidden_size, num_v_heads=32)
                        "out_proj",
                    ),
                    # Full attention (standard QKV, 8/32 layers)
                    "self_attn": (
                        "q_norm:!",
                        "k_norm:!",
                        "q_proj:0",
                        "k_proj:0",
                        "v_proj:0",
                        "o_proj:1",
                    ),
                    "post_attention_layernorm": ("post_attention_layernorm:!",),
                    # Dense MLP (no MoE)
                    "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            ]

            # Note: no pre/post quantize hooks needed. gptqmodel's
            # offload-to-disk mechanism handles device placement for the
            # shell/turtle model pattern. Custom hooks that move embed_tokens
            # to GPU cause device mismatch (input_ids on CPU, weights on GPU)
            # during the cache_inputs forward pass.

        # Register under both model_type keys (MODEL_MAP + SUPPORTED_MODELS)
        MODEL_MAP["qwen3_5"] = Qwen3_5GPTQ
        MODEL_MAP["qwen3_5_text"] = Qwen3_5GPTQ
        if "qwen3_5" not in SUPPORTED_MODELS:
            SUPPORTED_MODELS.append("qwen3_5")
        if "qwen3_5_text" not in SUPPORTED_MODELS:
            SUPPORTED_MODELS.append("qwen3_5_text")
        logger.info("Registered custom gptqmodel definition for qwen3_5 / qwen3_5_text")
    except ImportError:
        logger.warning("gptqmodel not available — skipping qwen3_5 registration")


def quantize_prime(model_path: str, output_path: str) -> bool:
    """Quantize a bf16 HuggingFace model to GPTQ format for vLLM serving.

    Uses gptqmodel with a custom qwen3_5_text definition for Qwen3.5's
    hybrid architecture (linear_attn + self_attn + dense MLP).
    Requires GPU. For 4B models, uses ~6-8GB VRAM.
    """
    # Pre-flight memory check: GPTQ needs ~14GB system RAM + GPU VRAM
    if require_memory is not None:
        require_memory(needed_mb=14000, label="GPTQ quantization")

    logger.info("═══ GPTQ Quantization ═══")
    logger.info("  Input:  %s", model_path)
    logger.info("  Output: %s", output_path)

    try:
        from gptqmodel import GPTQModel, QuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("gptqmodel not installed. Run: pip install gptqmodel")
        return False

    # Register custom model definition before loading
    _register_qwen3_5()

    t0 = time.time()

    import torch
    if not torch.cuda.is_available():
        logger.error("No GPU available — GPTQ quantization requires CUDA")
        return False

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("GPU: %s (%.1f GB VRAM)", torch.cuda.get_device_name(0), gpu_mem)

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Configure GPTQ quantization
        # Note: lm_head=False because Qwen3.5 uses tied weights
        # (embed_tokens == lm_head) which GPTQ cannot quantize separately
        quant_config = QuantizeConfig(
            bits=GPTQ_BITS,
            group_size=GPTQ_GROUP_SIZE,
            lm_head=False,
        )

        # Load model for quantization (device_map="auto" to split across GPU/CPU
        # when model exceeds VRAM; gptqmodel handles layer-wise quantization)
        logger.info("Loading model for GPTQ quantization (device_map=auto)...")
        model = GPTQModel.load(
            model_path,
            quant_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Build calibration dataset (list of strings for gptqmodel)
        logger.info("Preparing calibration data (%d samples, seq_len=%d)...",
                     GPTQ_CALIB_SAMPLES, GPTQ_CALIB_SEQ_LEN)
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            calib_texts = [t for t in ds["text"] if len(t.strip()) > 100][:GPTQ_CALIB_SAMPLES]
        except Exception:
            logger.warning("Could not load wikitext; generating synthetic calibration data")
            calib_texts = [
                "The quick brown fox jumps over the lazy dog. " * 20
            ] * GPTQ_CALIB_SAMPLES

        # Quantize with calibration data
        logger.info("Quantizing (this takes ~5-10 min for 4B)...")
        model.quantize(
            calibration=calib_texts,
            calibration_concat_size=GPTQ_CALIB_SEQ_LEN,
            tokenizer=tokenizer,
        )

        # Save quantized model
        logger.info("Saving GPTQ model to %s ...", output_path)
        os.makedirs(output_path, exist_ok=True)
        model.save(output_path)
        tokenizer.save_pretrained(output_path)

    except Exception:
        raise

    # Copy extra config files
    for fname in ["generation_config.json", "chat_template.jinja"]:
        src = Path(model_path) / fname
        if src.exists() and not (Path(output_path) / fname).exists():
            shutil.copy2(src, Path(output_path) / fname)

    elapsed = time.time() - t0
    model_size = sum(
        f.stat().st_size for f in Path(output_path).rglob("*.safetensors")
    ) / (1024**3)
    logger.info("GPTQ quantization complete in %.1fs (%.2f GB)", elapsed, model_size)

    # Free GPU memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return True


# ── GGUF Conversion + Quantization ──────────────────────────────────────────

def convert_to_gguf(model_path: str, output_path: str, quant_type: str) -> bool:
    """Convert HuggingFace model to GGUF and quantize.

    Two-step process:
      1. convert_hf_to_gguf.py → F16 GGUF
      2. llama-quantize → target quantization (Q4_K_M, Q8_0, etc.)

    Runs entirely on CPU.
    """
    # Pre-flight memory check: GGUF conversion needs ~10GB for F16 intermediate
    if require_memory is not None:
        require_memory(needed_mb=10000, label="GGUF conversion")

    logger.info("═══ GGUF Conversion + Quantization ═══")
    logger.info("  Input:     %s", model_path)
    logger.info("  Output:    %s", output_path)
    logger.info("  Quant:     %s", quant_type)

    # Verify tools exist
    if not Path(GGUF_CONVERT_SCRIPT).exists():
        logger.error("convert_hf_to_gguf.py not found at %s", GGUF_CONVERT_SCRIPT)
        return False
    if not Path(LLAMA_QUANTIZE_BIN).exists():
        logger.error("llama-quantize not found at %s", LLAMA_QUANTIZE_BIN)
        return False

    t0 = time.time()

    # Derive filenames from model directory name
    model_name = Path(model_path).name
    output_dir = Path(output_path)
    os.makedirs(output_dir, exist_ok=True)

    f16_gguf = output_dir / f"{model_name}-f16.gguf"
    final_gguf = output_dir / f"{model_name}-{quant_type}.gguf"

    # Step 1: Convert HF → GGUF F16
    logger.info("Step 1: Converting HuggingFace → GGUF F16...")
    cmd_convert = [
        sys.executable, GGUF_CONVERT_SCRIPT,
        model_path,
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ]
    logger.info("  CMD: %s", " ".join(cmd_convert))

    result = subprocess.run(
        cmd_convert,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min timeout
    )
    if result.returncode != 0:
        logger.error("GGUF conversion failed:\n%s\n%s", result.stdout[-2000:], result.stderr[-2000:])
        return False
    logger.info("  F16 GGUF created: %s (%.2f GB)",
                f16_gguf, f16_gguf.stat().st_size / (1024**3))

    # Step 2: Quantize F16 → target type
    logger.info("Step 2: Quantizing F16 → %s ...", quant_type)
    cmd_quantize = [
        LLAMA_QUANTIZE_BIN,
        str(f16_gguf),
        str(final_gguf),
        quant_type,
    ]
    logger.info("  CMD: %s", " ".join(cmd_quantize))

    result = subprocess.run(
        cmd_quantize,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        logger.error("GGUF quantization failed:\n%s\n%s", result.stdout[-2000:], result.stderr[-2000:])
        return False

    final_size = final_gguf.stat().st_size / (1024**3)
    logger.info("  Quantized GGUF created: %s (%.2f GB)", final_gguf, final_size)

    # Clean up F16 intermediate (large file)
    logger.info("Cleaning up intermediate F16 GGUF...")
    f16_gguf.unlink()

    elapsed = time.time() - t0
    logger.info("GGUF pipeline complete in %.1fs", elapsed)

    return True


# ── Full Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> bool:
    """Run the full quantization pipeline based on CLI args."""
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tiers_requested = set()
    if args.only:
        tiers_requested = set(args.only)
    else:
        if args.prime_base:
            tiers_requested.add("prime")
        if args.core_base:
            tiers_requested.add("core")
        if args.nano_base:
            tiers_requested.add("nano")

    if not tiers_requested:
        logger.error("No model bases specified. Use --prime-base, --core-base, --nano-base")
        return False

    results = {}

    # ── Prime (GPTQ) ─────────────────────────────────────────────────────
    if "prime" in tiers_requested and args.prime_base:
        logger.info("\n" + "=" * 70)
        logger.info("PRIME TIER: %s → GPTQ", args.prime_base)
        logger.info("=" * 70)

        # Merge adapter if specified
        source = args.prime_base
        if args.adapter:
            merged_path = str(output_dir / f"{Path(args.prime_base).name}-merged")
            source = merge_adapter(args.prime_base, args.adapter, merged_path)

        gptq_output = str(output_dir / f"{Path(args.prime_base).name}-GPTQ")
        results["prime"] = quantize_prime(source, gptq_output)

        # Clean up merged intermediate if we created one
        if args.adapter and source != args.prime_base and Path(source).exists():
            if not args.keep_merged:
                logger.info("Cleaning up merged intermediate: %s", source)
                shutil.rmtree(source)
            else:
                logger.info("Keeping merged bf16 at: %s", source)

    # ── Core (GGUF Q4_K_M) ──────────────────────────────────────────────
    if "core" in tiers_requested and args.core_base:
        logger.info("\n" + "=" * 70)
        logger.info("CORE TIER: %s → GGUF %s", args.core_base, CORE_QUANT)
        logger.info("=" * 70)

        # Merge adapter if specified (separate training for smaller models)
        source = args.core_base
        if args.core_adapter:
            merged_path = str(output_dir / f"{Path(args.core_base).name}-merged")
            source = merge_adapter(args.core_base, args.core_adapter, merged_path)

        results["core"] = convert_to_gguf(source, str(output_dir), CORE_QUANT)

        if args.core_adapter and source != args.core_base and Path(source).exists():
            if not args.keep_merged:
                shutil.rmtree(source)

    # ── Nano (GGUF Q8_0) ────────────────────────────────────────────────
    if "nano" in tiers_requested and args.nano_base:
        logger.info("\n" + "=" * 70)
        logger.info("NANO TIER: %s → GGUF %s", args.nano_base, NANO_QUANT)
        logger.info("=" * 70)

        source = args.nano_base
        if args.nano_adapter:
            merged_path = str(output_dir / f"{Path(args.nano_base).name}-merged")
            source = merge_adapter(args.nano_base, args.nano_adapter, merged_path)

        results["nano"] = convert_to_gguf(source, str(output_dir), NANO_QUANT)

        if args.nano_adapter and source != args.nano_base and Path(source).exists():
            if not args.keep_merged:
                shutil.rmtree(source)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    for tier, success in results.items():
        status = "OK" if success else "FAILED"
        logger.info("  %-8s %s", tier.upper(), status)

    # List output files
    logger.info("\nOutput files:")
    for p in sorted(output_dir.iterdir()):
        if p.suffix in (".gguf",) or p.is_dir():
            size = p.stat().st_size / (1024**3) if p.is_file() else sum(
                f.stat().st_size for f in p.rglob("*") if f.is_file()
            ) / (1024**3)
            logger.info("  %-50s %.2f GB", p.name, size)

    return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="GAIA Identity Baking & Quantization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model bases (bf16 HuggingFace format)
    parser.add_argument("--prime-base", type=str,
                        help="Path to Prime bf16 model (e.g., /models/Qwen3.5-4B-Abliterated)")
    parser.add_argument("--core-base", type=str,
                        help="Path to Core bf16 model (e.g., /models/Qwen3.5-4B-Abliterated)")
    parser.add_argument("--nano-base", type=str,
                        help="Path to Nano bf16 model (e.g., /models/Qwen3.5-0.8B-Abliterated)")

    # Adapter paths (optional — for identity baking)
    parser.add_argument("--adapter", type=str, default=None,
                        help="LoRA adapter to merge into Prime base before quantizing")
    parser.add_argument("--core-adapter", type=str, default=None,
                        help="LoRA adapter to merge into Core base before quantizing")
    parser.add_argument("--nano-adapter", type=str, default=None,
                        help="LoRA adapter to merge into Nano base before quantizing")

    # Output
    parser.add_argument("--output-dir", type=str, default="/models",
                        help="Directory for quantized output (default: /models)")

    # Tier selection
    parser.add_argument("--only", nargs="+", choices=["prime", "core", "nano"],
                        help="Only run specified tiers (default: all specified bases)")

    # Options
    parser.add_argument("--keep-merged", action="store_true",
                        help="Keep merged bf16 intermediate (useful as new training base)")

    args = parser.parse_args()
    success = run_pipeline(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
