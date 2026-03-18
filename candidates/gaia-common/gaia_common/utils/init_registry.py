"""Initialize the model registry with current GAIA model state."""

from gaia_common.utils.model_registry import ModelRegistry, ModelEntry


def initialize_default_registry():
    """Populate registry with current model inventory."""
    registry = ModelRegistry()

    registry.register(ModelEntry(
        name="nano",
        role="reflex",
        base_model="Qwen3.5-0.8B",
        safetensors_path="/models/Qwen3.5-0.8B-Abliterated-merged",
        gguf_path="/models/Qwen3.5-0.8B-Abliterated-Q8_0.gguf",
        gguf_quantization="Q8_0",
        parent="Qwen3.5-0.8B-Abliterated",
        training_tier="I",
        notes="Identity-baked Nano. Abliterated base (legacy — future: Qwen3.5-0.8B-Base).",
    ))

    registry.register(ModelEntry(
        name="core",
        role="operator",
        base_model="Qwen3.5-2B",
        safetensors_path="/models/Qwen3.5-2B-GAIA-Core",
        gguf_path="",  # TODO: derive GGUF from merged safetensors
        gguf_quantization="Q8_0",
        parent="Qwen3.5-2B",
        training_tier="I",
        notes="Tier I identity-baked Core. Post-trained Qwen3.5-2B base, clean geometry.",
    ))

    registry.register(ModelEntry(
        name="prime",
        role="thinker",
        base_model="Huihui-Qwen3-8B-abliterated-v2",
        safetensors_path="/models/Huihui-Qwen3-8B-abliterated-v2-merged",
        gguf_path="",  # Prime doesn't need GGUF (GPU-only via vLLM)
        parent="Huihui-Qwen3-8B-abliterated-v2",
        training_tier="I",
        notes="Identity-baked Prime. Abliterated 8B, fp8 via vLLM.",
    ))

    print(f"Registry initialized with {len(registry.list_models())} models")
    for name, entry in registry.list_models().items():
        print(f"  {name} ({entry.role}): {entry.safetensors_path} v{entry.version}")


if __name__ == "__main__":
    initialize_default_registry()
