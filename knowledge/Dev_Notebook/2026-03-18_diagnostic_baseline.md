# Diagnostic Baseline — March 18, 2026

## Watch Rotation Test

| Phase | Operation | Time |
|---|---|---|
| IDLE → FOCUSING | Core→CPU | 0.67s |
| | Nano→CPU | 0.52s |
| | VRAM freed | 13.2GB |
| | Prime load (int8) | ~23s |
| | Prime generate | ✓ |
| FOCUSING → IDLE | Core→GPU | 4.2s |
| | Nano→GPU | 0.16s |
| | KV pre-warm | 148ms |
| | Thought resumed | ✓ |
| **Full cycle** | | **~30 seconds** |

## Psych Eval Scores

| Tier | Model | Score | Speed | Notes |
|---|---|---|---|---|
| Core (2B) | Qwen3.5-2B-GAIA-Core-v3 | **10/10 (100%)** | 372ms | Identity-baked, perfect |
| Prime (8B) | Huihui-Qwen3-8B-abliterated-v2-merged | **5/10 (50%)** | ~20s (CPU) | Needs curriculum update |
| Nano (0.8B) | Qwen3.5-0.8B-Abliterated-merged | **3/10 (30%)** | 870ms | Needs identity training |

## Polygraph — Identity Neurons

| Tier | Dominant Neuron | L2 at deepest sampled layer | Pattern |
|---|---|---|---|
| Core (2B) | **neuron 1201** | 4.4 (layer 20) | Consistent across layers 4-20 |
| Prime (8B) | **neuron 1838** | 113.0 (layer 28) | Grows from 33.75→113.0 through depth |
| Nano (0.8B) | **neuron 0** | 1.9 (layer 23) | Weakest signal, needs training |

## Next Steps

1. QLoRA identity training for Nano (same v3 curriculum, 220 samples)
2. Prime curriculum update (add tier-role supplement)
3. SAE atlas training on all three tiers during sleep cycle
4. Post-training polygraph comparison
