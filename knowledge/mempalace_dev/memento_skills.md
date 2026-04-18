# 🧠 GAIA Development Memento — Procedural Skill Graph

> This file tracks successful development patterns, surgical implementation skills, and "Golden Rules" for the GAIA Chord. Check this before high-risk mutations to avoid known regressions.

---

## 🛠️ Skill: Surgical MoE Migration
**Context:** Loading large MoE models (26B-A4B) on consumer-tier VRAM (16GB).

*   **The Pattern:** NEVER use `model.to("cuda")` or `accelerate` dispatch for the entire model. It will OOM.
*   **Successful Action:**
    1. Load model in `bf16` on CPU first.
    2. Iterate through `model.named_parameters()`.
    3. Manually migrate foundation parameters (Attention, Norms, Embeddings) via `param.data = param.data.to("cuda")`.
    4. Keep `experts` on CPU for JIT 3D-slicing.
*   **Lesson:** Manual parameter migration bypasses the "Ghost Migration" OOM where PyTorch tries to allocate double memory during a full module move.

## 🌉 Skill: Neural Bridge (Autograd Firewall)
**Context:** Training a model with CPU-resident weights without breaking gradient flow.

*   **The Pattern:** Wrap non-resident modules in a `torch.autograd.Function`.
*   **Successful Action:** 
    1. `forward`: Compute output inside `torch.no_grad()` to protect VRAM. Save only small metadata (like router logits) in `ctx`.
    2. `backward`: Approximate the gradient pass-through using the saved metadata (e.g., scaling `grad_output` by router weights).
*   **Lesson:** This "Firewall" prevents PyTorch from attempting to build a compute graph for 128 frozen CPU experts, keeping VRAM stable at ~4.5GB.

## 🔀 Skill: Chord Synchronization
**Context:** Multi-agent collaboration between Advisor (Gemini) and Engineer (Claude).

*   **The Pattern:** Use the `watchdog` and `chord_send.sh` for high-fidelity alignment.
*   **Successful Action:**
    1. Keep `COUNCIL_CHAMBER.md` updated with AAAK strategy entries.
    2. Use `chord_send.sh <from> <to> "<message>"` for direct tactical handoffs.
    3. Ensure the `watchdog` triggers `flatten_soa.sh` so NotebookLM (Long-Term Memory) is never more than 5 minutes out of date.
*   **Lesson:** Explicit AAAK signaling prevents "Phase Drift" where agents start working on conflicting versions of the plan.

## 📜 Golden Rules (Development Constraints)
1.  **Identity is Shared:** Foundation Tuning must target `self_attn` and `shared_mlp` ONLY. Leave sparse specialists untouched.
2.  **VRAM is Sovereign:** 16GB is the hard ceiling. If an operation exceeds 12GB, it must be offloaded or bridged.
3.  **Gemma is the Standard:** Purge all references to Huihui/Qwen in active Phase 5 config.
