# GAIA Bootstrap Install — Self-Tuning Distribution Blueprint

**Status:** Concept / Not Scheduled
**Recorded:** 2026-02-28
**Author:** Azrael

---

## Problem Statement

GAIA is currently configured for a specific hardware stack (GPU VRAM, model paths, quantization settings, Docker volume layout, etc.). Distributing GAIA to other users means they'd face a non-trivial tuning problem: wrong quantization level, mismatched model size for available VRAM, incorrect constants, paths that don't exist on their system. The goal is to make GAIA capable of bootstrapping herself onto a new system with minimal human intervention.

---

## Core Idea

A **bootstrap install process** that:

1. Requires no local models to be loaded at install time
2. Uses a **Groq Free API key** (provided by the user during install) to power a cloud-hosted Oracle
3. Routes that Oracle through GAIA's existing **Oracle pipeline** to issue commands, inspect the system, and make configuration decisions
4. Allows the Oracle to run GAIA's own tooling (quantization scripts, GGUF conversion, config writes) within a defined permission boundary
5. Results in a GAIA installation that is correctly tuned for the target system — selected model, quantization level, config constants — without requiring the user to understand the internals

---

## Flow Sketch

```
User clones GAIA repo
  └─ Runs: ./bootstrap_install.sh

Bootstrap prompts:
  ├─ Enter Groq API key (free tier, no local model needed)
  ├─ Accept install scope (what Oracle is allowed to do)
  └─ Basic hardware survey (GPU, VRAM, disk space, CPU)

Bootstrap launches Oracle via Groq (no gaia-prime needed)
  └─ Oracle receives: hardware survey + GAIA repo layout + available models

Oracle decision loop (via Oracle pipeline):
  ├─ Assess available VRAM → select appropriate base model
  ├─ Assess disk space → decide on quantization level (Q4, Q5, Q8, AWQ, etc.)
  ├─ If needed: download abliterated model → run quantization script
  ├─ If needed: run GGUF conversion script (llama.cpp or equivalent)
  ├─ Write gaia_constants.json + config files for target hardware
  ├─ Set Docker Compose env vars (VRAM limits, model paths, etc.)
  └─ Verify: attempt to start gaia-prime → confirm model loads → report success

Install completes:
  └─ GAIA running on target system, tuned by Oracle
```

---

## Key Components Required

### 1. Groq-backed Oracle Bootstrap Mode
- A lightweight Oracle client mode that authenticates via Groq API key (no local model)
- Uses GAIA's existing Oracle pipeline structure but substitutes Groq as the LLM backend
- Groq Free tier models (e.g. Llama 3.1 8B, Mixtral 8x7B) are sufficient for config reasoning

### 2. Hardware Survey Script
- GPU model, total VRAM, available VRAM
- Disk space on model target path
- CPU core count / RAM (for GGUF CPU fallback assessment)
- Existing model files (detect what's already cached)
- OS / Docker version

### 3. Bounded Permission Scope for Bootstrap Oracle
- Oracle may: read system info, write config files, run scripts from an allowlist
- Oracle may NOT: access network beyond model download endpoints, modify non-GAIA paths, run arbitrary shell commands
- Allowlist includes: quantization script, GGUF conversion script, gaia_constants.json writer, Docker Compose env writer

### 4. Model Selection Logic
- Decision table: VRAM → model family + quant level
  - < 4GB: GGUF Q4_K_M on CPU (Phi-3 Mini or similar)
  - 4–6GB: Qwen 1.5B / Gemma 2B AWQ
  - 6–10GB: Qwen 3 4B AWQ (current GAIA default)
  - 10–16GB: Qwen 3 7B AWQ or Llama 3.1 8B AWQ
  - 16–24GB: Qwen 3 14B or Llama 3.1 13B
  - 24GB+: Qwen 3 32B / Mistral Large AWQ
- Abliterated model variants preferred when available

### 5. Quantization & Conversion Scripts (already partially exist)
- AWQ quantization: `scripts/quantize_awq.py` (exists, needs integration)
- GGUF conversion: `llama.cpp convert_hf_to_gguf.py` (standard tooling)
- Bootstrap Oracle calls these as subprocesses within its permission scope

### 6. Config Writer
- Writes `gaia-core/config/gaia_constants.json` with tuned values
- Writes `docker-compose.override.yml` with correct model path, VRAM limits, service profiles
- Optionally writes `.env.bootstrap` for the target system

---

## Open Questions (to resolve at implementation time)

- **Groq tier limits**: Free tier has rate limits — quantization decisions are reasoning-heavy. Need to be efficient with tokens, possibly cache the hardware survey result locally and resume if rate-limited.
- **Model download trust**: Should bootstrap download models from HuggingFace automatically, or only from a curated allowlist of repos? Probably allowlist for safety.
- **GGUF vs AWQ**: vLLM (gaia-prime) prefers AWQ; GGUF is for llama.cpp fallback. Bootstrap should detect which inference engine the target system will use.
- **Groq key persistence**: After bootstrap, should the Groq key be retained as an Oracle fallback (for when gaia-prime is offline), or discarded? This connects to the existing Oracle fallback system.
- **Reproducibility**: Bootstrap should write a `bootstrap_manifest.json` capturing every decision made, so the install can be reproduced or audited.
- **Multi-GPU**: Out of scope for v1; document the limitation.

---

## Security Model

This section captures security requirements that must be designed in from the start — not bolted on later. Several of these are explicitly about preventing the **developer themselves** from having privileged access to user instances.

### Threat Model

| Threat | Description |
|---|---|
| Prompt injection via survey | Attacker crafts hostname, path, or environment value that gets passed to Groq and manipulates Oracle behavior |
| Bootstrap script tampering | Attacker modifies bootstrap script before user runs it (supply chain, MITM on git clone) |
| Groq API key theft | User's Groq key is logged, exfiltrated, or persisted in plaintext |
| Developer backdoor | Developer encodes a "phone home" endpoint or hidden credential that gives them access to installed instances |
| Oracle scope escape | Oracle issues commands outside its allowlist (arbitrary shell, network access, etc.) |
| GAIA principle violation at birth | Bootstrap Oracle is jailbroken or given a system prompt that bypasses GAIA's normal ethical constraints |
| Persistent elevated privilege | Bootstrap process leaves residual elevated access or credentials after install completes |

---

### Mitigations

#### 1. Prompt Injection Defense
- **All hardware survey values must be treated as untrusted data.** Pass them to Groq as structured JSON, never interpolated directly into freeform prompt text.
- Survey fields must be validated and sanitized before inclusion: GPU name, paths, hostnames are strings with strict allowed character sets (alphanumeric + common path chars).
- The Groq system prompt should explicitly instruct the Oracle to treat survey data as data, not instructions.
- No user-provided freeform text (e.g. "custom model path notes") should be passed to Groq at all.

#### 2. Bootstrap Script Integrity
- Ship a `bootstrap_install.sh.sha256` checksum alongside the script.
- Recommend (or enforce) that users verify the checksum before running.
- If GAIA is distributed via a signed git tag, document how to verify the signature.
- Bootstrap should never fetch additional script logic from a remote URL at runtime — all code runs from the cloned repo, statically.

#### 3. Groq API Key Handling
- Key is entered via stdin with echo disabled (never visible in terminal history).
- Never written to disk in plaintext — not to `.env`, not to any log file.
- Held only in process memory for the duration of bootstrap.
- If retained for Oracle fallback use (see Open Questions), it must be stored in a secrets store or encrypted file, not a plaintext config.
- Bootstrap should explicitly warn the user to treat the Groq key as a credential, not share it, and to revoke it if the install system is compromised.

#### 4. Developer Backdoor Prevention (explicit design-out)
- Bootstrap script must contain **no hardcoded remote endpoints** other than Groq's official API and HuggingFace model repos.
- No telemetry, analytics, or "phone home" calls — zero. Not even anonymous install counts.
- The Oracle system prompt used during bootstrap must be **stored in the repo** (readable by the user) and loaded from the local clone at runtime. It must not be fetched from a remote server (which could be swapped out).
- The permission allowlist (what scripts the Oracle can call) must be **hardcoded in the bootstrap script itself**, not fetched from any external source.
- There must be no "developer key" or "master credential" embedded anywhere in the codebase that would grant privileged access to a running instance.
- These constraints must be auditable: a user reading the bootstrap code should be able to verify all of the above without trusting the developer's word.

#### 5. Oracle Scope Containment
- The allowlist of callable scripts/operations is defined as a **hardcoded constant** in the bootstrap runner — not derived from Oracle output.
- Oracle outputs structured JSON decisions (e.g. `{"action": "write_config", "payload": {...}}`). The bootstrap runner maps those to actual function calls. Oracle never directly calls shell.
- Network access during bootstrap is limited to: Groq API (LLM calls) and a curated list of HuggingFace model repos. No other outbound connections.
- File system writes are scoped to the GAIA install directory only.

#### 6. GAIA's Principles During Bootstrap
- The bootstrap Oracle system prompt must include GAIA's full constitutional constraints — the same principles she operates under at runtime.
- There must be no "install mode" that loosens or suspends those principles. GAIA should not be jailbreakable at birth by the installer.
- The system prompt must be loaded from the local repo (not fetched remotely) so users can inspect it.
- The Oracle should refuse and halt the bootstrap if asked to do something outside its scope, just as GAIA would refuse at runtime.

#### 7. Post-Bootstrap Cleanup
- After install completes, bootstrap should explicitly prompt the user on whether to retain the Groq key for Oracle fallback or discard it.
- If discarded: zero-fill the in-memory key, confirm no disk traces.
- If retained: explain exactly where it's stored, how it's protected, and how to remove it later.
- Bootstrap should leave no residual elevated privileges, temporary credentials, or open network listeners.

---

### Design Principle: Symmetric Trust

The security model must be symmetric: **the developer gets no more trust than any other party once GAIA is installed.** This should be the default for any distributed software — it's only notable because the industry has normalized the opposite. The bootstrap process should be something the developer could hand to a security researcher and say "find my backdoor" — and there should be nothing to find.

---

## Relationship to Existing Systems

- **Oracle pipeline**: Bootstrap Oracle reuses the same pipeline structure; Groq replaces gaia-prime as the LLM backend during install only.
- **gaia_constants.json**: The config file that bootstrap writes is the same one gaia-core reads at runtime — no new config format needed.
- **quantize_awq.py / GGUF scripts**: Already exist or are in progress; bootstrap calls them rather than duplicating logic.
- **gaia-prime warm pool**: After bootstrap, gaia-prime is started from the warm pool as normal. Bootstrap does not change how gaia-prime runs, only which model it points to.

---

## Why This Is Compelling

- Lowers the barrier to GAIA distribution from "technical GPU user who can tune configs" to "anyone who can run a shell script and get a free API key"
- GAIA effectively installs herself — consistent with the self-improving, autonomous system philosophy
- Groq Free tier means zero cost to the end user for the install step
- The permission-bounded Oracle scope is a clean security model: Oracle can configure GAIA but cannot escape its sandbox

---

*Not scheduled for implementation. Record only.*
