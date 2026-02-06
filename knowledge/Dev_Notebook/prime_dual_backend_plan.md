---

# 📑 GAIA Proposal: Unified Prime Model (CPU/GPU Variants)

---

## 1. Problem Framing

GAIA currently defines **Prime** as a `.gguf` model in `gaia_constants.json`.

* On the **laptop**, Prime should remain a small CPU-friendly `.gguf` model.
* On the **RTX rig**, Prime should run as a Hugging Face Nemotron model with GPU acceleration.
* Regardless of hardware, GAIA’s council logic and persona system should always see **Prime as Prime**.

---

## 2. Proposal

* Add **cpu\_prime** and **gpu\_prime** entries in `MODEL_CONFIGS`.
* Keep a `prime` alias pointing to whichever is active.
* Switch at runtime via **`GAIA_BACKEND` env var** or constants default.
* Add loader logic to handle `local` (llama.cpp), `hf` (Hugging Face), and `api` models.
* Use separate Docker Compose files to mount the correct model directories.

---

## 3. File Changes

### 🔹 gaia\_constants.json

```jsonc
"MODEL_CONFIGS": {
  "cpu_prime": {
    "type": "local",
    "path": "/models/gaia-prime.gguf",
    "enabled": true
  },
  "gpu_prime": {
    "type": "hf",
    "model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "path": "/models/nemotron-8b",
    "enabled": true
  },
  "prime": {
    "alias": "cpu_prime",   // default to CPU for portability
    "enabled": true
  },
  "lite": {
    "type": "local",
    "path": "/models/gaia-lite.gguf",
    "enabled": true
  },
  "oracle": {
    "type": "api",
    "provider": "openai",
    "model": "gpt-4o",
    "enabled": true
  }
}
```

---

### 🔹 config.py (patch)

```diff
@@ class Config:
-        self.model_path = self.constants.get("model_paths", {}).get("Prime", None)
-        self.llm_backend = os.getenv("GAIA_BACKEND", self.constants.get("llm_backend", None))
+        # Determine which backend GAIA Prime should use
+        backend_choice = os.getenv("GAIA_BACKEND", self.constants.get("llm_backend", "prime"))
+        alias_cfg = self.MODEL_CONFIGS.get(backend_choice, {})
+        if "alias" in alias_cfg:
+            backend_choice = alias_cfg["alias"]
+
+        self.llm_backend = backend_choice
+        self.model_path = self.MODEL_CONFIGS.get(backend_choice, {}).get("path", None)
```

---

## 4. Loader Snippet

This unified loader respects the `type` field in `MODEL_CONFIGS`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llama_cpp import Llama
import torch

def load_model(cfg: dict):
    mtype = cfg.get("type")
    if mtype == "local":
        # llama.cpp backend
        return Llama(
            model_path=cfg["path"],
            n_threads=cfg.get("n_threads", 4),
            n_gpu_layers=cfg.get("n_gpu_layers", 0),
        )
    elif mtype == "hf":
        # Hugging Face backend (with 4-bit quantization by default)
        tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            quantization_config=bnb_config,
            device_map="auto"
        )
        return (tokenizer, model)
    elif mtype == "api":
        # Proxy to API-based provider (e.g. OpenAI)
        return {"provider": cfg["provider"], "model": cfg["model"]}
    else:
        raise ValueError(f"Unknown model type: {mtype}")
```

---

## 5. Docker Notes

* **docker-compose.yml (CPU laptop)**
  Mount `.gguf` models only:

  ```yaml
  volumes:
    - ../gaia-models:/models
  ```

* **docker-compose.gpu.yml (RTX rig)**
  Mount Nemotron for GPU Prime:

  ```yaml
  volumes:
    - ../gaia-models/nemotron-8b:/models/nemotron-8b
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [gpu]
  ```

---

## 6. Run Instructions

* **Laptop (CPU Prime)**

  ```bash
  docker compose up
  ```

* **RTX rig (GPU Prime)**

  ```bash
  GAIA_BACKEND=gpu_prime docker compose -f docker-compose.gpu.yml up
  ```

* **Explicit CPU override anywhere**

  ```bash
  GAIA_BACKEND=cpu_prime docker compose up
  ```

---

## 7. Future-Proofing

* Add more Prime variants (`quant_prime`, `cloud_prime`, etc.) without breaking identity.
* Observer/council always sees Prime as Prime.
* This pattern keeps GAIA portable and flexible across machines.

---

✨ With this setup, GAIA:

* **Always identifies as Prime**, preserving consistency.
* **Chooses CPU/GPU Prime at runtime**, no code duplication.
* **Handles multiple backends** (llama.cpp, Hugging Face, API) gracefully.

---

