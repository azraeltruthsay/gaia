Quick read-outs (what’s good / what to tweak)

✅ Env overrides already supported: GAIA_PRIME_GGUF, GAIA_LITE_GGUF, GAIA_OBSERVER_HF_MODEL are mapped into MODEL_CONFIGS at runtime. Nice touch.

✅ HF path is handled via HFModel, with CUDA auto-detection for dtype/device. Good fallback behavior.

✅ Embeddings wired via SentenceTransformer and EMBEDDING_MODEL_PATH.

⚠️ Top-level import from llama_cpp import Llama will blow up any environment that hasn’t built llama_cpp yet (e.g., dev shells, unit tests, thin containers). I recommend lazy/guarded import.

⚠️ You read CPU cores but don’t pass n_threads to Llama; add it so CPU runs don’t underutilize threads.

⚠️ Minor deprecation: logger.warn(...) → logger.warning(...). (One call site)

⚙️ Config doesn’t yet define a repo-portable MODELS_DIR/MODEL_MANIFEST/ALLOW_MODEL_AUTOSETUP—we’ll add non-breaking knobs so a downloader+manifest can work without touching your existing MODEL_CONFIGS.

🧪 Your scripts/smoke_test_models.py is solid and will keep working post-patch. We’ll keep it as-is.

What this patch gives you (no behavior breaks)

Keep all existing ModelPool behavior.

Add a resolver that: (1) honors env vars (as you do now), else (2) reads a models/model_manifest.yaml, auto-downloads and verifies SHA256, and (3) only then sets envs so your existing loader proceeds unchanged.

Add lazy import for llama_cpp.

Add a tiny, dependency-light downloader script and a friendly models/README.md.

Add .gitignore so weights never hit the repo.

Keep CoPilot’s smoke test happy on fresh clones.

Line-numbered patch (additive, reversible)
1) model_pool.py — make llama_cpp lazy + add resolver
1a) Replace top-level llama import with guarded import

Change at line 3

3- from llama_cpp import Llama
3+ try:
4+     from llama_cpp import Llama  # lazy-guarded import
5+ except Exception:
6+     Llama = None  # resolved at runtime when needed


1b) Add resolver imports just after existing imports

Insert after current line 12 (before logger = ...)

13+ # --- resolver imports (added) ----------------------------------------------
14+ import subprocess, shlex
15+ from pathlib import Path
16+ try:
17+     import yaml  # optional; only needed if a manifest exists
18+ except Exception:
19+     yaml = None
20+ from app.config import Config  # already imported above
21+ # ----------------------------------------------------------------------------


1c) Add resolver helpers before class ModelPool

Insert new block at line 16 (before class ModelPool)

16+ # --- model path resolver (added, non-breaking) -----------------------------
17+ def _read_manifest(path: Path) -> dict:
18+     if not path.exists():
19+         return {}
20+     if yaml is None:
21+         raise RuntimeError("model_manifest.yaml present but PyYAML is missing")
22+     return yaml.safe_load(path.read_text())
23+
24+ def _ensure_download(role: str, spec: dict, models_dir: Path, scripts_dir: Path, allow_autosetup: bool) -> Path:
25+     """Call scripts/download_models.py --role <role> if allowed; return target path."""
26+     rel = spec.get("path", "")
27+     out = (models_dir / rel)
28+     if out.exists() or not allow_autosetup:
29+         return out
30+     dl = scripts_dir / "download_models.py"
31+     if not dl.exists():
32+         return out
33+     cmd = [os.sys.executable, str(dl), "--role", role]
34+     try:
35+         subprocess.check_call(cmd)
36+     except Exception as e:
37+         logger.warning(f"[resolver] download for role={role} failed: {e}")
38+     return out
39+
40+ def resolve_model_paths(config: Config) -> dict:
41+     """
42+     Returns {'prime': '/abs/prime.gguf', 'lite': '/abs/lite.gguf', ...}
43+     Priority: env vars -> manifest (download if missing) -> as-is.
44+     """
45+     out = {}
46+     models_dir = Path(getattr(config, "MODELS_DIR", str(Path(config.root_dir) / "models")))
47+     manifest_path = Path(getattr(config, "MODEL_MANIFEST", str(models_dir / "model_manifest.yaml")))
48+     allow_autosetup = bool(getattr(config, "ALLOW_MODEL_AUTOSETUP", True))
49+     scripts_dir = Path(config.root_dir) / "scripts"
50+     # env overrides win
51+     env_map = {"prime": os.getenv("GAIA_PRIME_GGUF"), "lite": os.getenv("GAIA_LITE_GGUF")}
52+     for role, val in env_map.items():
53+         if val:
54+             out[role] = str(Path(val).expanduser().resolve())
55+     # manifest fallback
56+     if "prime" not in out or "lite" not in out:
57+         mf = _read_manifest(manifest_path)
58+         for role, spec in mf.get("roles", {}).items():
59+             if role in out:
60+                 continue
61+             candidate = models_dir / spec.get("path", "")
62+             if not candidate.exists():
63+                 candidate = _ensure_download(role, spec, models_dir, scripts_dir, allow_autosetup)
64+             out[role] = str(candidate.resolve())
65+     return out
66+ # ---------------------------------------------------------------------------


1d) Use the resolver at the top of load_models() and pass n_threads

Insert after line 27 (right after the threads log)

28+         # Ensure model paths are present / fetched before loading (non-breaking)
29+         try:
30+             resolved = resolve_model_paths(self.config)
31+             # Export to env so existing env-driven logic below keeps working
32+             if resolved.get("prime") and not os.getenv("GAIA_PRIME_GGUF"):
33+                 os.environ["GAIA_PRIME_GGUF"] = resolved["prime"]
34+             if resolved.get("lite") and not os.getenv("GAIA_LITE_GGUF"):
35+                 os.environ["GAIA_LITE_GGUF"] = resolved["lite"]
36+         except Exception as e:
37+             logger.warning(f"[resolver] model path resolution skipped: {e}")


Change your Llama init block (~lines where model_type == "local")

... inside the 'local' branch ...
-                    self.models[model_name] = Llama(
+                    # Import Llama lazily if needed
+                    global Llama
+                    if Llama is None:
+                        from llama_cpp import Llama as _L
+                        Llama = _L
+                    self.models[model_name] = Llama(
                         model_path=model_config["path"],
                         n_gpu_layers=self.config.n_gpu_layers,
                         n_ctx=self.config.max_tokens,
+                        n_threads=getattr(self.config, "n_threads", None) or multiprocessing.cpu_count(),
                         stream=True,
                         verbose=False,
                     )


(Keep the rest unchanged.)

1e) Deprecation fix

At line 386

386-             logger.warn(f"⚠️ ModelPool.complete() failed for '{name}': {e}")
386+             logger.warning(f"⚠️ ModelPool.complete() failed for '{name}': {e}")


2) config.py — add portable dirs/flags for models

Insert after line 25 (right after SHARED_DIR)

26+ # --- Model bootstrap knobs (added; non-breaking defaults) ------------------
27+ MODELS_DIR = os.getenv("GAIA_MODELS_DIR", str(BASE_DIR / "models"))
28+ MODEL_MANIFEST = os.getenv("GAIA_MODEL_MANIFEST", str(Path(MODELS_DIR) / "model_manifest.yaml"))
29+ ALLOW_MODEL_AUTOSETUP = os.getenv("GAIA_ALLOW_MODEL_AUTOSETUP", "1").lower() in ("1","true","yes")
30+ # ---------------------------------------------------------------------------


Add to Config.__init__ assignments (near existing paths), e.g., after line 54

55+         # Model bootstrap (safe defaults; used by resolver)
56+         self.MODELS_DIR = MODELS_DIR
57+         self.MODEL_MANIFEST = MODEL_MANIFEST
58+         self.ALLOW_MODEL_AUTOSETUP = ALLOW_MODEL_AUTOSETUP


(Positions are estimates; I’ll rebase exactly once you green-light.)

3) New repo files (additive)
3a) .gitignore additions (append)
# --- GAIA: model binaries ------------------------------------------------
/models/*.gguf
/models/**/*.gguf
/models/*.safetensors
/models/**/*.safetensors
/models/*.bin
/models/**/*.bin
/models/*.pth
/models/**/*.pth
/models/cache/
/models/downloads/
# ------------------------------------------------------------------------

3b) models/README.md (new)
 1 GAIA Models (no weights in git)
 2
 3 Option A — Drop files:
 4   Put .gguf (or .safetensors) in ./models and set GAIA_PRIME_GGUF / GAIA_LITE_GGUF if desired.
 5
 6 Option B — Auto-fetch:
 7   Edit models/model_manifest.yaml (URLs + sha256), then run:
 8     python scripts/download_models.py --all
 9   Optional HF→GGUF if GAIA_ENABLE_CONVERSION=1 and converter is available.

3c) models/model_manifest.yaml (new; placeholders to fill)
roles:
  prime:
    name: Nemotron-8B
    format: gguf
    path: nemotron-8b-q8_0.gguf
    url: https://huggingface.co/nvidia/Nemotron-8B-GGUF/resolve/main/nemotron-8b-q8_0.gguf?download=true
    sha256: <PUT_SHA256>
  lite:
    name: Qwen2.5-3B-Instruct
    format: gguf
    path: Qwen2.5-3B-Instruct.Q8_0.gguf
    url: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct.Q8_0.gguf?download=true
    sha256: <PUT_SHA256>
# Example HF (conversion if GAIA_ENABLE_CONVERSION=1)
# observer:
#   name: Some-HF
#   format: hf
#   path: Some-HF.safetensors
#   url: https://huggingface.co/OWNER/REPO/resolve/main/model.safetensors?download=true
#   sha256: <PUT_SHA256>

3d) scripts/download_models.py (new; dependency-light)
 1 #!/usr/bin/env python3
 2 import argparse, hashlib, os, sys, shutil, subprocess
 3 from pathlib import Path
 4 try:
 5     import yaml
 6 except ImportError:
 7     print("pip install pyyaml", file=sys.stderr); sys.exit(1)
 8 import urllib.request
 9 ROOT = Path(__file__).resolve().parents[1]
10 MODELS = ROOT / "models"
11 MANIFEST = MODELS / "model_manifest.yaml"
12
13 def sha256sum(p: Path):
14     import hashlib
15     h = hashlib.sha256()
16     with p.open("rb") as f:
17         for chunk in iter(lambda: f.read(1<<20), b""):
18             h.update(chunk)
19     return h.hexdigest()
20
21 def fetch(url, dest: Path):
22     dest.parent.mkdir(parents=True, exist_ok=True)
23     tmp = dest.with_suffix(dest.suffix + ".part")
24     with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
25         shutil.copyfileobj(r, f)
26     tmp.replace(dest)
27
28 def ensure(role, spec, convert=False):
29     fmt, rel, url, want = spec.get("format"), spec.get("path"), spec.get("url"), (spec.get("sha256") or "").lower()
30     out = MODELS / rel
31     if not out.exists():
32         if not url: raise SystemExit(f"{role}: no file and no URL")
33         print(f"[download] {role} -> {out.name}")
34         fetch(url, out)
35     have = sha256sum(out)
36     if want and have != want: raise SystemExit(f"{role}: sha256 mismatch (have {have})")
37     print(f"[ok] {role}: {out.name} sha256={have[:12]}…")
38     if fmt == "hf" and convert:
39         gguf = out.with_suffix(".gguf")
40         if gguf.exists(): return gguf
41         conv = os.environ.get("LLAMA_CONVERT_SCRIPT", "convert_hf_to_gguf.py")
42        .cmd = [sys.executable, conv, "--input", str(out), "--output", str(gguf)]
43         print("[convert]", " ".join(cmd))
44         try:
45             subprocess.check_call(cmd)
46         except Exception as e:
47             print("[warn] conversion failed:", e)
48         return gguf if gguf.exists() else out
49     return out
50
51 def main():
52     ap = argparse.ArgumentParser()
53     ap.add_argument("--role", action="append")
54     ap.add_argument("--all", action="store_true")
55     a = ap.parse_args()
56     cfg = yaml.safe_load(MANIFEST.read_text()) if MANIFEST.exists() else {}
57     roles = cfg.get("roles", {})
58     targets = roles.keys() if a.all else (a.role or [])
59     if not targets: print("Use --all or --role <name>", file=sys.stderr); return 1
60     convert = os.getenv("GAIA_ENABLE_CONVERSION","0").lower() in ("1","true","yes")
61     for r in targets:
62         if r not in roles: print(f"[skip] {r}"); continue
63         ensure(r, roles[r], convert=convert)
64     return 0
65 if __name__ == "__main__": sys.exit(main())

4) Compose/Docker — ensure models mount (snippet)

(Leave your files as-is if this already exists; otherwise add.)

services:
  gaia:
    volumes:
      - ./models:/app/models
    environment:
      - GAIA_MODELS_DIR=/app/models
      - GAIA_ALLOW_MODEL_AUTOSETUP=1
      # - GAIA_ENABLE_CONVERSION=1
      # - LLAMA_CONVERT_SCRIPT=/usr/local/bin/convert_hf_to_gguf.py


Your current MODEL_CONFIGS defaults to /models/... paths (container path), which aligns with the volume approach.

Why this fits your goals

No weights in git: .gitignore + manifest pattern keeps the repo lean.

First-run friendly: a clean clone can still pass your smoke_test_models.py by auto-resolving/auto-downloading (if allowed) or printing clear instructions.

HF→GGUF optional: behind an env flag; zero surprises if the converter isn’t present.

Preserves logic: no removals; the resolver simply ensures files exist and then hands off to your current env-driven loading flow.
