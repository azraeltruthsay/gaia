🧭 CoPilot Brief – GAIA Model Environment Integration Plan

Context:
You are inside GAIA_Project/ on Arch Linux.
This workspace contains two sibling folders:

GAIA_Project/
│
├── gaia-assistant/     # The GitHub repo / main codebase
└── gaia-models/        # Large local model files (not versioned)


Goal:
Unify GAIA’s model setup workflow and centralize logs.
Implement the four scripts and one configuration update below so that model verification, linking, and cleanup work both locally and in Docker.

🧩 Target Structure
gaia-assistant/
├── app/
│   ├── config.py
│   └── setup_models.py        ← verify + symlink models
│
├── runtime/
│   ├── logs/                  ← all logs here (log files gitignored)
│   ├── tmp/
│   └── state/
│
├── scripts/
│   ├── build_model_env.sh     ← verify & link models
│   ├── clean_model_env.sh     ← remove broken links
│   ├── model_env_refresh.sh   ← clean + rebuild combo
│   └── (future) rotate_logs.sh
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yml


External real models live in:

../gaia-models/
    Nemotron-8B.gguf
    Qwen2.5-3B.gguf
    Hermes-2B.gguf

⚙️ Implementation Tasks for CoPilot

Config Update

In app/config.py, set

MODEL_DIR = str(Path(__file__).resolve().parents[2] / "gaia-models")
LOG_DIR   = str(Path(__file__).resolve().parents[1] / "runtime" / "logs")


Add PRIME_MODEL_PATH, LITE_MODEL_PATH, and CODEMIND_MODEL_PATH values referencing the files inside MODEL_DIR.

Modify Scripts

Update all existing setup/cleanup scripts to:

Read/write logs in runtime/logs.

Create symlinks in gaia-assistant/models/ pointing to ../gaia-models/*.

Never delete or modify actual model files.

Ensure scripts auto-create missing runtime/* subdirectories.

Docker Adjustments

Mount both paths:

volumes:
  - ./gaia-assistant:/app
  - ./gaia-models:/models


Inside the container, /app/models/*.gguf should symlink to /models/*.gguf.

Verification

Running ./scripts/model_env_refresh.sh from gaia-assistant/ should:

Clean broken links.

Re-link valid model files.

Output results to runtime/logs/model_setup.log.