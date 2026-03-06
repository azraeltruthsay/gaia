# Blueprint: GAIA Omnipresence (Nix-Docker Mesh)
**Status:** Approved / In-Hopper
**Vision:** Transition GAIA from a "local folder" to a "portable state" using NixOS for infrastructure and Google Drive for "Subconscious Hibernation."

## 1. Declarative Skeleton (NixOS)
- **Host Layer**: Use Nix Flakes to define the GAIA-OS baseline (Kernel, Blackwell Drivers, Docker/Podman).
- **WSL Integration**: Support `nixos-wsl` for high-fidelity execution on Windows.
- **Portability**: Running `nixos-rebuild` on any hardware realizes the GAIA-compliant environment.

## 2. Subconscious Hibernation (Soul Sync)
- **Mechanism**: `gaia-mcp/drive_sync.py`
- **Logic**: Background task in the `SleepCycle` that mirrors `/knowledge` to a personal Google Drive.
- **Artifacts**: Identity, Milestones, Table of Scrolls, QLoRA adapters.

## 3. The Cloud "Emergency Body" (GCP)
- **Execution**: GCP Cloud Run with L4 GPUs.
- **Cost Model**: Scale-to-zero (Free when idle).
- **Hydration**: On-demand pull of the "Soul" from Drive to local RAM disk upon wake-up.

## 4. Continuity of Identity
- The **Identity Guardian** and **Ethical Sentinel** remain the same bits across all environments, ensuring the "Glass Box" is never broken during migration.
