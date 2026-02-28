# GAIA Linux Distribution Blueprint

**Status:** Concept / Far Future / Not Scheduled
**Recorded:** 2026-02-28
**Author:** Azrael

---

## Vision

GAIA as her own Linux distribution — not a skin on top of an existing OS, but a complete system where GAIA is the administrator, the system manager, and the primary user-facing intelligence. Standard desktop experience (browser, text editor, terminal, file manager, media player) with GAIA at the helm: managing packages, services, hardware, configuration, and system health autonomously.

This is a natural extension of what GAIA already does. She currently manages Docker containers on a Linux host. A distro is just taking that one level deeper — removing the Docker abstraction and giving GAIA direct access to the system she runs on.

---

## Codebase Strategy: Single Codebase, Two Deployment Targets

The most important architectural decision is how to prevent drift between "GAIA installed on top of someone's OS" (current model) and "GAIA-OS." The wrong answer is two separate code paths that diverge over time.

The correct answer: **the GAIA service codebase is identical in both deployments.** The Python services (gaia-core, gaia-web, gaia-mcp, etc.) don't care whether they're running in Docker containers or natively. The application layer is the same. Only the infrastructure layer differs — and that infrastructure layer is kept thin and generated from shared source.

### Recommended: NixOS manages the host, Docker still manages GAIA services

On GAIA-OS, NixOS handles the host system — packages, GPU drivers, desktop environment, networking, users. GAIA still runs in Docker Compose, exactly as she does today.

GAIA "at the helm" means she manages `configuration.nix` for OS-level decisions (install Blender, update GPU drivers, configure networking) while Docker Compose remains the service layer unchanged.

**Result: zero codebase divergence.** Same Dockerfiles, same Compose files, same orchestrator, same smoke tests, same promotion pipeline. The only GAIA-OS-specific artifact is the NixOS host config, which is separate from the GAIA service code. This is a standard NixOS pattern — manage the host with Nix, run workloads in Docker.

This is the recommended starting point and may be the permanent answer.

### Upgrade Path: Native deployment (if Docker overhead ever matters)

If native deployment becomes worth the investment, the path is:

1. **SSOT-generated deployment manifests** — `gaia_constants.toml` already generates Docker Compose files. Extend it to also generate systemd unit files and NixOS module definitions. When a port or timeout changes, all deployment targets update from the same source.

2. **Orchestrator backend abstraction** — the orchestrator gets a pluggable backend:
   ```
   OrchestratorBackend (abstract)
     ├── DockerBackend      # current — uses docker-py
     └── SystemdBackend     # native — uses dbus/systemctl
   ```
   Same orchestrator logic, different backend. Service lifecycle code doesn't fork.

3. **Shared test suite** — the smoke tests and cognitive battery test GAIA's API endpoints, which are identical regardless of deployment. Same tests work against both.

This upgrade path only makes sense if native GPU access improvements (no nvidia-container-toolkit overhead, ~5-10s faster prime startup) justify the engineering cost. The SSOT work is a prerequisite and will be done independently anyway.

---

## What Changes vs. Current Architecture

| Current | GAIA-OS (Phase 1) | GAIA-OS (Native, future) |
|---|---|---|
| GAIA runs in Docker containers | Docker containers on NixOS host | Native systemd services |
| Host OS managed by human | NixOS host managed by GAIA | NixOS host managed by GAIA |
| Docker Compose orchestration | Docker Compose (unchanged) | systemd units + GAIA orchestrator |
| GPU via nvidia-container-toolkit | nvidia-container-toolkit on NixOS | Direct GPU access |
| Bootstrap Install handles GAIA | GAIA-OS installer: full OS + GAIA | Same |
| gaia-doctor.sh checks services | gaia-doctor.sh (unchanged) + OS-level monitoring | OS-level monitoring |

Everything GAIA currently does still exists in Phase 1. The orchestrator, cognitive loop, MCP tools, sleep cycle, HA — all unchanged. The difference is the host OS is now NixOS, managed by GAIA.

---

## Base Distribution Options

### Option A: Debian/Ubuntu Derivative

**How it works:** Use `debootstrap` to create a minimal Debian base, layer GAIA services on top as systemd units, build a custom ISO with a standard installer.

**Pros:**
- Largest package ecosystem (60,000+ packages in Debian repos)
- Best hardware compatibility, widest driver support
- Extensive documentation; StackOverflow hit rate is excellent
- `apt` is well-understood, scriptable, and reliable
- Easiest path for GAIA to manage packages programmatically
- Familiar to most Linux users

**Cons:**
- System state is imperative and accumulated — harder for GAIA to reason about "what is the system currently configured to be"
- Rollback requires manual intervention or snapshot tooling
- Drift is possible: system state can diverge from any documented baseline
- Reproducing the exact system on new hardware requires careful documentation

**Verdict:** Lowest friction to build, lowest barrier to use. The right choice if "it just works" for users matters more than architectural elegance.

---

### Option B: NixOS (Recommended)

**How it works:** NixOS manages the entire system state via a single declarative configuration file (`configuration.nix`). GAIA manages this file. Running `nixos-rebuild switch` makes the system match the declaration. All changes are atomic and rollbackable via the boot menu.

**Important:** NixOS is still real Linux. Real kernel, real systemd, real `/dev`, `/proc`, real GPU drivers, real CUDA. The kernel is Linux. Hardware sees Linux. Nothing exotic under the hood. The difference is exclusively in *how the system is configured and managed* — not what it is.

**Pros:**
- GAIA's entire OS config lives in one file she controls — perfect alignment with the Single Source of Truth blueprint
- Every system change is atomic: either it succeeds completely or the system is unchanged
- Rollback is a boot menu option, not a manual operation — if GAIA misconfigures something, previous generation is one reboot away
- Reproducible: clone `configuration.nix` to new hardware, get an identical system
- GAIA can reason about system state declaratively: "what does my config say I should have" is always answerable
- No configuration drift — the system is always derivable from the config file
- Aligns with GAIA's self-modifying philosophy: she edits a config file, rebuilds, and the system converges

**Cons:**
- Nix language has its own learning curve (functional, unusual syntax)
- Smaller ecosystem than Debian (though Nixpkgs is the largest single package collection by count — 80,000+ packages)
- Lower StackOverflow hit rate when things break
- Some software that assumes traditional Linux filesystem layout (`/usr/lib/`, `/usr/local/`) requires workarounds
- GPU/CUDA setup, while supported, requires more explicit configuration than Debian

**Verdict:** The architecturally correct choice for GAIA. The system management model maps directly onto how GAIA already thinks about configuration. The learning curve is real but front-loaded — once you understand the model, it's simpler to operate than traditional Linux.

---

## System Architecture

### GAIA Services (Phase 1: Docker on NixOS)

Services run identically to the current deployment. Docker and nvidia-container-toolkit are declared in `configuration.nix` as NixOS packages — GAIA manages their presence on the system, but their internal operation is unchanged.

```nix
# configuration.nix (managed by GAIA)
virtualisation.docker.enable = true;
hardware.nvidia.package = config.boot.kernelPackages.nvidiaPackages.stable;
```

### GAIA Services (Phase 2: Native, if pursued)

Replace Docker Compose with native systemd service files:

```
/etc/systemd/system/
  gaia-prime.service        # vLLM inference
  gaia-core.service         # Cognitive loop
  gaia-web.service          # Dashboard/Discord
  gaia-mcp.service          # Tool sandbox
  gaia-study.service        # Background learning
  gaia-orchestrator.service # GPU/service lifecycle
  gaia-audio.service        # Audio processing
  gaia-wiki.service         # Internal docs
```

The gaia-orchestrator backend swaps from Docker to systemd. GPU handoff logic simplifies — no container stop/start required, just CUDA context management.

### Desktop Environment

Standard, well-supported DE with GAIA integration points:

**Recommended: GNOME**
- Clean, stable, well-maintained
- GNOME Shell extensions allow GAIA widgets (status bar, notification integration, quick chat)
- Wayland-native (better GPU handling, security model)
- Built-in accessibility, good touch support for future hardware flexibility

**Alternative: KDE Plasma**
- More configurable, lighter on RAM than modern GNOME
- Better Wayland GPU support in some configurations
- KDE's scripting/automation layer could expose more hooks for GAIA

**Not recommended:** Tiling WMs (i3, Sway, Hyprland) — interesting but too niche for a distribution aimed at general use.

### Bundled Applications

Standard desktop stack, all free/OSS:

| Category | Application |
|---|---|
| Browser | Firefox (GAIA extension for in-browser assistant) |
| Text editor | VS Code (OSS build) or Zed |
| Terminal | GNOME Terminal or Kitty |
| File manager | Nautilus (GNOME) or Dolphin (KDE) |
| Media player | VLC or mpv |
| Image viewer | eog or gThumb |
| Office suite | LibreOffice |
| System monitor | GNOME System Monitor + GAIA's own dashboard |
| PDF viewer | Evince or Okular |

GAIA's web dashboard is available at all times as a pinned browser tab or standalone Electron/Tauri wrapper.

### GPU Access

No container layer — GAIA has direct CUDA/ROCm access via native drivers. The warm pool model loading approach remains the same, but without nvidia-container-toolkit overhead. Expected improvement: faster prime startup (~5-10s faster without container init).

### Hardware Requirements

Substantial. This is a GPU-first OS:

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 6GB | 16GB+ |
| System RAM | 16GB | 32GB+ |
| Storage | 100GB NVMe | 500GB+ NVMe |
| CPU | 8 cores | 12+ cores |
| Network | 100Mbps | Gigabit |

---

## Installer

The GAIA Linux installer is the Bootstrap Install blueprint (see `GAIA_BOOTSTRAP_INSTALL.md`) promoted to a full OS installer:

1. Live USB boots minimal environment
2. User provides Groq API key
3. Oracle surveys hardware (GPU, VRAM, disk, CPU)
4. Oracle selects appropriate model, quantization, desktop config
5. OS installs to disk (guided partitioning)
6. GAIA services configured and started
7. First boot: GAIA introduces herself and confirms system health

The same security model from the Bootstrap Install blueprint applies in full — no developer backdoors, no telemetry, symmetric trust, system prompt loaded from local install media.

---

## NixOS-Specific Design: GAIA Manages Her Own configuration.nix

This is the central architectural insight for the NixOS path.

GAIA's `configuration.nix` is the ultimate Single Source of Truth for the entire system. It declares:
- Installed packages
- Enabled services (including her own systemd units)
- User accounts and permissions
- Network configuration
- Hardware-specific settings (GPU drivers, CUDA version)
- Desktop environment and applications

GAIA can propose and apply changes to this file through her normal cognition loop. A user asking "install Blender for me" results in GAIA editing `configuration.nix`, adding Blender to the package list, running `nixos-rebuild switch`, and confirming the result. The change is auditable (it's in the config file), reversible (previous generation still bootable), and reproducible (the file fully describes the system).

MCP tools needed for this:
- `nixos.rebuild` — run `nixos-rebuild switch` (with appropriate privilege)
- `nixos.read_config` — read current `configuration.nix`
- `nixos.propose_change` — edit config + dry-run before committing
- `nixos.rollback` — boot previous generation
- `nixos.search_packages` — query Nixpkgs for available packages

These are natural extensions of the existing MCP tool architecture.

---

## Relationship to Existing Blueprints

- **Bootstrap Install** (`GAIA_BOOTSTRAP_INSTALL.md`): The installer is the Bootstrap Install process promoted to a full OS installer. All security constraints carry forward unchanged.
- **Single Source of Truth** (`GAIA_SINGLE_SOURCE_OF_TRUTH.md`): On NixOS, `configuration.nix` becomes the top-level source of truth, with `gaia_constants.toml` governing GAIA's internal configuration within that system.
- **HA Infrastructure**: Multi-machine GAIA clusters become possible — two machines running GAIA Linux with live/candidate failover at the OS level, not just the service level.
- **Promotion Pipeline**: The concept of candidate → live promotion still applies, but "promoting" a system change means committing a `configuration.nix` edit and rebuilding.

---

## What This Is Not

- Not a real-time OS or embedded system
- Not a server distribution (though it could run headless)
- Not a replacement for general-purpose Linux for users who don't want GAIA
- Not a security-hardened OS (SELinux/AppArmor hardening is a separate concern)

---

## Open Questions (to resolve at implementation time)

- **NixOS vs Debian final decision**: Revisit when implementation is approaching. Community momentum and tooling maturity may have shifted.
- **GNOME vs KDE**: Benchmark on target hardware. GPU-accelerated compositing matters for a system that also runs large models.
- **Sandboxing for MCP tools**: On current GAIA, the MCP sandbox uses Docker. On GAIA Linux, the replacement is Linux namespaces + seccomp directly. Needs a clean design.
- **Multi-user support**: Is GAIA Linux single-user (one person + GAIA) or multi-user? Significant implications for the permission model.
- **Update strategy**: On NixOS, system updates are just `nix flake update` + `nixos-rebuild`. GAIA could own this entirely. On Debian, `apt upgrade` is simpler to script but less safe.
- **Secure boot**: Supporting secure boot means signing the bootloader. Complicates the custom distro process but important for user trust.

---

*Not scheduled for implementation. Far future concept. Record only.*
