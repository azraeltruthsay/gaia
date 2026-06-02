#!/usr/bin/env bash
# Resilient launcher for train_core_multimodal.py.
#
# Works around GAIA_Project-22a: the RTX 5080 drives both the desktop
# compositor and the CUDA training workload, so the NVRM Robust-Channel
# watchdog (7s) occasionally resets the GPU channel mid-run and the trainer
# dies with "the launch timed out and was terminated" (bitsandbytes ops.cu).
#
# The trainer checkpoints every --save-steps and auto-resumes from the latest
# checkpoint in its adapter dir on restart. This wrapper relaunches it after a
# crash until the run completes (exit 0) or MAX_RETRIES is hit, so a long run
# survives any number of transient watchdog hits.
#
# Usage:
#   MAX_RETRIES=12 TRAIN_LOG=/shared/training_runs/v15_launch.log \
#     scripts/train_core_multimodal_resilient.sh \
#       --base-model /models/google/gemma-4-E4B --curriculum-name core_v2x_spiral ...
#
# Forwards all args verbatim to the trainer.
set -u

SCRIPT=/gaia/GAIA_Project/scripts/train_core_multimodal.py
MAX_RETRIES="${MAX_RETRIES:-12}"
LOG="${TRAIN_LOG:-/shared/training_runs/resilient_launch.log}"
RETRY_SLEEP="${RETRY_SLEEP:-20}"

i=0
while : ; do
    i=$((i + 1))
    echo "=== [resilient] attempt ${i}/${MAX_RETRIES} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python3 "$SCRIPT" "$@" >>"$LOG" 2>&1
    rc=$?
    echo "=== [resilient] attempt ${i} exited rc=${rc} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    if [ "$rc" -eq 0 ]; then
        echo "=== [resilient] COMPLETE after ${i} attempt(s) ===" | tee -a "$LOG"
        break
    fi
    if [ "$i" -ge "$MAX_RETRIES" ]; then
        echo "=== [resilient] GAVE UP after ${i} attempts (rc=${rc}) ===" | tee -a "$LOG"
        exit 1
    fi
    echo "=== [resilient] crash (likely 22a RC watchdog) — resuming from latest checkpoint in ${RETRY_SLEEP}s ===" | tee -a "$LOG"
    sleep "$RETRY_SLEEP"
done
