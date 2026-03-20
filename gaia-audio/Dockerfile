# gaia-audio: The Ears & Mouth — Three-tier Qwen3 STT/TTS sensory service
# STT:  Qwen3-ASR-0.6B (GPU) — speech recognition
# TTS:  Qwen3-TTS-0.6B (CPU, voice clone) / Qwen3-TTS-1.7B (GPU, on-demand)
# Fallback: espeak-ng (emergency TTS if Qwen3 unavailable)

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libportaudio2 \
    portaudio19-dev \
    espeak-ng \
    libespeak-ng1 \
    libsndfile1 \
    libpulse0 \
    libgomp1 \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (match host UID)
RUN groupadd -g 1000 gaia && useradd -u 1000 -g gaia -m -s /bin/bash gaia
ENV HOME=/home/gaia
ENV USER=gaia

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ═══════════════════════════════════════════════════════════════════════════
# Install dependencies
# ═══════════════════════════════════════════════════════════════════════════

COPY candidates/gaia-audio/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Qwen3 audio models — STT and TTS inference packages
RUN pip install --no-cache-dir qwen-asr qwen-tts

# Copy and install gaia-common and gaia-audio
COPY candidates/gaia-common/ /app/gaia_common/
COPY candidates/gaia-audio/ /app/

# Install gaia-audio in editable mode
RUN pip install -e /app/.[dev]

# Set ownership
RUN chown -R gaia:gaia /app /home/gaia/.local

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENV PYTHONPATH="${PYTHONPATH}:/app"

USER gaia

CMD ["uvicorn", "gaia_audio.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
