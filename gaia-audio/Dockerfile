# gaia-audio: The Ears & Mouth - Half-duplex STT/TTS sensory service
# GPU-capable container for Whisper STT and RealtimeTTS

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
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
# Install dependencies (better layer caching)
# ═══════════════════════════════════════════════════════════════════════════

COPY gaia-audio/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install gaia-common and gaia-audio
COPY gaia-common/ /app/gaia_common/
COPY gaia-audio/ /app/

# Install gaia-audio in editable mode
RUN pip install -e /app/.[dev]

# Pre-accept Coqui XTTS v2 TOS for headless operation
RUN mkdir -p /home/gaia/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2 \
    && echo "I have read, understood and agreed to the Terms and Conditions." \
       > /home/gaia/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/tos_agreed.txt

# Set ownership
RUN chown -R gaia:gaia /app /home/gaia/.local

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENV PYTHONPATH="${PYTHONPATH}:/app"

USER gaia

CMD ["uvicorn", "gaia_audio.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
