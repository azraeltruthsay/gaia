# 📜 GAIA-AUDIO Module Contract

## 🎭 Role
The **Ears & Mouth** of GAIA. Responsible for three-tier Qwen3 STT/TTS sensory processing, managing real-time transcription and synthesis with auto-routing between CPU and GPU tiers.

## 🔌 API Interface
- **Endpoint:** `http://gaia-audio:8080`
- **Protocol:** REST/HTTP
- **Contract Definition:** [contract.yaml](./contract.yaml)
- **Key Endpoints:** `/transcribe` (STT), `/synthesize` (TTS), `/status` (WebSocket).

## ⚙️ Configuration
- **Source File:** [config.json](./config.json)
- **Key Parameters:**
    - `stt_model`: Path to Whisper or Qwen STT model.
    - `tts_engine`: Engine selection (e.g., XTTS, Piper).
    - `vram_budget_mb`: Memory limit for GPU audio tasks.

## 🛠️ Integration
Resolve via `config.endpoints["audio"]`. Use POST `/transcribe` for audio processing and POST `/synthesize` for generating speech.
