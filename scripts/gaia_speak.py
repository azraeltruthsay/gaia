#!/usr/bin/env python3
"""
GAIA Speak — host-side utility to synthesize and play text.

Fetches audio from gaia-audio /synthesize and plays it via ffplay or aplay.
"""

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import requests

DEFAULT_GAIA_AUDIO_URL = "http://localhost:8080"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SPEAK] %(levelname)s %(message)s")
logger = logging.getLogger("gaia_speak")

def play_audio(audio_bytes: bytes, sample_rate: int):
    """Play raw PCM audio via ffplay."""
    try:
        # Use ffplay for reliable raw PCM playback
        cmd = [
            "ffplay", "-autoexit", "-nodisp",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
            "-"
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        proc.communicate(input=audio_bytes)
    except FileNotFoundError:
        logger.error("ffplay not found. Install ffmpeg on host.")
        # Fallback to aplay (Alsa)
        try:
            cmd = ["aplay", "-r", str(sample_rate), "-f", "S16_LE", "-c", "1"]
            subprocess.run(cmd, input=audio_bytes, check=True)
        except FileNotFoundError:
            logger.error("aplay not found either. Cannot play audio.")

def save_audio(audio_bytes: bytes, sample_rate: int, output_path: str):
    """Save raw PCM audio as a WAV file."""
    import wave
    try:
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        logger.info("Audio saved to %s", output_path)
    except Exception as e:
        logger.error("Failed to save audio: %s", e)

def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Split text into manageable chunks at sentence boundaries."""
    chunks = []
    current_chunk = []
    current_len = 0
    
    # Simple split by sentence-ending punctuation
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sentence)
        current_len += len(sentence)
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def main():
    parser = argparse.ArgumentParser(description="GAIA Speak utility")
    parser.add_argument("text", nargs="?", help="Text to speak (if omitted, reads from stdin)")
    parser.add_argument("--file", help="Path to text file to speak")
    parser.add_argument("--url", default=DEFAULT_GAIA_AUDIO_URL, help="gaia-audio URL")
    parser.add_argument("--voice", help="Voice ID/name override")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max chars per synthesis chunk")
    parser.add_argument("--output", help="Path to save output audio file (e.g. out.wav)")
    parser.add_argument("--no-play", action="store_true", help="Skip playback (useful with --output)")
    args = parser.parse_args()

    text = ""
    if args.file:
        with open(args.file, "r") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    if not text.strip():
        logger.warning("No text provided to speak.")
        return

    # Clean up text (remove model artifacts)
    text = text.replace("[Prime]", "").replace("[Lite]", "").strip()
    
    chunks = chunk_text(text, args.chunk_size)
    logger.info("Synthesizing %d chars in %d chunks...", len(text), len(chunks))
    
    all_audio_bytes = b""
    sample_rate = 22050

    for i, chunk in enumerate(chunks):
        logger.info("  Processing chunk %d/%d (%d chars)...", i+1, len(chunks), len(chunk))
        try:
            payload = {"text": chunk}
            if args.voice:
                payload["voice"] = args.voice
                
            resp = requests.post(f"{args.url}/synthesize", json=payload, timeout=300)
            if resp.status_code != 200:
                logger.error("Synthesis failed for chunk %d (HTTP %d): %s", i+1, resp.status_code, resp.text)
                continue
                
            data = resp.json()
            audio_b64 = data.get("audio_base64")
            if not audio_b64:
                logger.error("No audio returned for chunk %d.", i+1)
                continue
                
            chunk_bytes = base64.b64decode(audio_b64)
            sample_rate = data.get("sample_rate", 22050)
            all_audio_bytes += chunk_bytes
            
            if not args.no_play:
                logger.info("  Playing chunk %d (%d bytes)...", i+1, len(chunk_bytes))
                play_audio(chunk_bytes, sample_rate)
            
        except Exception as e:
            logger.error("Speak failed for chunk %d: %s", i+1, e)

    if args.output and all_audio_bytes:
        save_audio(all_audio_bytes, sample_rate, args.output)

if __name__ == "__main__":
    main()
