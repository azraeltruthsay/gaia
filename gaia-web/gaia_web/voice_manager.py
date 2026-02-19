"""Discord Voice Manager — auto-answer, VAD, and audio pipeline.

Manages voice connections for GAIA's Discord bot. When a whitelisted user
joins a voice channel, GAIA auto-joins and enters a listen-transcribe-respond
loop using the gaia-audio service for STT/TTS.

Architecture:
  Discord Voice (48kHz stereo) → VAD segmentation → gaia-audio /transcribe
  → gaia-core /process_packet → gaia-audio /synthesize → Discord playback
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import struct
import subprocess
import threading
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    import discord

logger = logging.getLogger("GAIA.Web.Voice")

# ---------------------------------------------------------------------------
# Voice Whitelist — JSON persistence
# ---------------------------------------------------------------------------


class VoiceWhitelist:
    """Persistent whitelist of Discord users whose calls GAIA auto-answers.

    Also tracks all users GAIA has seen in Discord channels so the dashboard
    can offer a selectable list to toggle whitelisting.
    """

    def __init__(self, data_dir: str = "/app/data") -> None:
        self._path = Path(data_dir) / "voice_whitelist.json"
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"whitelisted": [], "seen_users": {}}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = self._path.read_text()
                self._data = json.loads(raw)
            except Exception:
                logger.warning("Failed to load voice whitelist; starting fresh")
                self._data = {"whitelisted": [], "seen_users": {}}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2))
        except Exception:
            logger.error("Failed to save voice whitelist", exc_info=True)

    # -- Whitelist operations --

    def add(self, user_id: str) -> None:
        with self._lock:
            if user_id not in self._data["whitelisted"]:
                self._data["whitelisted"].append(user_id)
                self._save()

    def remove(self, user_id: str) -> None:
        with self._lock:
            if user_id in self._data["whitelisted"]:
                self._data["whitelisted"].remove(user_id)
                self._save()

    def is_whitelisted(self, user_id: str) -> bool:
        with self._lock:
            return user_id in self._data["whitelisted"]

    def get_whitelisted(self) -> list[str]:
        with self._lock:
            return list(self._data["whitelisted"])

    # -- Seen user tracking --

    def record_seen(self, user_id: str, name: str, guild_id: str | None = None) -> None:
        with self._lock:
            entry = self._data["seen_users"].get(user_id, {
                "name": name,
                "last_seen": datetime.now().isoformat(),
                "guild_ids": [],
            })
            entry["name"] = name
            entry["last_seen"] = datetime.now().isoformat()
            if guild_id and guild_id not in entry.get("guild_ids", []):
                entry.setdefault("guild_ids", []).append(guild_id)
            self._data["seen_users"][user_id] = entry
            # Save periodically — skip if called very frequently
            self._save()

    def get_seen_users(self) -> list[dict]:
        with self._lock:
            result = []
            for uid, info in self._data["seen_users"].items():
                result.append({
                    "user_id": uid,
                    "name": info.get("name", "Unknown"),
                    "last_seen": info.get("last_seen", ""),
                    "guild_ids": info.get("guild_ids", []),
                    "whitelisted": uid in self._data["whitelisted"],
                })
            return sorted(result, key=lambda u: u["name"].lower())


# ---------------------------------------------------------------------------
# VAD (Voice Activity Detection)
# ---------------------------------------------------------------------------


class SimpleVAD:
    """Energy-based voice activity detection with webrtcvad fallback.

    Segments continuous audio into utterances based on silence detection.
    Operates on 16kHz mono 16-bit PCM frames.
    """

    def __init__(
        self,
        silence_threshold_ms: int = 800,
        min_speech_ms: int = 300,
        max_utterance_seconds: int = 30,
    ) -> None:
        self.silence_threshold_frames = silence_threshold_ms // 20  # 20ms per frame
        self.min_speech_frames = min_speech_ms // 20
        self.max_frames = (max_utterance_seconds * 1000) // 20
        self._vad = None
        self._init_vad()

        # State
        self._buffer: list[bytes] = []
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False

    def _init_vad(self) -> None:
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(2)  # Mode 2: moderate aggressiveness
        except ImportError:
            logger.warning("webrtcvad not available; using energy-based VAD fallback")

    def feed_frame(self, frame_16k_mono: bytes) -> bytes | None:
        """Feed a 20ms frame of 16kHz mono 16-bit PCM.

        Returns the complete utterance bytes when end-of-speech is detected,
        or None if still accumulating.
        """
        is_speech = self._detect_speech(frame_16k_mono)

        if is_speech:
            self._buffer.append(frame_16k_mono)
            self._speech_frames += 1
            self._silence_frames = 0
            self._in_speech = True
        elif self._in_speech:
            self._buffer.append(frame_16k_mono)  # Include trailing silence
            self._silence_frames += 1

        # Flush if silence threshold exceeded after enough speech
        if (
            self._in_speech
            and self._silence_frames >= self.silence_threshold_frames
            and self._speech_frames >= self.min_speech_frames
        ):
            return self._flush()

        # Safety: flush if utterance is too long
        if len(self._buffer) >= self.max_frames:
            return self._flush()

        return None

    def _detect_speech(self, frame: bytes) -> bool:
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, 16000)
            except Exception:
                pass
        # Energy-based fallback
        if len(frame) < 4:
            return False
        samples = struct.unpack(f"<{len(frame) // 2}h", frame)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return rms > 300  # Empirical threshold

    def _flush(self) -> bytes:
        audio = b"".join(self._buffer)
        self._buffer.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
        return audio

    def reset(self) -> None:
        self._buffer.clear()
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False


# ---------------------------------------------------------------------------
# Audio format conversion helpers
# ---------------------------------------------------------------------------


def pcm_48k_stereo_to_16k_mono(pcm_data: bytes) -> bytes:
    """Convert 48kHz stereo 16-bit PCM to 16kHz mono 16-bit PCM using FFmpeg."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "s16le", "-ar", "48000", "-ac", "2", "-i", "pipe:0",
                "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
            ],
            input=pcm_data,
            capture_output=True,
            timeout=10,
        )
        if proc.returncode != 0:
            logger.error("FFmpeg downsample failed: %s", proc.stderr.decode()[:200])
            return b""
        return proc.stdout
    except Exception:
        logger.error("PCM conversion failed", exc_info=True)
        return b""


def pcm_to_wav_base64(pcm_16k_mono: bytes, sample_rate: int = 16000) -> str:
    """Wrap raw 16-bit PCM in a WAV header and return base64-encoded string."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_16k_mono)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Voice Manager — connection orchestrator
# ---------------------------------------------------------------------------


class VoiceManager:
    """Manages GAIA's Discord voice connections and the audio pipeline.

    Handles auto-joining when whitelisted users enter voice channels,
    listening via VAD, and the transcribe → think → speak loop.
    """

    def __init__(
        self,
        core_endpoint: str,
        audio_endpoint: str,
        whitelist: VoiceWhitelist,
        voice_config: dict | None = None,
    ) -> None:
        self.core_endpoint = core_endpoint
        self.audio_endpoint = audio_endpoint
        self.whitelist = whitelist

        cfg = voice_config or {}
        self._silence_threshold_ms = cfg.get("silence_threshold_ms", 800)
        self._min_speech_ms = cfg.get("min_speech_ms", 300)
        self._max_utterance_s = cfg.get("max_utterance_seconds", 30)

        self._vc: discord.VoiceClient | None = None
        self._listen_task: asyncio.Task | None = None
        self._speaking = False
        self._state = "disconnected"  # disconnected | listening | transcribing | responding | speaking
        self._channel_name: str | None = None
        self._connected_since: float | None = None
        self._connected_user: str | None = None
        self._processing_lock = asyncio.Lock()

    # -- Public status --

    def get_status(self) -> dict:
        connected = self._vc is not None and self._vc.is_connected()
        return {
            "connected": connected,
            "channel_name": self._channel_name if connected else None,
            "connected_user": self._connected_user,
            "duration_seconds": round(time.monotonic() - self._connected_since, 1) if self._connected_since and connected else 0,
            "state": self._state if connected else "disconnected",
        }

    # -- Voice state events (called from discord_interface.py) --

    async def handle_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """React to users joining/leaving voice channels."""
        import discord as _discord

        # Record all users as seen (for dashboard user list)
        guild_id = str(member.guild.id) if member.guild else None
        self.whitelist.record_seen(str(member.id), member.display_name, guild_id)

        # Ignore bot's own voice state changes
        if member.bot:
            return

        user_id = str(member.id)
        is_whitelisted = self.whitelist.is_whitelisted(user_id)

        # User joined a voice channel
        if after.channel is not None and before.channel != after.channel:
            if is_whitelisted and self._vc is None:
                logger.info("Whitelisted user %s joined %s — auto-joining", member.display_name, after.channel.name)
                self._connected_user = member.display_name
                await self._join_channel(after.channel)

        # User left a voice channel
        if before.channel is not None and after.channel != before.channel:
            if self._vc and self._vc.channel == before.channel:
                # Check if any whitelisted users remain in the channel
                remaining = [
                    m for m in before.channel.members
                    if not m.bot and self.whitelist.is_whitelisted(str(m.id))
                ]
                if not remaining:
                    logger.info("No whitelisted users remain in %s — disconnecting", before.channel.name)
                    await self.disconnect()

    # -- Connection management --

    async def _join_channel(self, channel: discord.VoiceChannel) -> None:
        """Join a voice channel and start listening."""
        try:
            self._vc = await channel.connect()
            self._channel_name = channel.name
            self._connected_since = time.monotonic()
            self._state = "listening"
            logger.info("Connected to voice channel: %s", channel.name)

            # Start the listen loop
            self._listen_task = asyncio.create_task(self._listen_loop())
        except Exception:
            logger.error("Failed to join voice channel %s", channel.name, exc_info=True)
            self._vc = None
            self._state = "disconnected"

    async def disconnect(self) -> None:
        """Disconnect from voice and clean up."""
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._vc and self._vc.is_connected():
            await self._vc.disconnect()
        self._vc = None
        self._state = "disconnected"
        self._channel_name = None
        self._connected_since = None
        self._connected_user = None
        logger.info("Disconnected from voice")

    # -- Listen loop (captures audio, segments via VAD, processes utterances) --

    async def _listen_loop(self) -> None:
        """Continuously capture audio from Discord voice and process utterances.

        Discord delivers 20ms Opus frames. We use the VoiceClient's audio
        receiver to get decoded PCM, downsample, run through VAD, and process
        complete utterances.
        """
        vad = SimpleVAD(
            silence_threshold_ms=self._silence_threshold_ms,
            min_speech_ms=self._min_speech_ms,
            max_utterance_seconds=self._max_utterance_s,
        )

        # Accumulate raw PCM from the voice connection
        pcm_buffer = bytearray()
        # Discord sends 20ms of 48kHz stereo = 3840 bytes per frame
        frame_size_48k_stereo = 3840  # 20ms * 48000 * 2ch * 2bytes

        logger.info("Voice listen loop started")

        try:
            while self._vc and self._vc.is_connected():
                if self._speaking:
                    # Don't capture while GAIA is speaking (prevent echo)
                    await asyncio.sleep(0.1)
                    continue

                # Read audio from voice connection
                # discord.py uses a Sink or recv() depending on version
                try:
                    data = await asyncio.wait_for(
                        self._read_voice_audio(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                if not data:
                    await asyncio.sleep(0.02)
                    continue

                pcm_buffer.extend(data)

                # Process in frame_size_48k_stereo chunks
                while len(pcm_buffer) >= frame_size_48k_stereo:
                    frame_48k = bytes(pcm_buffer[:frame_size_48k_stereo])
                    del pcm_buffer[:frame_size_48k_stereo]

                    # Downsample to 16kHz mono
                    frame_16k = pcm_48k_stereo_to_16k_mono(frame_48k)
                    if not frame_16k:
                        continue

                    # Feed to VAD
                    utterance = vad.feed_frame(frame_16k)
                    if utterance is not None:
                        logger.info("Utterance detected (%d bytes)", len(utterance))
                        # Process in background (don't block the listen loop)
                        asyncio.create_task(self._process_utterance(utterance))

        except asyncio.CancelledError:
            logger.info("Voice listen loop cancelled")
        except Exception:
            logger.error("Voice listen loop error", exc_info=True)

    async def _read_voice_audio(self) -> bytes | None:
        """Read raw PCM audio from the voice connection.

        Uses discord.py's audio receive capabilities. Returns 20ms of
        48kHz stereo PCM (3840 bytes) or None if no data available.
        """
        if not self._vc or not self._vc.is_connected():
            return None

        # discord.py 2.x: use the recv() method or listen with a sink
        # The exact API depends on the discord.py version and whether
        # we're using pycord or standard discord.py
        try:
            # Try the listen/recv pattern
            if hasattr(self._vc, "recv"):
                return self._vc.recv()
            # Fallback: read from the underlying socket
            if hasattr(self._vc, "ws") and self._vc.ws:
                return await self._vc.ws.recv()
        except Exception:
            pass
        return None

    # -- Utterance processing pipeline --

    async def _process_utterance(self, pcm_16k_mono: bytes) -> None:
        """Transcribe → think → speak pipeline for a single utterance."""
        async with self._processing_lock:
            try:
                # 1. Transcribe via gaia-audio
                self._state = "transcribing"
                text = await self._transcribe(pcm_16k_mono)
                if not text or len(text.strip()) < 2:
                    logger.debug("Transcription empty or too short; skipping")
                    self._state = "listening"
                    return

                logger.info("Transcribed: %s", text[:100])

                # 2. Process via gaia-core
                self._state = "responding"
                response_text = await self._get_response(text)
                if not response_text:
                    logger.warning("No response from gaia-core")
                    self._state = "listening"
                    return

                logger.info("Response: %s", response_text[:100])

                # 3. Synthesize via gaia-audio
                self._state = "speaking"
                self._speaking = True
                await self._speak(response_text)
                self._speaking = False

            except Exception:
                logger.error("Utterance processing failed", exc_info=True)
            finally:
                self._speaking = False
                if self._vc and self._vc.is_connected():
                    self._state = "listening"

    async def _transcribe(self, pcm_16k_mono: bytes) -> str | None:
        """Send audio to gaia-audio /transcribe and return text."""
        audio_b64 = pcm_to_wav_base64(pcm_16k_mono)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.audio_endpoint}/transcribe",
                    json={"audio_base64": audio_b64, "sample_rate": 16000},
                )
                if resp.status_code == 200:
                    return resp.json().get("text", "")
                logger.error("Transcribe failed: %d %s", resp.status_code, resp.text[:200])
        except Exception:
            logger.error("Transcribe request failed", exc_info=True)
        return None

    async def _get_response(self, text: str) -> str | None:
        """Send transcribed text to gaia-core as a CognitionPacket and get response."""
        from gaia_common.protocols.cognition_packet import (
            CognitionPacket, Header, Persona, Origin, OutputRouting,
            DestinationTarget, Content, DataField, OutputDestination,
            PersonaRole, Routing, Model, OperationalStatus, SystemTask,
            Intent, Context, SessionHistoryRef, Constraints, Reasoning,
            Response, Governance, Safety, Metrics, TokenUsage, Status,
            PacketState, ToolRoutingState, TargetEngine,
        )

        packet_id = str(uuid.uuid4())
        user_id = self._connected_user or "voice_user"
        session_id = f"discord_voice_{user_id}"

        packet = CognitionPacket(
            version="0.2",
            header=Header(
                datetime=datetime.now().isoformat(),
                session_id=session_id,
                packet_id=packet_id,
                sub_id="0",
                persona=Persona(
                    identity_id="default_user",
                    persona_id="default_persona",
                    role=PersonaRole.DEFAULT,
                    tone_hint="conversational",
                ),
                origin=Origin.USER,
                routing=Routing(target_engine=TargetEngine.PRIME, priority=5),
                model=Model(name="default_model", provider="default_provider", context_window_tokens=8192),
                output_routing=OutputRouting(
                    primary=DestinationTarget(
                        destination=OutputDestination.AUDIO,
                        metadata={"source": "discord_voice", "user": user_id},
                    ),
                    source_destination=OutputDestination.AUDIO,
                    addressed_to_gaia=True,
                ),
                operational_status=OperationalStatus(status="initialized"),
            ),
            intent=Intent(user_intent="chat", system_task=SystemTask.GENERATE_DRAFT, confidence=0.0),
            context=Context(
                session_history_ref=SessionHistoryRef(type="discord_voice", value=session_id),
                cheatsheets=[],
                constraints=Constraints(max_tokens=512, time_budget_ms=15000, safety_mode="strict"),
            ),
            content=Content(
                original_prompt=text,
                data_fields=[DataField(key="user_message", value=text, type="text")],
            ),
            reasoning=Reasoning(),
            response=Response(candidate="", confidence=0.0, stream_proposal=False),
            governance=Governance(safety=Safety(execution_allowed=False, dry_run=True)),
            metrics=Metrics(token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0), latency_ms=0),
            status=Status(finalized=False, state=PacketState.INITIALIZED, next_steps=[]),
            tool_routing=ToolRoutingState(),
        )
        packet.compute_hashes()

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.core_endpoint}/process_packet",
                    json=packet.to_serializable_dict(),
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    result = resp.json()
                    completed = CognitionPacket.from_dict(result)
                    return completed.response.candidate or None
                logger.error("Core response failed: %d", resp.status_code)
        except Exception:
            logger.error("Core request failed", exc_info=True)
        return None

    async def _speak(self, text: str) -> None:
        """Synthesize text and play through Discord voice."""
        if not self._vc or not self._vc.is_connected():
            return

        try:
            # Get audio from gaia-audio
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.audio_endpoint}/synthesize",
                    json={"text": text},
                )
                if resp.status_code != 200:
                    logger.error("Synthesize failed: %d", resp.status_code)
                    return
                data = resp.json()

            audio_b64 = data.get("audio_base64")
            if not audio_b64:
                logger.warning("No audio in synthesis response")
                return

            audio_bytes = base64.b64decode(audio_b64)
            sample_rate = data.get("sample_rate", 22050)

            # Use FFmpeg to convert to Discord's format (48kHz stereo s16le)
            # and play via FFmpegPCMAudio
            import discord as _discord

            # Write audio to a temporary pipe and play via FFmpeg
            audio_source = _discord.FFmpegPCMAudio(
                io.BytesIO(audio_bytes),
                pipe=True,
                before_options=f"-f s16le -ar {sample_rate} -ac 1",
                options="-ar 48000 -ac 2",
            )

            # Play and wait for completion
            play_done = asyncio.Event()

            def after_play(error):
                if error:
                    logger.error("Voice playback error: %s", error)
                play_done.set()

            self._vc.play(audio_source, after=after_play)

            # Wait for playback to finish (max 60s safety)
            try:
                await asyncio.wait_for(play_done.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                if self._vc.is_playing():
                    self._vc.stop()

        except Exception:
            logger.error("Speech playback failed", exc_info=True)
