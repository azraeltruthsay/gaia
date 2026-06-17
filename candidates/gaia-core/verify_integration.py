#!/usr/bin/env python3
"""
Comprehensive Integration Verification Script (Milestone 4 - R4).
Simulates and verifies:
  1. A2 Thought Seed: Parsing and saving thought seeds from LLM output.
  2. A3 Observer Integration/Health Probe: Observer degradation and health status reflection.
  3. A4 Affective Chitchat: Dynamic affect ingestion (Samvega), drive bump, mathematical decay,
     and operational vital trimming in greeting flow.
"""

import os
import sys
import math
import shutil
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "candidates" / "gaia-core"))
sys.path.insert(0, str(PROJECT_ROOT / "candidates" / "gaia-common"))

# Setup basic logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("VerifyIntegration")

# Import packages
from gaia_common.protocols.cognition_packet import (
    CognitionPacket,
    DataField,
)
from gaia_core.config import Config
from gaia_core.cognition import thought_seed
from gaia_core.utils.output_router import route_output
from gaia_core.utils.stream_observer import observer_health, _OBS_HEALTH
from gaia_core.cognition import affect_runtime
from gaia_core.cognition.affect_appraiser import note_samvega
from gaia_core.utils.prompt_builder import build_from_packet
from gaia_common.utils.knowledge_graph import KnowledgeGraph
from gaia_common.utils.affect_kg import AffectKG
from gaia_core.main import health_check

class DummyAIManager:
    def __init__(self, config):
        self.config = config

def run_thought_seed_verification(temp_dir: Path):
    """
    Verify A2 Thought Seed: Directives in response text are parsed and persisted.
    """
    logger.info("=== STEP 1: Verifying A2 Thought Seed ===")
    
    # Path SEEDS_DIR to a temporary testing folder
    test_seeds_dir = temp_dir / "seeds"
    test_seeds_dir.mkdir(parents=True, exist_ok=True)
    orig_seeds_dir = thought_seed.SEEDS_DIR
    thought_seed.SEEDS_DIR = test_seeds_dir
    
    try:
        config = Config()
        ai_manager = DummyAIManager(config)
        packet = CognitionPacket()
        packet.header.packet_id = "test-pkt-a2"
        packet.intent.primary_goal = "Test user query for thought seed"
        
        response_text = (
            "I should check the database configuration.\n"
            "THOUGHT_SEED: This is a novel pattern showing that memory constraints are too low under pressure."
        )
        
        result = route_output(response_text, packet, ai_manager, "test-session", "cli")
        
        # Verify result side effects
        side_effects = result.get("side_effects", [])
        thought_seed_effect = next((se for se in side_effects if se["type"] == "thought_seed"), None)
        assert thought_seed_effect is not None, "Thought seed side effect not returned by route_output"
        assert thought_seed_effect["ok"] is True, "Thought seed saving failed in side effects"
        
        # Verify the file exists and has correct content
        unreviewed = thought_seed.list_unreviewed_seeds()
        assert len(unreviewed) == 1, f"Expected 1 unreviewed seed, found {len(unreviewed)}"
        
        saved_file, saved_data = unreviewed[0]
        assert saved_data["seed_type"] == "general"
        assert "memory constraints are too low" in saved_data["seed"]
        assert saved_data["context"]["packet_id"] == "test-pkt-a2"
        
        logger.info("✓ A2 Thought Seed successfully saved and verified via route_output.")
    finally:
        # Restore original SEEDS_DIR
        thought_seed.SEEDS_DIR = orig_seeds_dir

async def run_observer_health_verification():
    """
    Verify A3 Observer Integration / Health Probe.
    """
    logger.info("=== STEP 2: Verifying A3 Observer Health Probe ===")
    
    # Backup baseline health state
    orig_health = dict(_OBS_HEALTH)
    
    try:
        # 1. Reset health and verify it starts healthy
        _OBS_HEALTH.update({"count": 0, "fail": 0, "last_ts": 0.0, "last_fail_ts": 0.0})
        h_init = observer_health()
        assert h_init["healthy"] is True, "Initial observer health should be healthy"
        
        # 2. Simulate high failure rate (3 failures out of 5 observations, rate = 0.6 >= 0.5)
        _OBS_HEALTH.update({
            "count": 5,
            "fail": 3,
            "last_ts": datetime.now(timezone.utc).timestamp(),
            "last_fail_ts": datetime.now(timezone.utc).timestamp()
        })
        h_degraded = observer_health()
        assert h_degraded["healthy"] is False, f"Observer health should be degraded (rate = {h_degraded['fail_rate']})"
        
        # 3. Call the HTTP health check endpoint function from main.py
        # Use an environment variable to prevent main.py from trying to query a live Core server if none is running
        os.environ["CORE_CPU_ENDPOINT"] = "http://invalid-endpoint-nonexistent"
        
        resp = await health_check()
        assert resp.status_code == 200, f"Expected status 200 from health endpoint, got {resp.status_code}"
        
        import json
        body = json.loads(resp.body.decode())
        
        assert body["status"] == "degraded", f"Expected response status 'degraded', got {body['status']}"
        assert "observer degraded" in body["inference_detail"], f"Expected observer degradation message, got {body['inference_detail']}"
        assert body["observer"]["healthy"] is False, "Expected observer nested field to show healthy=False"
        
        logger.info("✓ A3 Observer Health Probe correctly reflects failure rates and degrades health check status.")
    finally:
        # Restore health state
        _OBS_HEALTH.update(orig_health)

def run_affective_chitchat_verification(temp_dir: Path):
    """
    Verify A4 Affective Ingestion, Decay, and Prompt Formatting.
    """
    logger.info("=== STEP 3: Verifying A4 Affective Ingestion, Decay, and Prompt Formatting ===")
    
    # 1. Initialize a clean temporary KG and AffectKG database
    kg_path = temp_dir / "affect_kg.sqlite"
    kg = KnowledgeGraph(db_path=str(kg_path))
    affect_kg = AffectKG(kg)
    affect_runtime.reset_for_tests(affect_kg)
    
    # Setup environment variables for affect appraisal
    os.environ["AFFECT_APPRAISAL_ENABLED"] = "1"
    
    try:
        # A4.1 Ingestion
        # Verify that note_samvega() raises coherence drive
        note_samvega(weight=0.8, root_cause="validation test misalignment")
        
        # Retrieve snapshot
        snapshot = affect_runtime.current_affect_snapshot()
        drives = snapshot.get("drives", {})
        assert "coherence" in drives, "Coherence drive not found in drives snap after Samvega ingestion"
        
        initial_val = drives["coherence"]
        assert initial_val > 0.0, f"Coherence drive value should be positive, got {initial_val}"
        logger.info(f"Ingested Samvega. Initial Coherence drive value: {initial_val:.4f}")
        
        # A4.2 Decay
        # Verify decay using the 1-day half-life.
        # Compute flattened affect exactly 24 hours later.
        now_baseline = datetime.now(timezone.utc)
        now_plus_1_day = now_baseline + timedelta(days=1)
        
        snap_after_1_day = affect_kg.flatten_current_affect(now=now_plus_1_day)
        decayed_val = snap_after_1_day.get("drives", {}).get("coherence", 0.0)
        
        logger.info(f"Coherence drive value after 1 day: {decayed_val:.4f}")
        expected_decayed = initial_val * 0.5
        assert math.isclose(decayed_val, expected_decayed, rel_tol=1e-2), (
            f"Decayed value {decayed_val:.4f} deviates from expected half-life value {expected_decayed:.4f}"
        )
        logger.info("✓ Drive correctly decayed by exactly half (1-day half-life).")
        
        # A4.3 Organic Greeting Prompt Formatting
        # Construct a greeting intent packet
        packet = CognitionPacket()
        packet.intent.user_intent = "greeting"
        packet.content.original_prompt = "Hello! How are you doing today?"
        
        # Set some affect values above threshold (0.15) to make sure they are rendered
        affect_kg.record_feeling("curiosity", 0.65)
        affect_kg.record_feeling("fatigue", 0.70)
        
        # Set world state snap with operational metrics
        operational_metrics_raw = (
            "Clock: 2026-06-16 22:15:00 UTC\n"
            "User's local time: 2026-06-16 15:15:00 PDT (UTC-0700)\n"
            "Uptime: 3600s | load: 0.12 | mem: 4.2GB/16.0GB (26.2%)\n"
            "Context: in Discord\n"
            "Immune health: nominal\n"
            "Model paths: /models/gemma4-8b-e4b.gguf"
        )
        packet.content.data_fields.append(DataField(key="world_state_snapshot", value=operational_metrics_raw))
        
        # Build prompt
        prompt = build_from_packet(packet)
        system_content = next((msg["content"] for msg in prompt if msg["role"] == "system"), "")
        
        # Verification 1: Trim operational vitals
        # The system prompt should contain Clock/Context but NOT uptime or memory metrics
        assert "Clock:" in system_content, "Expected Clock: to be kept in system prompt"
        assert "Context: in Discord" in system_content, "Expected Context: to be kept in system prompt"
        assert "Uptime:" not in system_content, "Operational Uptime should be trimmed in greeting flow"
        assert "mem:" not in system_content, "Operational memory usage should be trimmed in greeting flow"
        assert "load:" not in system_content, "Operational load should be trimmed in greeting flow"
        assert "Model paths:" not in system_content, "Operational model paths should be trimmed in greeting flow"
        logger.info("✓ Operational vitals (Uptime, load, mem) successfully trimmed from greeting system prompt.")
        
        # Verification 2: declarative felt-fact, NOT a behavioral instruction.
        # The clean A4 design renders affect as a number-free "Inner weather:"
        # fact (her words to voice), and drops the mechanical 'Current Affect
        # (feels): x=0.62' stat lines on the casual path. It must NOT carry the
        # "express your affect" instruction (that backfires on Gemma4-E4B).
        assert "Inner weather:" in system_content, (
            "Expected a declarative 'Inner weather:' felt-fact in the greeting prompt"
        )
        # The felt fact is number-free: no raw affect stat line leaks in.
        assert "Current Affect (feels)" not in system_content, (
            "Mechanical 'Current Affect (feels):' stat line must NOT appear on the casual path"
        )
        # No leftover behavioral instruction from the prior (rejected) approach.
        assert "Organically express your current affect" not in system_content, (
            "The rejected 'express your affect' instruction must be gone"
        )
        # Isolate the felt line and confirm it carries no digits.
        _felt = next((ln for ln in system_content.splitlines() if ln.startswith("Inner weather:")), "")
        assert _felt and not any(c.isdigit() for c in _felt), f"Felt fact must be number-free, got: {_felt!r}"
        logger.info("✓ Affect reaches the greeting prompt as a number-free declarative felt-fact (no instruction).")
        
    finally:
        affect_runtime.reset_for_tests(None)

def main():
    logger.info("Starting Programmatic Verification Simulation...")
    
    # Create temporary scratchpad for test files
    temp_dir = Path("/tmp/gaia_verification_scratch")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        run_thought_seed_verification(temp_dir)
        
        # Run observer health probe async test
        asyncio.run(run_observer_health_verification())
        
        run_affective_chitchat_verification(temp_dir)
        
        logger.info("======================================================")
        logger.info("🎉 All programmatic integration checks passed successfully!")
        logger.info("======================================================")
        sys.exit(0)
    except AssertionError as ae:
        logger.error(f"❌ Verification ASSERTION failed: {ae}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Verification run encountered an unexpected exception: {e}")
        sys.exit(1)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
