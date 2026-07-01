"""
Activation stream routes for Mission Control.

SSE endpoint that tails ``/logs/activation_stream.jsonl`` in real-time,
giving the Neural Mind Map live visibility into per-token SAE feature
activations as Nano, Core, and Prime generate.

Also serves the SAE atlas (feature labels) for human-readable
visualization.
"""

import asyncio
import json
import logging
import os
import time

from fastapi import APIRouter
from starlette.responses import StreamingResponse

logger = logging.getLogger("GAIA.Web.Activations")

router = APIRouter(prefix="/api/activations", tags=["activations"])

_LOG_PATH = os.getenv("ACTIVATION_STREAM_PATH", "/logs/activation_stream.jsonl")
_ATLAS_DIR = os.getenv("SAE_ATLAS_DIR", "/shared/atlas/core")
_POLL_INTERVAL = 0.05  # 50 ms
_HEARTBEAT_INTERVAL = 2.0  # seconds


@router.get("/stream")
async def activation_stream(session_id: str = ""):
    """SSE endpoint — tails activation_stream.jsonl in real-time.

    Optional query params:
      - ``session_id`` — filter to a specific session's activations
    """
    return StreamingResponse(
        _event_generator(session_id=session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _event_generator(session_id: str = ""):
    """Async generator that tails the activation JSONL log and yields SSE frames."""
    last_heartbeat = 0.0
    file_pos = 0

    # Start from the end of the file (don't replay old events)
    try:
        file_pos = os.path.getsize(_LOG_PATH)
    except OSError:
        pass

    while True:
        lines_sent = False
        try:
            with open(_LOG_PATH, "r") as f:
                # Handle file rotation: if file shrank, start from beginning
                current_size = os.fstat(f.fileno()).st_size
                if current_size < file_pos:
                    file_pos = 0
                f.seek(file_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Filter by session_id if provided
                    if session_id and record.get("session_id", "") != session_id:
                        continue
                    yield f"data: {json.dumps(record)}\n\n"
                    lines_sent = True
                file_pos = f.tell()
        except OSError:
            pass  # file doesn't exist yet — wait for it

        now = time.monotonic()
        if not lines_sent and (now - last_heartbeat) >= _HEARTBEAT_INTERVAL:
            yield ":keepalive\n\n"
            last_heartbeat = now

        await asyncio.sleep(_POLL_INTERVAL)


_STREAM_TAIL_BYTES = 256 * 1024  # how far back to scan when synthesizing layers
_STREAM_MAX_LINES = 2000          # cap on lines parsed from the tail


def _discover_layers_from_stream(
    stream_path: str = None,
    *,
    tail_bytes: int = _STREAM_TAIL_BYTES,
    max_lines: int = _STREAM_MAX_LINES,
) -> dict:
    """Synthesize layer/tier info from the activation_stream.jsonl tail.

    GAIA_Project-874: when meta.json is stale or missing, the API
    derives the live layer set by scanning the tail of the stream.
    Each stream record carries ``tier`` and per-feature ``layer``, so
    no engine call is needed.

    Returns ``{"layers": sorted_list, "tier": str | None,
               "sample_count": int}``. Empty layers if the stream
    doesn't exist or has no parseable records.
    """
    path = stream_path or _LOG_PATH
    layers: set[int] = set()
    tiers: set[str] = set()
    sample_count = 0
    try:
        size = os.path.getsize(path)
    except OSError:
        return {"layers": [], "tier": None, "sample_count": 0}
    if size == 0:
        return {"layers": [], "tier": None, "sample_count": 0}
    try:
        with open(path, "rb") as f:
            f.seek(max(0, size - tail_bytes))
            chunk = f.read().decode("utf-8", errors="replace")
    except OSError:
        return {"layers": [], "tier": None, "sample_count": 0}
    # Drop the (likely truncated) first line when we seeked mid-file
    lines = chunk.splitlines()
    if len(lines) > 1 and tail_bytes < size:
        lines = lines[1:]
    # Limit how many we parse — recent lines are more representative
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        sample_count += 1
        if isinstance(rec.get("tier"), str):
            tiers.add(rec["tier"])
        feats = rec.get("features") or []
        if isinstance(feats, list):
            for feat in feats:
                if isinstance(feat, dict) and isinstance(feat.get("layer"), int):
                    layers.add(feat["layer"])
    # Tier inference: most-common is the "primary" but we only return
    # something definitive when the tail is dominated by one tier.
    tier = None
    if len(tiers) == 1:
        tier = next(iter(tiers))
    return {
        "layers": sorted(layers),
        "tier": tier,
        "sample_count": sample_count,
    }


def _load_atlas_meta(atlas_dir: str) -> dict:
    """Read meta.json or return an empty dict."""
    meta_path = os.path.join(atlas_dir, "meta.json")
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_atlas_meta(atlas_dir: str, payload: dict) -> bool:
    """Atomic-ish write of meta.json. Failures are logged but
    non-blocking — the synthesized result is still returned to the
    caller."""
    meta_path = os.path.join(atlas_dir, "meta.json")
    tmp_path = meta_path + ".tmp"
    try:
        os.makedirs(atlas_dir, exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, meta_path)
        return True
    except OSError as e:
        logger.debug("Could not write atlas meta.json (%s)", e)
        return False


@router.get("/atlas")
async def get_atlas(tier: str = "core"):
    """Return SAE feature labels for the mind map.

    ``tier`` selects the per-tier atlas dir (core → /shared/atlas/core,
    prime → /shared/atlas/prime). Core and Prime are SEPARATELY trained SAEs
    whose feature indices are not comparable, so labels must be served (and
    consumed) per tier. Defaults to core for backward compatibility.

    Reads ``meta.json`` if present; **synthesizes layer info from the
    activation stream tail** when meta.json is missing, stale, or has
    no ``layers`` field (GAIA_Project-874). When synthesis succeeds and
    differs from the on-disk meta.json, the file is rewritten so the
    next request hits the cache.

    Also merges any per-layer feature label files (``layer_N_labels.json``).
    """
    # Resolve the per-tier atlas dir. Core keeps _ATLAS_DIR (env-overridable +
    # test-patchable, defaults to /shared/atlas/core); prime resolves under the
    # atlas base. Unknown tier falls back to the core default.
    atlas_dir = os.path.join(_ATLAS_BASE, "prime") if tier == "prime" else _ATLAS_DIR

    result = {"layers": {}, "model": None, "timestamp": None, "tier": tier}

    on_disk = _load_atlas_meta(atlas_dir)
    if on_disk:
        result["model"] = on_disk.get("model")
        result["timestamp"] = on_disk.get("timestamp")
        if "layers" in on_disk:
            result["layers"] = on_disk["layers"]

    # Stage 1 (874): synthesize layer info from the live stream tail.
    # If the stream tells us layers that aren't in the on-disk meta.json,
    # the on-disk file is stale — overwrite with the synthesized layer
    # set so the UI matches what's actually being recorded. Only apply this
    # when the stream tail is dominated by the SAME tier we're serving —
    # otherwise a core-heavy stream would clobber prime's meta (and vice versa).
    stream_info = _discover_layers_from_stream()
    stream_layers = stream_info.get("layers") or []
    if stream_layers and stream_info.get("tier") == tier:
        on_disk_layers = on_disk.get("layers") if isinstance(on_disk.get("layers"), list) else None
        if on_disk_layers != stream_layers:
            logger.info(
                "Atlas meta layers differ from live stream — refreshing "
                "(tier=%s, on_disk=%s, stream=%s)",
                tier, on_disk_layers, stream_layers,
            )
            refreshed = {
                **on_disk,
                "layers": stream_layers,
                "tier": tier,
                "timestamp": time.time(),
                "source": "auto_from_activation_stream",
                "sample_count": stream_info.get("sample_count", 0),
            }
            _write_atlas_meta(atlas_dir, refreshed)
            # Reflect the refreshed layers in this response too. We use
            # the list-of-ints form here; the labels dict (populated
            # below from layer_N_labels.json) is keyed by str(layer).
            result["layers"] = {str(L): {"features": {}} for L in stream_layers}
            result["model"] = refreshed.get("model") or result["model"]
            result["timestamp"] = refreshed["timestamp"]

    # Always check for per-layer label files (layer_N_labels.json)
    try:
        for entry in os.listdir(atlas_dir):
            if entry.startswith("layer_") and entry.endswith("_labels.json"):
                try:
                    layer_idx = int(entry.split("_")[1])
                    with open(os.path.join(atlas_dir, entry), "r") as f:
                        labels = json.load(f)
                    # Merge — per-layer files override meta.json
                    layer_key = str(layer_idx)
                    # Coerce list-form layers into the labels-merge dict
                    if isinstance(result["layers"], list):
                        result["layers"] = {
                            str(L): {"features": {}} for L in result["layers"]
                        }
                    if layer_key not in result["layers"]:
                        result["layers"][layer_key] = {"features": {}}
                    result["layers"][layer_key]["features"].update(labels)
                except (ValueError, json.JSONDecodeError, OSError):
                    continue
    except OSError:
        pass

    return result


# Default canonical (top-k, discriminative) atlas tags per tier, both holding a
# synapse_graph.json (within-layer co-activation + cross-layer causal) — A4/72q.
_SYNAPSE_DEFAULT_TAG = {"core": "CORE_IDENTITY_V3_gguf", "prime": "PRIME_ABLITERATED_gguf"}
_ATLAS_BASE = os.getenv("SAE_ATLAS_BASE", "/shared/atlas")


@router.get("/synapse_graph")
async def get_synapse_graph(tier: str = "core", tag: str = ""):
    """Serve the assembled feature synapse graph for the mind map.

    Nodes = (layer, feature) with brain region (re-derived A4 map); edges carry
    kind=coactivation|causal, weights, polarity. Built by build_synapse_graph.py.
    """
    tag = tag or _SYNAPSE_DEFAULT_TAG.get(tier, "")
    path = os.path.join(_ATLAS_BASE, tier, tag, "synapse_graph.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except OSError:
        return {"error": f"no synapse_graph for {tier}/{tag}", "tier": tier, "tag": tag,
                "node_count": 0, "edge_count": 0, "nodes": [], "edges": []}
