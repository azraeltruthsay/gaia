from pathlib import Path
from gaia_core.config import Config
# TODO: [GAIA-REFACTOR] vector_indexer.py module not yet migrated.
# from app.utils.vector_indexer import embed_gaia_reference
from .manager import GAIAState, load_models
import logging

logger = logging.getLogger("GAIA.Bootstrap")

VECTOR_INDEX_PATH = Path("./knowledge/system_reference/vector_store/gaia_rescue_index/index_store.json")
MANIFEST_PATH = VECTOR_INDEX_PATH / "vector_index_manifest.json"

def manifest_is_stale():
    try:
        if not MANIFEST_PATH.exists():
            return True
        import json
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for fname, recorded in manifest.get("files", {}).items():
            fpath = Path("./knowledge/system_reference/GAIA_Function_Map") / fname
            if not fpath.exists():
                return True
            if abs(int(fpath.stat().st_mtime) - int(recorded)) > 3:
                return True
        return False
    except Exception as e:
        logger.warning(f"Manifest check failed: {e}")
        return True

def bootstrap_gaia():
    config = Config()
    state = GAIAState(config=config)

    # Load models using your robust loader
    load_models(config, state.model_pool)
    if not state.model_pool.get("Prime"):
        logger.error("❌ No Prime model loaded; GAIA cannot boot.")
        raise RuntimeError("No Prime model loaded; GAIA cannot boot.")

    # Optionally: embed core knowledge if needed
    if not VECTOR_INDEX_PATH.exists() or manifest_is_stale():
        logger.info("⚠️ Vector index missing or stale. Re-embedding...")
        # embed_gaia_reference()

    # Return state for use in CLI or web
    return state
