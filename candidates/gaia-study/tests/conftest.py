"""Shared fixtures for gaia-study tests."""

import pytest
import tempfile
from pathlib import Path

from gaia_study.study_mode_manager import StudyModeManager


@pytest.fixture
def tmp_adapter_dir(tmp_path):
    """Provide a temporary directory for adapter storage."""
    return tmp_path / "adapters"


@pytest.fixture
def study_manager(tmp_adapter_dir):
    """Create a StudyModeManager with test-safe configuration."""
    config = {
        "governance": {
            "forbidden_patterns": ["CLASSIFIED", "TOP SECRET"],
            "max_session_adapters": 3,
            "max_user_adapters": 10,
        },
        "max_training_time_seconds": 60,
        "max_training_samples": 100,
        "max_training_content_kb": 50,
        "use_real_training": False,  # Always simulate in tests
    }
    return StudyModeManager(config=config, adapter_base_dir=str(tmp_adapter_dir))
