"""Tests for StudyModeManager validation and data preparation."""

import pytest
from pathlib import Path

from gaia_study.study_mode_manager import (
    StudyModeManager,
    StudyModeState,
    TrainingConfig,
)


class TestContentValidation:
    """Test governance rules for training content."""

    def test_valid_content_passes(self, study_manager):
        is_valid, reason = study_manager.validate_content("Normal training text about Python.")
        assert is_valid is True
        assert reason == ""

    def test_forbidden_pattern_rejected(self, study_manager):
        is_valid, reason = study_manager.validate_content("This is CLASSIFIED material")
        assert is_valid is False
        assert "forbidden pattern" in reason.lower()

    def test_forbidden_pattern_case_insensitive(self, study_manager):
        is_valid, reason = study_manager.validate_content("top secret info here")
        assert is_valid is False

    def test_oversized_content_rejected(self, study_manager):
        # 50KB limit in test config
        big_content = "A" * (51 * 1024)
        is_valid, reason = study_manager.validate_content(big_content)
        assert is_valid is False
        assert "exceeds limit" in reason.lower()

    def test_content_just_under_limit(self, study_manager):
        content = "A" * (49 * 1024)
        is_valid, reason = study_manager.validate_content(content)
        assert is_valid is True


class TestDataPreparation:
    """Test training data preparation from source documents."""

    def test_prepare_from_files(self, study_manager, tmp_path):
        doc = tmp_path / "training_doc.txt"
        doc.write_text("This is a paragraph about Python programming.\n\n"
                       "This is another paragraph about machine learning and neural networks.")

        samples, metadata = study_manager.prepare_training_data(
            source_documents=[str(doc)],
            output_format="instruction",
        )
        assert len(samples) > 0
        assert metadata["total_samples"] == len(samples)
        assert metadata["output_format"] == "instruction"

    def test_prepare_skips_missing_files(self, study_manager):
        samples, metadata = study_manager.prepare_training_data(
            source_documents=["/nonexistent/file.txt"],
        )
        assert len(samples) == 0

    def test_prepare_skips_forbidden_content(self, study_manager, tmp_path):
        doc = tmp_path / "forbidden.txt"
        doc.write_text("This document is CLASSIFIED and should not be trained on.")

        samples, metadata = study_manager.prepare_training_data(
            source_documents=[str(doc)],
        )
        assert len(samples) == 0

    def test_prepare_limits_samples(self, study_manager, tmp_path):
        # Create a document that generates many samples
        content = "\n\n".join([f"Paragraph {i} with enough text to pass the 50 char minimum threshold for samples." for i in range(200)])
        doc = tmp_path / "big_doc.txt"
        doc.write_text(content)

        samples, metadata = study_manager.prepare_training_data(
            source_documents=[str(doc)],
        )
        # Config limits to 100 samples
        assert len(samples) <= 100

    def test_completion_format(self, study_manager, tmp_path):
        doc = tmp_path / "completion.txt"
        doc.write_text("A substantial paragraph about neural network architectures and their applications in modern AI systems.")

        samples, metadata = study_manager.prepare_training_data(
            source_documents=[str(doc)],
            output_format="completion",
        )
        assert metadata["output_format"] == "completion"
        # Completion samples have a "text" key
        if samples:
            assert "text" in samples[0]


class TestStatusAndCancel:
    """Test status reporting and cancellation."""

    def test_initial_state_is_idle(self, study_manager):
        status = study_manager.get_status()
        assert status["state"] == "idle"
        assert status["progress"] == 0.0

    def test_cancel_while_idle_returns_false(self, study_manager):
        assert study_manager.cancel_training() is False


class TestAdapterManagement:
    """Test adapter CRUD operations."""

    def test_list_adapters_empty(self, study_manager):
        adapters = study_manager.list_adapters()
        assert adapters == []

    def test_list_adapters_by_tier(self, study_manager):
        adapters = study_manager.list_adapters(tier=1)
        assert adapters == []

    def test_delete_tier1_protected(self, study_manager):
        assert study_manager.delete_adapter("any_adapter", tier=1) is False

    def test_delete_nonexistent_adapter(self, study_manager):
        assert study_manager.delete_adapter("ghost", tier=3) is False

    def test_tier_directory_mapping(self, study_manager):
        tier1_dir = study_manager._get_tier_directory(1)
        tier2_dir = study_manager._get_tier_directory(2)
        tier3_dir = study_manager._get_tier_directory(3)
        assert "tier1_global" in str(tier1_dir)
        assert "tier2_user" in str(tier2_dir)
        assert "tier3_session" in str(tier3_dir)
