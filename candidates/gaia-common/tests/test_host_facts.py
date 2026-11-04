"""Facts come from reads; what can't be read is left out."""
import subprocess
from unittest.mock import MagicMock

import pytest

from gaia_common.utils import host_facts as hf

CPUINFO = "processor\t: 0\nmodel name\t: Test Chip 9000\nprocessor\t: 1\nmodel name\t: Test Chip 9000\n"
MEMINFO = "MemTotal:       33554432 kB\nMemFree:         1048576 kB\nMemAvailable:   16777216 kB\n"


def test_reads_the_cpu_and_memory(tmp_path):
    (tmp_path / "c").write_text(CPUINFO)
    (tmp_path / "m").write_text(MEMINFO)
    assert hf._read_cpu(str(tmp_path / "c")) == {"model": "Test Chip 9000", "threads": 2}
    assert hf._read_memory(str(tmp_path / "m")) == {"total_gb": 32.0, "available_gb": 16.0}


def test_reads_the_gpu(monkeypatch):
    monkeypatch.setattr(hf.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hf.subprocess, "run", lambda *a, **k: MagicMock(stdout="Test GPU, 16384, 2048\n"))
    assert hf._read_gpu() == {"name": "Test GPU", "total_gb": 16.0, "used_gb": 2.0}


def test_no_gpu_tool_means_no_gpu_fact(monkeypatch):
    monkeypatch.setattr(hf.shutil, "which", lambda n: None)
    assert hf._read_gpu() is None


def test_a_failed_read_is_left_out_not_guessed(monkeypatch):
    monkeypatch.setattr(hf, "_cache", {})
    monkeypatch.setattr(hf, "ENGINES", {})

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 3)
    monkeypatch.setattr(hf, "_read_gpu", boom)
    facts = hf.host_facts()
    assert "gpu" not in facts
    assert "GPU" not in hf.host_facts_text(facts)


def test_reads_are_not_repeated_within_the_cache_window(monkeypatch):
    monkeypatch.setattr(hf, "_cache", {})
    calls = []
    assert hf._cached("x", lambda: calls.append(1) or 5) == 5
    assert hf._cached("x", lambda: calls.append(1) or 6) == 5
    assert len(calls) == 1


def test_the_text_says_what_was_read():
    text = hf.host_facts_text({"gpu": {"name": "Test GPU", "total_gb": 16.0, "used_gb": 2.0},
                               "cpu": {"model": "Test Chip", "threads": 2},
                               "engines": {"Core": {"model": "Gemma-X", "device": "cuda"}}})
    assert "Test GPU with 16.0 GB" in text and "Test Chip, 2 threads" in text
    assert "2.0 GB of GPU memory in use" in text and "Core is running Gemma-X on the CUDA" in text


def test_nothing_read_means_nothing_said():
    assert hf.host_facts_text({}) == "" and hf.machine_description({}) == ""


@pytest.mark.parametrize("text", ["What hardware are you running on?", "What GPU do you have?", "What are you running on right now?",
                                  "Tell me about your environment.", "You should know your system resources.",
                                  "How much RAM is free?", "What's your CPU?"])
def test_questions_about_the_machine(text):
    assert hf.asks_about_host(text)


@pytest.mark.parametrize("text", ["What is in your memory palace?", "What is a GPU?", "Tell me about the Raven.",
                                  "Who are you?", "How do I fix my computer?", "What do you think about rain?"])
def test_other_questions(text):
    assert not hf.asks_about_host(text)
