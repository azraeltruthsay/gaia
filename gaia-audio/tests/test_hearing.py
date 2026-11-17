"""2026-11-17: her hearing sense measures what Core's audio tower can't."""
import numpy as np

from gaia_audio import hearing

SR = 16000


def _tone(freqs, secs=2.0, amp=0.3):
    t = np.arange(int(SR * secs)) / SR
    return sum(amp * np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)


def test_silence():
    m = hearing.measure(np.zeros(SR * 2), SR)
    assert m["silent"] and "silence" in hearing.hud_line(m, [])


def test_steady_tone_named():
    m = hearing.measure(_tone([440]), SR)
    assert m["tonal"] and m["notes"] == ["A4"] and abs(m["tones_hz"][0] - 440) < 5


def test_chord_is_three_tones():
    m = hearing.measure(_tone([329.6, 392.0, 493.9]), SR)   # E minor
    assert m["notes"] == ["E4", "G4", "B4"]
    assert "chord" in hearing.hud_line(m, [])


def test_rising_sweep():
    t = np.arange(SR * 2) / SR
    f = 300 * (4 ** (t / 2))                                # 300 -> 1200 Hz
    x = 0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    m = hearing.measure(x, SR)
    assert "rises" in m["movement"] and "tones_hz" not in m


def test_pulses_counted():
    x = np.zeros(SR * 3)
    tone = _tone([660], secs=0.2)
    for start in (0.2, 1.0, 1.8):
        i = int(start * SR)
        x[i:i + len(tone)] = tone
    assert hearing.measure(x, SR)["sounds"] == 3


def test_white_noise():
    rng = np.random.default_rng(0)
    m = hearing.measure(0.1 * rng.standard_normal(SR * 2), SR)
    assert not m["tonal"] and m["noise_colour"].startswith("flat")


def test_labels_in_line():
    line = hearing.hud_line(hearing.measure(_tone([440]), SR), [{"label": "Sine wave", "score": 0.8}])
    assert line.startswith("Your hearing measured this clip:") and "Sine wave (0.80)" in line
