"""Benchmark utility — compare Whisper model sizes on this hardware."""

from __future__ import annotations

import time

import numpy as np


MODELS = ["tiny.en", "base.en", "small.en", "medium.en"]
SAMPLE_DURATION = 5.0  # seconds of synthetic speech-like audio
SAMPLE_RATE = 16000


def _generate_reference_audio() -> np.ndarray:
    """Generate a deterministic audio clip for consistent benchmarking.

    Uses a mix of sine waves to approximate speech-band energy.
    Real transcription quality comes from actual speech — this just
    exercises the model's compute path at a known duration.
    """
    rng = np.random.RandomState(42)
    t = np.linspace(0, SAMPLE_DURATION, int(SAMPLE_RATE * SAMPLE_DURATION), dtype=np.float32)
    # Speech-band frequencies with some noise
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 2000 * t)
        + 0.05 * rng.randn(len(t)).astype(np.float32)
    )
    return signal / np.abs(signal).max()


def _record_reference_audio(duration: float = SAMPLE_DURATION) -> np.ndarray:
    """Record live audio from the default mic for benchmarking."""
    import sounddevice as sd

    print(f"\n  Recording {duration:.0f}s of audio — speak a test sentence...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("  Done recording.\n")
    return audio[:, 0]


def run_benchmark(
    latency_target: float = 2.0,
    compute_type: str = "int8",
    cpu_threads: int = 0,
    use_mic: bool = False,
) -> None:
    """Run all models and print a comparison table.

    Args:
        latency_target: max acceptable latency in seconds for recommendation
        compute_type: CTranslate2 compute type
        cpu_threads: 0 = auto
        use_mic: if True, record live audio instead of using synthetic
    """
    from faster_whisper import WhisperModel

    if use_mic:
        audio = _record_reference_audio()
    else:
        audio = _generate_reference_audio()

    duration = len(audio) / SAMPLE_RATE

    # Header
    print()
    print(f"  Benchmarking on {duration:.0f}s audio | compute={compute_type} | threads={cpu_threads or 'auto'}")
    print()
    print(f"  {'Model':<14} {'Load':>6}  {'Transcribe':>11}  {'RTF':>6}  {'Est. 5s':>8}  Output")
    print(f"  {'─' * 14} {'─' * 6}  {'─' * 11}  {'─' * 6}  {'─' * 8}  {'─' * 30}")

    best_model = None

    for model_name in MODELS:
        # Load
        t0 = time.monotonic()
        try:
            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
        except Exception as e:
            print(f"  {model_name:<14} FAILED to load: {e}")
            continue
        load_time = time.monotonic() - t0

        # Transcribe
        t0 = time.monotonic()
        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=3,
            vad_filter=False,  # synthetic audio won't pass VAD
        )
        text = " ".join(seg.text for seg in segments).strip()
        transcribe_time = time.monotonic() - t0

        rtf = transcribe_time / duration
        est_5s = rtf * 5.0

        # Recommendation marker
        marker = ""
        if est_5s <= latency_target:
            best_model = model_name
            marker = " ★"

        # Truncate output for display
        preview = text[:40] + "…" if len(text) > 40 else text
        preview = preview or "(no output)"

        print(
            f"  {model_name:<14} {load_time:5.1f}s  {transcribe_time:10.1f}s  {rtf:5.2f}x  {est_5s:7.1f}s  {preview!r}{marker}"
        )

        # Free memory before loading next model
        del model

    print()
    if best_model:
        print(f"  ★ Recommended: {best_model} (fits within {latency_target:.1f}s target)")
    else:
        print(f"  No model fits the {latency_target:.1f}s target. Consider --latency-target or a faster machine.")
    print()
