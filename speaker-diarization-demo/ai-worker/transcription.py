"""
transcription.py — Faster-Whisper ASR module.

Inference uses CTranslate2 (not PyTorch). ONNX Runtime in the worker is
reserved for speaker embeddings in diarization.py.

Exposes two entry points:
  load_model()          — call once at startup
  transcribe_window()   — transcribe an audio array with an absolute time offset

The session buffer management is handled in main.py.  This module is stateless
per-call: it takes raw float32 audio, returns TranscriptSegment objects.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE: str = os.getenv("DEVICE", "cpu").strip().lower()
COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "int8")
MODEL_SIZE: str = os.getenv("MODEL_SIZE", "small.en")
LANGUAGE: str = os.getenv("LANGUAGE", "en").strip().lower()
BEAM_SIZE: int = int(os.getenv("BEAM_SIZE", "5"))
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
# CTranslate2 thread count for CPU (0 = library default, often 4).
try:
    _thr = int(os.getenv("WHISPER_CPU_THREADS", "0"))
except ValueError:
    _thr = 0
WHISPER_CPU_THREADS: int = max(0, _thr)

# Keep converted Whisper weights next to the repo (see .env.example).
_raw_whisper_cache = os.getenv("WHISPER_MODEL_CACHE_DIR", "").strip()
WHISPER_MODEL_CACHE_DIR: Optional[str] = (
    str(Path(_raw_whisper_cache).expanduser())
    if _raw_whisper_cache
    else None
)
VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.4"))
VAD_MIN_SILENCE_MS: int = int(os.getenv("VAD_MIN_SILENCE_MS", "200"))
VAD_SPEECH_PAD_MS: int = int(os.getenv("VAD_SPEECH_PAD_MS", "120"))

# Energy threshold — windows below this are considered silence and skipped.
VAD_ENERGY_THRESHOLD: float = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.0015"))

# Hallucination filters applied to every Whisper segment.
# avg_logprob: very negative values indicate the model was guessing.
MIN_AVG_LOGPROB: float = float(os.getenv("MIN_AVG_LOGPROB", "-1.0"))
# no_speech_prob close to 1 means Whisper itself thinks this is non-speech.
MAX_NO_SPEECH_PROB: float = float(os.getenv("MAX_NO_SPEECH_PROB", "0.6"))
# High compression ratio is the classic signature of repeated-text loops.
MAX_COMPRESSION_RATIO: float = float(os.getenv("MAX_COMPRESSION_RATIO", "2.4"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    segment_id: str
    text: str
    start: float          # absolute seconds from session start
    end: float
    is_final: bool
    confidence: Optional[float] = None
    words: Optional[List[TranscriptWord]] = None


# ---------------------------------------------------------------------------
# Module-level model handle
# ---------------------------------------------------------------------------

_whisper_model = None
_model_error: Optional[str] = None
_seg_counter: int = 0


def load_model() -> bool:
    global _whisper_model, _model_error
    try:
        from faster_whisper import WhisperModel  # type: ignore

        if DEVICE != "cpu":
            raise RuntimeError(
                f"DEVICE must be 'cpu' for this demo (got {DEVICE!r}). "
                "Set DEVICE=cpu in .env."
            )
        logger.info(
            "Loading Faster-Whisper '%s' (CTranslate2) device=%s compute=%s threads=%s",
            MODEL_SIZE,
            DEVICE,
            COMPUTE_TYPE,
            WHISPER_CPU_THREADS if WHISPER_CPU_THREADS else "default",
        )
        _whisper_model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
            download_root=WHISPER_MODEL_CACHE_DIR,
        )
        logger.info("Faster-Whisper loaded.")
        return True
    except Exception as exc:
        _model_error = str(exc)
        logger.error("Failed to load Faster-Whisper: %s", exc)
        return False


def is_model_ready() -> bool:
    return _whisper_model is not None


def get_model_error() -> Optional[str]:
    return _model_error


def get_model_config() -> Dict[str, Any]:
    return {
        "model": MODEL_SIZE,
        "language": LANGUAGE if LANGUAGE and LANGUAGE != "auto" else "auto",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "beam_size": BEAM_SIZE,
        "backend": "ctranslate2",
        "cpu_threads": WHISPER_CPU_THREADS,
    }


# ---------------------------------------------------------------------------
# Transcription — stateless per-call
# ---------------------------------------------------------------------------

# Kept for compatibility with session-based callers in tests.
class TranscriptionSession:
    """Thin wrapper kept for interface compatibility.  All logic is in transcribe_window()."""
    def __init__(self, session_id: str, sample_rate: int = SAMPLE_RATE) -> None:
        self.session_id = session_id
        self.sample_rate = sample_rate

    def transcribe_window(
        self, audio: np.ndarray, window_start_s: float
    ) -> List[TranscriptSegment]:
        return transcribe_window(audio, window_start_s, self.session_id)


def transcribe_window(
    audio: np.ndarray,
    window_start_s: float,
    session_id: str = "default",
) -> List[TranscriptSegment]:
    """
    Transcribe a float32 audio array.

    Parameters
    ----------
    audio          : float32 ndarray at SAMPLE_RATE, values in [-1, 1]
    window_start_s : absolute session time for sample index 0
    session_id     : used only for generating unique segment IDs

    Returns
    -------
    List of TranscriptSegment with absolute timestamps.
    """
    global _seg_counter

    if _whisper_model is None:
        return []

    if len(audio) == 0:
        return []

    # Quick energy check — skip obvious silence to avoid wasting CPU.
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < VAD_ENERGY_THRESHOLD:
        return []

    results: List[TranscriptSegment] = []
    try:
        def _run(use_vad: bool):
            return _whisper_model.transcribe(
                audio,
                beam_size=BEAM_SIZE,
                language=LANGUAGE if LANGUAGE and LANGUAGE != "auto" else None,
                # Temperature fallback: if the greedy decode looks degenerate
                # (high compression / repeated text), Whisper retries hotter.
                temperature=[0.0, 0.2, 0.4],
                vad_filter=use_vad,
                vad_parameters=dict(
                    # Lower than the 0.5 default so quieter/softer speech
                    # isn't clipped before Whisper ever sees it.
                    threshold=VAD_THRESHOLD,
                    min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                    speech_pad_ms=VAD_SPEECH_PAD_MS,
                ),
                condition_on_previous_text=False,
                word_timestamps=True,
            )

        wh_segments, _info = _run(use_vad=True)
        wh_segments = list(wh_segments)

        # Incomplete-speech guard: if VAD stripped the whole window but the
        # raw energy says speech was present, retry once without VAD.
        if not wh_segments and rms > VAD_ENERGY_THRESHOLD:
            logger.info("VAD removed all speech — retrying without VAD filter")
            wh_segments, _info = _run(use_vad=False)
            wh_segments = list(wh_segments)

        filtered_count = 0
        for seg in wh_segments:
            text = seg.text.strip()
            if not text:
                continue

            # ── False-positive / hallucination filtering ────────────── #
            avg_lp = getattr(seg, "avg_logprob", None)
            no_sp = getattr(seg, "no_speech_prob", None)
            comp = getattr(seg, "compression_ratio", None)

            if avg_lp is not None and avg_lp < MIN_AVG_LOGPROB:
                logger.info("Dropped low-confidence segment: '%s' (avg_logprob=%.2f)",
                            text[:40], avg_lp)
                filtered_count += 1
                continue
            if comp is not None and comp > MAX_COMPRESSION_RATIO:
                logger.info("Dropped repetitive segment: '%s' (compression=%.2f)",
                            text[:40], comp)
                filtered_count += 1
                continue
            if (no_sp is not None and no_sp > MAX_NO_SPEECH_PROB
                    and (avg_lp is None or avg_lp < -0.5)):
                logger.info("Dropped likely non-speech: '%s' (no_speech_prob=%.2f)",
                            text[:40], no_sp)
                filtered_count += 1
                continue

            _seg_counter += 1
            word_timings = getattr(seg, "words", None)
            words = None
            if word_timings:
                words = [
                    TranscriptWord(
                        text=word.word,
                        start=round(word.start + window_start_s, 3),
                        end=round(word.end + window_start_s, 3),
                    )
                    for word in word_timings
                    if word.word.strip()
                    and word.start is not None
                    and word.end is not None
                    and word.end >= word.start
                ]
            results.append(
                TranscriptSegment(
                    segment_id=f"seg_{session_id}_{_seg_counter:05d}",
                    text=text,
                    start=round(seg.start + window_start_s, 3),
                    end=round(seg.end + window_start_s, 3),
                    is_final=True,
                    confidence=round(avg_lp, 3) if avg_lp is not None else None,
                    words=words,
                )
            )

        if filtered_count and not results:
            logger.info(
                "All %d Whisper segment(s) filtered out — window produced no transcript",
                filtered_count,
            )
    except Exception as exc:
        logger.exception("Whisper error: %s", exc)
        raise

    return results
