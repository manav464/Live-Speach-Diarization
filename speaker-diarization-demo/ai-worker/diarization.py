"""
diarization.py — CPU speaker diarization using an ECAPA-TDNN ONNX model.

The embedding step uses ONNX Runtime only (no PyTorch / SpeechBrain). The
default model is a community ONNX export of SpeechBrain's ECAPA-TDNN
(same 192-dim voiceprint idea as before).

Key design decisions
--------------------
- We extract ONE embedding per processing window (the whole window), not per
  VAD sub-segment.  VAD sub-segments inside a 3 s window are often only
  0.5–1 s long, which produces unreliable embeddings.  The full 3 s window
  always contains enough context.
- We gate on the window-level RMS: if the whole window is too quiet we skip it.
- We use a cosine-similarity threshold to decide same-speaker vs new-speaker.
- Embeddings are updated with a slow EMA so the identity stays stable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))

# Hugging Face repo / file for the default ECAPA ONNX (Apache-2.0; see model card).
ECAPA_ONNX_REPO_ID: str = os.getenv(
    "ECAPA_ONNX_REPO_ID",
    "kenrunio/transcriber-speaker-onnx",
).strip()
ECAPA_ONNX_FILENAME: str = os.getenv(
    "ECAPA_ONNX_FILENAME",
    "ecapa-any-length.onnx",
).strip()

# Optional explicit path to an .onnx file (skips Hub download when set).
_raw_onnx_path = os.getenv("ECAPA_ONNX_PATH", "").strip()
ECAPA_ONNX_PATH: Optional[Path] = (
    Path(_raw_onnx_path).expanduser() if _raw_onnx_path else None
)

# Cache directory for hf_hub_download (repo snapshots).
ECAPA_ONNX_CACHE_DIR: Path = Path(
    os.getenv(
        "ECAPA_ONNX_CACHE_DIR",
        str(Path(__file__).resolve().parent / "pretrained_models" / "ecapa_onnx"),
    )
).expanduser()

# Optional: cap clip length in samples (0 = use full clip). Some graphs are dynamic;
# fixed-shape graphs may require padding/truncation to their expected length.
ECAPA_ONNX_MAX_SAMPLES: int = int(os.getenv("ECAPA_ONNX_MAX_SAMPLES", "0"))

# Lower thresholds make it easier to match the same speaker across changes
# in volume, delivery, and short/partial windows. Increase if distinct
# speakers are being merged.
SPEAKER_SIMILARITY_THRESHOLD: float = float(
    os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.45")
)

SPEAKER_CONTINUITY_FLOOR: float = float(
    os.getenv("SPEAKER_CONTINUITY_FLOOR", "0.32")
)

# Minimum window RMS to attempt embedding (avoids junk from silence windows).
MIN_WINDOW_RMS: float = float(os.getenv("MIN_WINDOW_RMS", "0.015"))

# Minimum speech inside a segment/window to attempt embedding.
# Short utterances still produce usable ECAPA embeddings down to ~0.3 s;
# below that we fall back to the continuity heuristic in main.py.
MIN_EMBED_SPEECH_S: float = float(os.getenv("MIN_EMBED_SPEECH_S", "0.3"))

# Minimum score gap between the best and second-best speaker before a
# weak (below-threshold) match is accepted.  Prevents borderline scores
# from being assigned to the wrong identity when two speakers are close.
SPEAKER_MATCH_MARGIN: float = float(os.getenv("SPEAKER_MATCH_MARGIN", "0.08"))

# EMA alpha for updating a speaker's stored embedding.
# 0.1 = slow drift (stable identity), 0.3 = faster adaptation.
EMBEDDING_SMOOTHING_ALPHA: float = float(os.getenv("EMBEDDING_SMOOTHING_ALPHA", "0.15"))

# Optional: force every line to one label (disables ECAPA; main.py can short-circuit).
_raw_single_mode = os.getenv("SINGLE_SPEAKER_MODE", "false").strip().lower()
SINGLE_SPEAKER_MODE: bool = _raw_single_mode in ("1", "true", "yes", "on")
SOLE_SPEAKER_ID: str = os.getenv("SOLE_SPEAKER_ID", "speaker_001").strip() or "speaker_001"
SOLE_SPEAKER_DISPLAY: str = os.getenv("SOLE_SPEAKER_DISPLAY", "Speaker 1").strip() or "Speaker 1"

# When only one speaker is registered, the legacy behaviour allowed "weak"
# matches (score between floor and threshold) if score >= 75% of the strong
# threshold — a second person's voice could still look similar enough to merge
# into speaker_001. Set to true to use that 75%-of-threshold rule; keep false
# and use LONE_SPEAKER_WEAK_MERGE_MIN for the weak band (recommended).
_raw_relax = os.getenv("WEAK_MATCH_LONE_SPEAKER_RELAXATION", "false").strip().lower()
WEAK_MATCH_LONE_SPEAKER_RELAXATION: bool = _raw_relax in ("1", "true", "yes", "on")

# Lone speaker only (weak band): merge if best_score is in [min, threshold).
# Below min, a second person can become speaker_002 (if the segment is long enough).
_lone_weak_raw = os.getenv("LONE_SPEAKER_WEAK_MERGE_MIN", "0.30").strip()
try:
    LONE_SPEAKER_WEAK_MERGE_MIN: float = (
        float(_lone_weak_raw) if _lone_weak_raw else 0.30
    )
except ValueError:
    LONE_SPEAKER_WEAK_MERGE_MIN = 0.30

# Consecutive weak-band embeddings required before a new speaker is created.
# Per-segment ECAPA embeddings are noisy — a single borderline score (e.g.
# 0.28 for the same voice) used to split one speaker into speaker_001 and
# speaker_002. With N>1 the weak evidence must repeat before minting an
# identity; until then the segment stays on the best existing speaker.
# Set to 1 to restore the old create-on-first-weak-match behaviour.
NEW_SPEAKER_CONFIRM_COUNT: int = max(
    1, int(os.getenv("NEW_SPEAKER_CONFIRM_COUNT", "2"))
)

# Log cosine scores vs every enrolled speaker at INFO (for tuning thresholds).
_raw_diar_scores = os.getenv("LOG_DIAR_SCORES", "false").strip().lower()
LOG_DIAR_SCORES: bool = _raw_diar_scores in ("1", "true", "yes", "on")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SpeakerActivity:
    speaker_id: str
    start: float
    end: float


@dataclass
class Speaker:
    speaker_id: str
    display_name: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    sample_count: int = 0


# ---------------------------------------------------------------------------
# ONNX Runtime session
# ---------------------------------------------------------------------------

_ort_session = None  # type: ignore[assignment]
_model_error: Optional[str] = None
_model_available: bool = False
_onnx_input_binding: Optional[Tuple[str, ...]] = None
_onnx_path_loaded: Optional[Path] = None


def _ensure_onnx_model_path() -> Path:
    """Return path to the ECAPA .onnx file, downloading from the Hub if needed."""
    if ECAPA_ONNX_PATH is not None:
        p = ECAPA_ONNX_PATH.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"ECAPA_ONNX_PATH does not exist: {p}")
        return p

    from huggingface_hub import hf_hub_download  # type: ignore

    ECAPA_ONNX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=ECAPA_ONNX_REPO_ID,
            filename=ECAPA_ONNX_FILENAME,
            cache_dir=str(ECAPA_ONNX_CACHE_DIR),
            local_files_only=False,
        )
    )


def _classify_onnx_io(session: object) -> dict:
    """
    Map ONNX graph inputs to our names. Supports common ECAPA exports:
    - kenrunio/transcriber-speaker-onnx: wav + lens -> emb
    - Pawls2theWall/equivl-ecapa-voxceleb-onnx: waveform -> embedding
    """
    in_meta = {x.name: x for x in session.get_inputs()}
    wav_key = None
    lens_key = None
    for name in in_meta:
        lower = name.lower()
        if "lens" in lower or "length" in lower:
            lens_key = name
        elif any(
            k in lower for k in ("wav", "waveform", "audio", "signal", "pcm")
        ):
            wav_key = name
    if wav_key is None and in_meta:
        # Single-input graphs: assume audio
        if len(in_meta) == 1:
            wav_key = next(iter(in_meta))
        else:
            raise ValueError(
                "Cannot infer audio input name from ONNX model; inputs="
                + repr(list(in_meta))
            )

    out_meta = session.get_outputs()
    if not out_meta:
        raise ValueError("ONNX model has no outputs")

    emb_out = out_meta[0].name
    for o in out_meta:
        lo = o.name.lower()
        if "emb" in lo or "embedding" in lo or "output" in lo:
            emb_out = o.name
            break

    opts: dict = {"wav": wav_key, "emb_out": emb_out}
    if lens_key is not None:
        opts["lens"] = lens_key
    return opts


def load_model() -> bool:
    global _ort_session, _model_error, _model_available, _onnx_input_binding, _onnx_path_loaded
    if SINGLE_SPEAKER_MODE:
        _model_available = False
        _ort_session = None
        _onnx_input_binding = None
        _onnx_path_loaded = None
        logger.info(
            "SINGLE_SPEAKER_MODE: ECAPA ONNX skipped; all speech labeled %s",
            SOLE_SPEAKER_ID,
        )
        return True
    try:
        import onnxruntime as ort  # type: ignore

        onnx_path = _ensure_onnx_model_path()
        logger.info("Loading ECAPA speaker embedding ONNX (%s)…", onnx_path)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ort_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        opts = _classify_onnx_io(_ort_session)
        _onnx_input_binding = (
            opts.get("wav"),
            opts.get("lens"),
            opts.get("emb_out"),
        )
        _onnx_path_loaded = onnx_path
        _model_available = True
        ins = [x.name for x in _ort_session.get_inputs()]
        outs = [x.name for x in _ort_session.get_outputs()]
        logger.info(
            "ECAPA ONNX loaded. inputs=%s outputs=%s",
            ins,
            outs,
        )
        return True
    except Exception as exc:
        _model_error = str(exc)
        _model_available = False
        logger.warning("ECAPA ONNX unavailable: %s", exc)
        return False


def is_model_ready() -> bool:
    return SINGLE_SPEAKER_MODE or _model_available


def get_model_error() -> Optional[str]:
    return _model_error


def _run_onnx_embedding(waveform_1d: np.ndarray) -> Optional[np.ndarray]:
    """Run ONNX ECAPA on float32 mono waveform, return 1D embedding or None."""
    if _ort_session is None or _onnx_input_binding is None:
        return None

    wav_key, lens_key, emb_out_name = _onnx_input_binding
    if wav_key is None:
        return None

    x = np.ascontiguousarray(waveform_1d, dtype=np.float32)
    if ECAPA_ONNX_MAX_SAMPLES > 0 and x.shape[0] > ECAPA_ONNX_MAX_SAMPLES:
        x = x[: ECAPA_ONNX_MAX_SAMPLES]

    batch = x.reshape(1, -1)
    feed: dict = {wav_key: batch}
    if lens_key is not None:
        # Fraction of each row that is real audio (1.0 = no padding), per kenrunio card.
        feed[lens_key] = np.array([1.0], dtype=np.float32)

    try:
        if emb_out_name:
            outs = _ort_session.run([emb_out_name], feed)
        else:
            outs = _ort_session.run(None, feed)
        emb = outs[0]
    except Exception as exc:
        logger.error("ONNX embedding run failed: %s", exc)
        return None

    vec = np.asarray(emb, dtype=np.float64)
    if vec.ndim == 3:
        vec = vec[0, 0]
    elif vec.ndim == 2:
        vec = vec[0]
    elif vec.ndim != 1:
        logger.error("Unexpected ONNX embedding shape: %s", vec.shape)
        return None

    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return None
    return vec / norm


# ---------------------------------------------------------------------------
# Embedding extraction — whole window
# ---------------------------------------------------------------------------


def _extract_embedding(audio: np.ndarray) -> Optional[np.ndarray]:
    """Extract a unit-normalised speaker embedding from a float32 audio array.

    Strips leading/trailing silence before passing to ECAPA so the model
    sees dense speech rather than a padded, mostly-silent window.
    """
    if not _model_available:
        return None

    # --- Strip silence: keep only frames above energy threshold ---
    frame_len = int(0.02 * SAMPLE_RATE)  # 20 ms frames
    hop_len = int(0.01 * SAMPLE_RATE)  # 10 ms hop
    speech_frames: List[np.ndarray] = []
    i = 0
    while i + frame_len <= len(audio):
        frame = audio[i : i + frame_len]
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms >= MIN_WINDOW_RMS:
            speech_frames.append(frame)
        i += hop_len

    if not speech_frames:
        logger.debug("No speech frames found — skipping embedding")
        return None

    # Require a minimum amount of actual speech (configurable —
    # lowered so short utterances can still be embedded).
    speech_audio = np.concatenate(speech_frames)
    if len(speech_audio) < int(MIN_EMBED_SPEECH_S * SAMPLE_RATE):
        logger.debug(
            "Too little speech (%.2fs) — skipping embedding",
            len(speech_audio) / SAMPLE_RATE,
        )
        return None

    speech_f32 = np.ascontiguousarray(speech_audio, dtype=np.float32)
    return _run_onnx_embedding(speech_f32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both are already unit-normalised


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class DiarizationSession:
    def __init__(self, session_id: str, sample_rate: int = SAMPLE_RATE) -> None:
        self.session_id = session_id
        self.sample_rate = sample_rate
        self._speakers: Dict[str, Speaker] = {}
        self._speaker_counter: int = 0
        self.last_speaker_id: Optional[str] = None
        # Weak-band embeddings awaiting new-speaker confirmation.
        self._pending_new: List[np.ndarray] = []

    @property
    def speakers(self) -> List[Speaker]:
        return list(self._speakers.values())

    def process_chunk(
        self, audio: np.ndarray, chunk_start_s: float
    ) -> Tuple[List[SpeakerActivity], List[Speaker]]:
        """
        Extract ONE embedding from the whole window and match/create a speaker.

        Returns (activities, new_speakers).
        activities covers [chunk_start_s, chunk_start_s + duration].
        """
        if not _model_available:
            return [], []

        # Skip windows that are mostly silence.
        window_rms = float(np.sqrt(np.mean(audio**2)))
        if window_rms < MIN_WINDOW_RMS:
            logger.debug(
                "Skipping window (RMS=%.4f < %.4f)", window_rms, MIN_WINDOW_RMS
            )
            return [], []

        embedding = _extract_embedding(audio)
        if embedding is None:
            return [], []

        speaker, is_new, _score = self._match_or_create_speaker(embedding)
        self.last_speaker_id = speaker.speaker_id
        duration = len(audio) / self.sample_rate

        activity = SpeakerActivity(
            speaker_id=speaker.speaker_id,
            start=round(chunk_start_s, 3),
            end=round(chunk_start_s + duration, 3),
        )

        new_speakers = [speaker] if is_new else []
        return [activity], new_speakers

    def process_segment(
        self,
        audio: np.ndarray,
        seg_start_s: float,
        seg_end_s: float,
        allow_new_speaker: bool = True,
    ) -> Tuple[Optional[Speaker], bool, float]:
        """
        Identify the speaker of a single transcript segment.

        `audio` must be the slice of PCM that produced the segment (with a
        small pad already applied by the caller).  Matching a per-segment
        embedding — rather than one embedding for the whole 4 s window —
        is what keeps speaker identity correct when two people speak in
        quick succession inside the same window.

        Returns (speaker, is_new, best_score).  speaker is None when the
        slice has too little speech to embed; the caller should then fall
        back to continuity heuristics.
        """
        if not _model_available:
            return None, False, -1.0

        embedding = _extract_embedding(audio)
        if embedding is None:
            return None, False, -1.0

        speaker, is_new, score = self._match_or_create_speaker(
            embedding, allow_new_speaker=allow_new_speaker
        )
        self.last_speaker_id = speaker.speaker_id
        return speaker, is_new, score

    def _should_match_existing(
        self,
        best_id: Optional[str],
        best_score: float,
        second_best_score: float,
        num_registered: int,
    ) -> bool:
        if best_id is None:
            return False
        if best_score >= SPEAKER_SIMILARITY_THRESHOLD:
            return True
        if best_score < SPEAKER_CONTINUITY_FLOOR:
            return False
        # Weak band [floor, threshold): need a decisive best vs second-best when
        # *two or more* centroids exist. With only one centroid, second_best is
        # sentinel (-1); the old check always passed and merged everyone into S1.
        if num_registered >= 2:
            if best_score - second_best_score < SPEAKER_MATCH_MARGIN:
                return False
        # One registered voice — weak band rules for lone centroid.
        if num_registered == 1:
            if WEAK_MATCH_LONE_SPEAKER_RELAXATION:
                single_speaker_floor = max(
                    SPEAKER_CONTINUITY_FLOOR,
                    SPEAKER_SIMILARITY_THRESHOLD * 0.75,
                )
                return best_score >= single_speaker_floor
            # Merge weak scores in [LONE_SPEAKER_WEAK_MERGE_MIN, threshold).
            if LONE_SPEAKER_WEAK_MERGE_MIN >= SPEAKER_SIMILARITY_THRESHOLD:
                return False
            return (
                best_score >= LONE_SPEAKER_WEAK_MERGE_MIN
                and best_score < SPEAKER_SIMILARITY_THRESHOLD
            )
        # Multiple speakers, weak band: margin already enforced that best beats second.
        # The old `second < floor` rule rejected whenever second was ≥ floor (e.g.
        # 0.19 vs floor 0.18), fragmenting one voice into speaker_003, speaker_004, …
        return True

    def _match_or_create_speaker(
        self, embedding: np.ndarray, allow_new_speaker: bool = True
    ) -> Tuple[Speaker, bool, float]:
        scores: List[Tuple[str, float]] = []

        for spk_id, spk in self._speakers.items():
            if spk.embedding is None:
                continue
            score = _cosine_similarity(embedding, spk.embedding)
            scores.append((spk_id, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        best_id: Optional[str] = scores[0][0] if scores else None
        best_score: float = scores[0][1] if scores else -1.0
        second_best_score: float = scores[1][1] if len(scores) > 1 else -1.0

        n_reg = len(self._speakers)
        matched = self._should_match_existing(
            best_id, best_score, second_best_score, n_reg
        )

        if LOG_DIAR_SCORES and scores:
            parts = ", ".join(f"{sid}={sc:.3f}" for sid, sc in scores)
            logger.info("Diar scores (%d speakers): %s", n_reg, parts)

        if scores:
            logger.info(
                "Speaker match: best_score=%.3f second=%.3f n_spk=%d threshold=%.3f floor=%.3f margin=%.3f lone_min=%.3f → %s",
                best_score,
                second_best_score,
                n_reg,
                SPEAKER_SIMILARITY_THRESHOLD,
                SPEAKER_CONTINUITY_FLOOR,
                SPEAKER_MATCH_MARGIN,
                LONE_SPEAKER_WEAK_MERGE_MIN,
                f"matched {best_id}" if matched else "NEW SPEAKER",
            )
        else:
            logger.info("Speaker match: no registered speakers → NEW SPEAKER (first)")

        if matched and best_id is not None:
            self._pending_new.clear()
            spk = self._speakers[best_id]
            # Slow EMA update — keeps identity stable while adapting to drift.
            alpha = EMBEDDING_SMOOTHING_ALPHA
            updated = (1 - alpha) * spk.embedding + alpha * embedding
            norm = np.linalg.norm(updated)
            spk.embedding = updated / max(norm, 1e-9)
            spk.sample_count += 1
            logger.info(
                "Matched %s (score=%.3f, n=%d)",
                spk.speaker_id,
                best_score,
                spk.sample_count,
            )
            return spk, False, best_score

        # Short/unreliable segments may not create new speakers — assign the
        # best existing identity instead of fragmenting the roster.
        if not matched and not allow_new_speaker and best_id is not None:
            spk = self._speakers[best_id]
            spk.sample_count += 1
            logger.info(
                "Weak short-segment match → %s (score=%.3f, n=%d)",
                spk.speaker_id,
                best_score,
                spk.sample_count,
            )
            return spk, False, best_score

        # Weak-band evidence (score >= floor but below threshold, or a lost
        # margin between two centroids): a single noisy embedding must not
        # mint a new identity. Hold the segment on the best existing speaker
        # until NEW_SPEAKER_CONFIRM_COUNT consecutive weak embeddings agree.
        # Scores below the floor mean a genuinely different voice — those
        # still create a new speaker immediately.
        pending_pool: List[np.ndarray] = []
        if (
            allow_new_speaker
            and best_id is not None
            and best_score >= SPEAKER_CONTINUITY_FLOOR
        ):
            self._pending_new.append(embedding.copy())
            if len(self._pending_new) < NEW_SPEAKER_CONFIRM_COUNT:
                spk = self._speakers[best_id]
                spk.sample_count += 1
                logger.info(
                    "Weak-band match held on %s (score=%.3f, pending=%d/%d)",
                    spk.speaker_id,
                    best_score,
                    len(self._pending_new),
                    NEW_SPEAKER_CONFIRM_COUNT,
                )
                return spk, False, best_score
            pending_pool = self._pending_new

        self._pending_new.clear()

        # New speaker. When confirmation embeddings were collected, seed the
        # centroid with their mean — more stable than the single last clip.
        new_embedding = embedding
        if pending_pool:
            stacked = np.stack(pending_pool)
            avg = stacked.mean(axis=0)
            norm = np.linalg.norm(avg)
            if norm > 1e-9:
                new_embedding = avg / norm

        self._speaker_counter += 1
        spk_id = f"speaker_{self._speaker_counter:03d}"
        new_spk = Speaker(
            speaker_id=spk_id,
            display_name=f"Speaker {self._speaker_counter}",
            embedding=new_embedding,
            sample_count=1,
        )
        self._speakers[spk_id] = new_spk
        logger.info("New speaker created: %s (best_score=%.3f)", spk_id, best_score)
        return new_spk, True, best_score
