"""
diarization.py — Speaker diarization with nvidia/Nemotron-3-Diarization.

Uses Hugging Face Transformers (streaming Sortformer + speaker cache) with
automatic GPU preference when CUDA is available, otherwise CPU.

NeMo SortformerEncLabelModel is attempted only when DIAR_BACKEND=nemo and a
compatible NeMo build is installed; otherwise Transformers is used.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))

NEMOTRON_MODEL_ID: str = os.getenv(
    "NEMOTRON_MODEL_ID", "nvidia/Nemotron-3-Diarization"
).strip()

DIAR_BACKEND: str = os.getenv("DIAR_BACKEND", "transformers").strip().lower()
DIAR_DEVICE: str = os.getenv("DIAR_DEVICE", "auto").strip().lower()
DIAR_PRECISION: str = os.getenv("DIAR_PRECISION", "auto").strip().lower()
DIAR_STREAMING_MODE: str = os.getenv(
    "DIAR_STREAMING_MODE", "ultra_low_latency"
).strip()

DIAR_CACHE_DIR: Path = Path(
    os.getenv(
        "DIAR_CACHE_DIR",
        str(Path(__file__).resolve().parent / "pretrained_models" / "nemotron_diar"),
    )
).expanduser()

_raw_single_mode = os.getenv("SINGLE_SPEAKER_MODE", "false").strip().lower()
SINGLE_SPEAKER_MODE: bool = _raw_single_mode in ("1", "true", "yes", "on")
SOLE_SPEAKER_ID: str = os.getenv("SOLE_SPEAKER_ID", "speaker_001").strip() or "speaker_001"
SOLE_SPEAKER_DISPLAY: str = (
    os.getenv("SOLE_SPEAKER_DISPLAY", "Speaker 1").strip() or "Speaker 1"
)

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
    nemotron_channel: Optional[int] = None


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_backend: Optional[str] = None  # "transformers" | "nemo"
_model_error: Optional[str] = None
_model_available: bool = False
_device_label: str = "cpu"
_dtype_label: str = "float32"

# Transformers backend
_t_processor = None
_t_model = None
_t_device = None
_t_dtype = None

# NeMo backend (optional)
_nemo_model = None

# ---------------------------------------------------------------------------
# Device / dtype
# ---------------------------------------------------------------------------


def _resolve_torch_device():
    import torch

    pref = DIAR_DEVICE
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("DIAR_DEVICE=%s but CUDA is unavailable — using CPU", pref)
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(device):
    import torch

    pref = DIAR_PRECISION
    if pref == "fp32":
        return torch.float32
    if pref == "fp16":
        return torch.float16
    if pref == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _apply_nemo_streaming_preset(model) -> None:
    mode = DIAR_STREAMING_MODE.strip().lower()
    presets = {
        "offline": (264, 40, 340, 40, 300),
        "low_latency": (264, 264, 9, 4, 222),
        "very_low_latency": (264, 264, 6, 2, 222),
        "ultra_low_latency": (264, 264, 3, 1, 222),
    }
    spk, fifo, chunk, right, update = presets.get(mode, presets["low_latency"])
    model.sortformer_modules.spkcache_len = spk
    model.sortformer_modules.fifo_len = fifo
    model.sortformer_modules.chunk_len = chunk
    model.sortformer_modules.chunk_right_context = right
    model.sortformer_modules.spkcache_update_period = update
    model._check_streaming_parameters()
    logger.info("NeMo streaming preset: %s", mode)


def _load_transformers() -> bool:
    global _t_processor, _t_model, _t_device, _t_dtype
    global _backend, _model_available, _model_error, _device_label, _dtype_label

    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    _t_device = _resolve_torch_device()
    _t_dtype = _resolve_dtype(_t_device)
    _device_label = str(_t_device)
    _dtype_label = str(_t_dtype).split(".")[-1]

    cache = str(DIAR_CACHE_DIR)
    DIAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading Nemotron diarization (Transformers) on %s (%s)…",
        _t_device,
        _t_dtype,
    )
    _t_processor = AutoProcessor.from_pretrained(NEMOTRON_MODEL_ID, cache_dir=cache)
    _t_processor.set_streaming_mode(DIAR_STREAMING_MODE)
    _t_model = AutoModelForAudioFrameClassification.from_pretrained(
        NEMOTRON_MODEL_ID,
        dtype=_t_dtype,
        cache_dir=cache,
    )
    _t_model = _t_model.to(_t_device)
    _t_model.eval()

    _backend = "transformers"
    _model_available = True
    _model_error = None
    logger.info(
        "Nemotron-3-Diarization ready (transformers, streaming=%s, latency≈%sms)",
        DIAR_STREAMING_MODE,
        getattr(_t_processor, "streaming_latency_ms", "?"),
    )
    return True


def _load_nemo() -> bool:
    global _nemo_model, _backend, _model_available, _model_error, _device_label

    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    device = _resolve_torch_device()
    _device_label = str(device)
    map_location = "cuda" if device.type == "cuda" else "cpu"

    logger.info("Loading Nemotron diarization (NeMo) on %s…", map_location)
    _nemo_model = SortformerEncLabelModel.from_pretrained(
        NEMOTRON_MODEL_ID,
        map_location=map_location,
    )
    _nemo_model.eval()
    if device.type == "cuda":
        _nemo_model = _nemo_model.cuda()
    else:
        _nemo_model = _nemo_model.cpu()
    _apply_nemo_streaming_preset(_nemo_model)

    _backend = "nemo"
    _model_available = True
    _model_error = None
    logger.info("Nemotron-3-Diarization ready (NeMo, streaming=%s)", DIAR_STREAMING_MODE)
    return True


def load_model() -> bool:
    global _model_error, _model_available, _dtype_label

    if SINGLE_SPEAKER_MODE:
        _model_available = False
        _model_error = None
        logger.info(
            "SINGLE_SPEAKER_MODE: Nemotron diarization skipped; label=%s",
            SOLE_SPEAKER_ID,
        )
        return True

    with _model_lock:
        if _model_available:
            return True

        backends: List[str]
        if DIAR_BACKEND == "nemo":
            backends = ["nemo", "transformers"]
        elif DIAR_BACKEND == "transformers":
            backends = ["transformers"]
        else:
            backends = ["nemo", "transformers"]

        errors: List[str] = []
        for name in backends:
            try:
                if name == "nemo" and _load_nemo():
                    return True
                if name == "transformers" and _load_transformers():
                    return True
            except Exception as exc:
                msg = f"{name}: {exc}"
                errors.append(msg)
                logger.warning("Diarization backend failed — %s", msg)

        _model_error = "; ".join(errors) if errors else "No backend available"
        _model_available = False
        return False


def is_model_ready() -> bool:
    return SINGLE_SPEAKER_MODE or _model_available


def get_model_error() -> Optional[str]:
    return _model_error


def get_model_config() -> Dict[str, Any]:
    return {
        "model": NEMOTRON_MODEL_ID,
        "backend": _backend,
        "device": _device_label,
        "dtype": _dtype_label,
        "streaming_mode": DIAR_STREAMING_MODE,
    }


# ---------------------------------------------------------------------------
# Segment parsing (NeMo tuple strings or HF dicts)
# ---------------------------------------------------------------------------


def _parse_segment_entry(
    entry,
) -> Optional[Tuple[float, float, int]]:
    """Return (start_s, end_s, speaker_channel)."""
    if isinstance(entry, dict):
        spk = entry.get("Speaker", entry.get("speaker"))
        if spk is None:
            return None
        return (
            float(entry.get("Start", entry.get("start", 0))),
            float(entry.get("End", entry.get("end", 0))),
            int(spk),
        )

    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        return float(entry[0]), float(entry[1]), int(entry[2])

    if isinstance(entry, str):
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), int(parts[2])
    return None


def _overlap_score(
    seg_start: float, seg_end: float, activities: List[SpeakerActivity]
) -> Tuple[Optional[str], float]:
    best_id: Optional[str] = None
    best_ratio = 0.0
    seg_dur = max(seg_end - seg_start, 1e-6)
    for act in activities:
        ov = max(0.0, min(act.end, seg_end) - max(act.start, seg_start))
        if ov <= 0:
            continue
        ratio = ov / seg_dur
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = act.speaker_id
    return best_id, best_ratio


# ---------------------------------------------------------------------------
# Session — streaming Transformers
# ---------------------------------------------------------------------------


class _TransformersStreamState:
    def __init__(self) -> None:
        self.speaker_cache = None
        self.is_first: bool = True
        self.mel_frame_idx: Optional[int] = None
        self.session_pcm: np.ndarray = np.zeros(0, dtype=np.float32)
        self.time_origin_s: float = 0.0
        self.logits_parts: List = []
        self.pending_flush: bool = False


class DiarizationSession:
    def __init__(self, session_id: str, sample_rate: int = SAMPLE_RATE) -> None:
        self.session_id = session_id
        self.sample_rate = sample_rate
        self._speakers: Dict[str, Speaker] = {}
        self._channel_to_id: Dict[int, str] = {}
        self._speaker_counter: int = 0
        self.last_speaker_id: Optional[str] = None
        self._activities: List[SpeakerActivity] = []
        self._stream: Optional[_TransformersStreamState] = None
        if not SINGLE_SPEAKER_MODE:
            self._stream = _TransformersStreamState()

    @property
    def speakers(self) -> List[Speaker]:
        return list(self._speakers.values())

    @property
    def stream_initialized(self) -> bool:
        if self._stream is None:
            return False
        return len(self._stream.session_pcm) > 0

    def _register_channel(self, channel: int) -> Tuple[Speaker, bool]:
        if channel in self._channel_to_id:
            spk = self._speakers[self._channel_to_id[channel]]
            spk.sample_count += 1
            return spk, False
        self._speaker_counter += 1
        spk_id = f"speaker_{self._speaker_counter:03d}"
        spk = Speaker(
            speaker_id=spk_id,
            display_name=f"Speaker {self._speaker_counter}",
            nemotron_channel=channel,
            sample_count=1,
        )
        self._speakers[spk_id] = spk
        self._channel_to_id[channel] = spk_id
        return spk, True

    def _rebuild_activities_from_logits(self, time_origin_s: float) -> List[Speaker]:
        if _t_processor is None or _t_model is None or self._stream is None:
            return []
        if not self._stream.logits_parts:
            self._activities.clear()
            return []

        import torch

        logits = torch.cat(self._stream.logits_parts, dim=1)
        raw_segments = _t_processor.extract_speaker_dict(logits)
        segments = raw_segments[0] if raw_segments else []

        new_speakers: List[Speaker] = []
        self._activities.clear()
        for entry in segments:
            parsed = _parse_segment_entry(entry)
            if parsed is None:
                continue
            start_s, end_s, channel = parsed
            spk, is_new = self._register_channel(channel)
            if is_new:
                new_speakers.append(spk)
            self._activities.append(
                SpeakerActivity(
                    speaker_id=spk.speaker_id,
                    start=round(time_origin_s + start_s, 3),
                    end=round(time_origin_s + end_s, 3),
                )
            )
        return new_speakers

    def _transformers_forward(self, inputs) -> None:
        import torch

        assert _t_model is not None and self._stream is not None
        model_inputs: dict = {}
        for key, val in inputs.items():
            if isinstance(val, torch.Tensor):
                val = val.to(device=_t_device)
                if val.is_floating_point():
                    val = val.to(dtype=_t_dtype)
                model_inputs[key] = val
            else:
                model_inputs[key] = val
        with torch.inference_mode():
            outputs = _t_model(
                **model_inputs, speaker_cache=self._stream.speaker_cache
            )
        self._stream.speaker_cache = outputs.speaker_cache
        self._stream.logits_parts.append(outputs.logits.detach().cpu())

    def _drain_transformers_stream(self) -> None:
        if (
            self._stream is None
            or _t_processor is None
            or _t_model is None
            or not _model_available
        ):
            return

        proc = _t_processor
        audio = self._stream.session_pcm

        while True:
            if self._stream.is_first:
                need = proc.num_samples_first_audio_chunk
                if len(audio) < need:
                    break
                chunk = audio[:need]
                inputs = proc(
                    chunk,
                    sampling_rate=self.sample_rate,
                    is_streaming=True,
                    is_first_audio_chunk=True,
                    return_tensors="pt",
                )
                self._stream.is_first = False
                self._stream.mel_frame_idx = proc.num_mel_frames_per_step
                self._transformers_forward(inputs)
                continue

            assert self._stream.mel_frame_idx is not None
            start_idx = proc.audio_chunk_start(self._stream.mel_frame_idx)
            end_idx = start_idx + proc.num_samples_per_audio_chunk
            if end_idx > len(audio):
                break
            chunk = audio[start_idx:end_idx]
            inputs = proc(
                chunk,
                sampling_rate=self.sample_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            self._stream.mel_frame_idx += proc.num_mel_frames_per_step
            self._transformers_forward(inputs)

    def _flush_transformers_tail(self) -> None:
        if (
            self._stream is None
            or _t_processor is None
            or _t_model is None
            or self._stream.is_first
        ):
            return

        proc = _t_processor
        audio = self._stream.session_pcm
        if self._stream.mel_frame_idx is None:
            return
        start_idx = proc.audio_chunk_start(self._stream.mel_frame_idx)
        if start_idx >= len(audio):
            return
        tail = audio[start_idx:]
        if len(tail) < int(0.08 * self.sample_rate):
            return

        inputs = proc(
            tail,
            sampling_rate=self.sample_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
            is_last_audio_chunk=True,
            return_tensors="pt",
        )
        self._transformers_forward(inputs)
        self._stream.is_first = True
        self._stream.mel_frame_idx = None

    def _nemo_diarize_window(
        self, audio: np.ndarray, window_start_s: float
    ) -> List[Speaker]:
        if _nemo_model is None:
            return []
        predicted = _nemo_model.diarize(
            audio=[np.ascontiguousarray(audio, dtype=np.float32)],
            batch_size=1,
            sample_rate=self.sample_rate,
        )
        rows = predicted[0] if predicted else []
        new_speakers: List[Speaker] = []
        window_activities: List[SpeakerActivity] = []
        for entry in rows or []:
            parsed = _parse_segment_entry(entry)
            if parsed is None:
                continue
            rel_start, rel_end, channel = parsed
            spk, is_new = self._register_channel(channel)
            if is_new:
                new_speakers.append(spk)
            act = SpeakerActivity(
                speaker_id=spk.speaker_id,
                start=round(window_start_s + rel_start, 3),
                end=round(window_start_s + rel_end, 3),
            )
            window_activities.append(act)
        self._activities = window_activities
        return new_speakers

    def stream_audio(
        self, audio: np.ndarray, chunk_start_s: float
    ) -> Tuple[List[SpeakerActivity], List[Speaker]]:
        """Append PCM at `chunk_start_s`, run streaming diarization, return updates."""
        if SINGLE_SPEAKER_MODE or not _model_available:
            return [], []

        with _model_lock:
            if _backend == "nemo":
                new_spk = self._nemo_diarize_window(audio, chunk_start_s)
                return list(self._activities), new_spk

            assert self._stream is not None
            if len(self._stream.session_pcm) == 0:
                self._stream.time_origin_s = chunk_start_s
            elif len(audio):
                expected = self._stream.time_origin_s + (
                    len(self._stream.session_pcm) / self.sample_rate
                )
                if abs(expected - chunk_start_s) > 0.25:
                    logger.warning(
                        "Diarization timeline drift %.3fs (expected %.3f got %.3f) — "
                        "appending anyway; check that diar runs on every window",
                        expected - chunk_start_s,
                        expected,
                        chunk_start_s,
                    )

            if len(audio):
                self._stream.session_pcm = np.concatenate(
                    [self._stream.session_pcm, np.ascontiguousarray(audio, dtype=np.float32)]
                )
                self._drain_transformers_stream()

            new_speakers = self._rebuild_activities_from_logits(
                self._stream.time_origin_s
            )
            return list(self._activities), new_speakers

    def flush(self) -> Tuple[List[SpeakerActivity], List[Speaker]]:
        if SINGLE_SPEAKER_MODE or not _model_available or _backend != "transformers":
            return [], []
        with _model_lock:
            assert self._stream is not None
            self._flush_transformers_tail()
            new_speakers = self._rebuild_activities_from_logits(
                self._stream.time_origin_s
            )
            return list(self._activities), new_speakers

    def process_chunk(
        self, audio: np.ndarray, chunk_start_s: float
    ) -> Tuple[List[SpeakerActivity], List[Speaker]]:
        return self.stream_audio(audio, chunk_start_s)

    def process_segment(
        self,
        audio: np.ndarray,
        seg_start_s: float,
        seg_end_s: float,
        allow_new_speaker: bool = True,
    ) -> Tuple[Optional[Speaker], bool, float]:
        del audio, allow_new_speaker  # Nemotron labels come from stream overlap
        if SINGLE_SPEAKER_MODE:
            spk = self._speakers.get(SOLE_SPEAKER_ID)
            if spk is None:
                spk = Speaker(
                    speaker_id=SOLE_SPEAKER_ID,
                    display_name=SOLE_SPEAKER_DISPLAY,
                    sample_count=1,
                )
                self._speakers[SOLE_SPEAKER_ID] = spk
            return spk, False, 1.0

        if not _model_available:
            return None, False, -1.0

        spk_id, score = _overlap_score(seg_start_s, seg_end_s, self._activities)
        if spk_id is None:
            return None, False, -1.0
        spk = self._speakers[spk_id]
        spk.sample_count += 1
        self.last_speaker_id = spk_id
        return spk, False, score
