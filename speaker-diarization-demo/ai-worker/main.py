"""
main.py — Python AI Worker entry point.

Pipeline design (v2 — unified window)
--------------------------------------
Previous design ran ASR and diarization on independent timers, causing
most transcript segments to have no matching speaker activity.

New design: ASR and diarization share the exact same audio window.
When ASR decides a window is ready to process, diarization runs on
the same audio slice immediately after.  This guarantees temporal
alignment: every transcript segment has a speaker candidate.

Audio flow:
  1. Small transport chunks (~170 ms) arrive from the browser.
  2. Both ASR and diarization share a single accumulation buffer per session.
  3. When the buffer exceeds WINDOW_S seconds we slice it, run:
       a. Faster-Whisper ASR  → transcript segments with timestamps
       b. ECAPA ONNX embeddings → speaker activities with timestamps
       c. Overlap matching    → assign speaker_id to each segment
  4. Results are emitted to the frontend.
  5. Buffer advances by STEP_S (window - overlap), preserving context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import websockets
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

import diarization as diar_module
import transcription as asr_module
from diarization import DiarizationSession, SpeakerActivity
from transcription import TranscriptionSession, TranscriptSegment, TranscriptWord

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai-worker")

HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))

# Processing window length in seconds.  Both ASR and diarization process
# this exact slice.  4 s ensures enough speech density for stable ECAPA
# embeddings even when the speaker pauses briefly.
WINDOW_S: float = float(os.getenv("WINDOW_S", "4.0"))

# Step size: how far to advance the buffer after each window.
# WINDOW_S - STEP_S = overlap kept for ASR context at boundaries.
STEP_S: float = float(os.getenv("STEP_S", "3.0"))

WINDOW_SAMPLES: int = int(WINDOW_S * SAMPLE_RATE)
STEP_SAMPLES: int = int(STEP_S * SAMPLE_RATE)

# When embedding match fails (quiet/short slice), still use overlap with prior
# activities if any. If false, do not reuse "last identified" speaker — avoids
# a second person inheriting Speaker 1 when the model returns no strong match.
_raw_last_fb = os.getenv("ALLOW_LAST_SPEAKER_FALLBACK", "false").strip().lower()
ALLOW_LAST_SPEAKER_FALLBACK: bool = _raw_last_fb in ("1", "true", "yes", "on")

# Whisper sometimes returns overlapping segment times across windows or
# back-to-back speakers; clip so the UI timeline is strictly non-overlapping.
_raw_resolve_tl = os.getenv("RESOLVE_TIMELINE_OVERLAP", "true").strip().lower()
RESOLVE_TIMELINE_OVERLAP: bool = _raw_resolve_tl in ("1", "true", "yes", "on")
MIN_EMIT_DURATION_S: float = float(os.getenv("MIN_EMIT_DURATION_S", "0.05"))

# Print each final/partial transcript line to the worker terminal (debug / tuning).
_raw_log_tc = os.getenv("LOG_TRANSCRIPTS_CONSOLE", "true").strip().lower()
LOG_TRANSCRIPTS_CONSOLE: bool = _raw_log_tc in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Per-session state — single unified buffer
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.asr = TranscriptionSession(session_id, SAMPLE_RATE)
        self.diar = DiarizationSession(session_id, SAMPLE_RATE)

        self.next_sequence: int = 0

        # Unified audio buffer shared by both pipelines.
        self.buffer: np.ndarray = np.zeros(0, dtype=np.float32)

        # Absolute session clock: seconds corresponding to buffer[0].
        self.buffer_start_s: float = 0.0

        # How many seconds of audio we have received in total.
        self.total_received_s: float = 0.0
        self.audio_end_sample: int = 0

        # All speaker activities accumulated this session (for overlap lookup).
        self.activities: List[SpeakerActivity] = []

        # Segments already emitted, for cross-window duplicate suppression.
        # Each entry: (start_s, end_s, normalized_text, word_set, speaker_id).
        self.emitted: List[tuple] = []
        self.emitted_words: List[Tuple[TranscriptWord, Optional[str]]] = []
        self.last_emitted_end: Dict[Optional[str], float] = {}
        # Latest segment end on the global transcript timeline (for overlap fixes).
        self.last_timeline_emit_end: float = 0.0


_sessions: Dict[str, Session] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pcm16_to_float32(data: bytes) -> np.ndarray:
    count = len(data) // 2
    samples = np.frombuffer(data, dtype="<i2", count=count).astype(np.float32)
    samples /= 32768.0
    return samples


def _make_msg(**kwargs: Any) -> str:
    return json.dumps(kwargs)


def _find_speaker(
    activities: List[SpeakerActivity], seg_start: float, seg_end: float
) -> Optional[str]:
    """Return speaker_id with the greatest time overlap with [seg_start, seg_end]."""
    best_id: Optional[str] = None
    best_overlap: float = 0.0
    for act in activities:
        overlap = max(0.0, min(act.end, seg_end) - max(act.start, seg_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = act.speaker_id
    return best_id


# Extra audio padded around each transcript segment before embedding, so the
# ECAPA model sees a little leading/trailing context.
SEGMENT_PAD_S: float = float(os.getenv("SEGMENT_PAD_S", "0.15"))

# Segments shorter than this may match existing speakers but may NOT create
# new ones — sub-second utterances produce unreliable ECAPA embeddings and
# were the main source of spurious speaker_NNN identities.
MIN_NEW_SPEAKER_SEG_S: float = float(os.getenv("MIN_NEW_SPEAKER_SEG_S", "1.0"))


def _words(text: str) -> frozenset:
    return frozenset(
        w.strip(".,!?;:'\"()[]{}-").lower()
        for w in text.split()
        if w.strip(".,!?;:'\"()[]{}-")
    )


def _resolve_timeline_overlap(
    session: Session, seg: TranscriptSegment
) -> Optional[TranscriptSegment]:
    """If seg starts before the last emitted segment ended, clip start forward.

    Whisper can assign overlapping timestamps when the turn boundary falls
    inside a sliding window; without this, two speakers' lines share seconds.
    """
    if not RESOLVE_TIMELINE_OVERLAP:
        return seg
    boundary = session.last_timeline_emit_end
    if seg.start >= boundary - 1e-6:
        return seg
    new_start = boundary
    if new_start >= seg.end - MIN_EMIT_DURATION_S:
        logger.info(
            "Dropped segment after timeline overlap trim: '%s' [was %.3f–%.3f]",
            seg.text[:50],
            seg.start,
            seg.end,
        )
        return None

    new_words: Optional[List[TranscriptWord]] = None
    if seg.words:
        new_words = []
        for w in seg.words:
            if w.end <= new_start + 1e-6:
                continue
            ws = max(w.start, new_start)
            we = w.end
            if we > ws + 1e-6:
                new_words.append(TranscriptWord(w.text, ws, we))
        if not new_words:
            logger.info(
                "Dropped segment (all words before overlap boundary): '%s'",
                seg.text[:50],
            )
            return None

    new_text = seg.text
    if new_words:
        parts = [w.text for w in new_words]
        new_text = " ".join(p.strip() for p in parts if p.strip())
        if not new_text:
            return None

    logger.info(
        "Timeline overlap trim: '%s' [%.3f–%.3f] → [%.3f–%.3f]",
        seg.text[:40],
        seg.start,
        seg.end,
        new_start,
        seg.end,
    )
    return TranscriptSegment(
        segment_id=seg.segment_id,
        text=new_text,
        start=round(new_start, 3),
        end=seg.end,
        is_final=seg.is_final,
        confidence=seg.confidence,
        words=new_words,
    )


def _is_duplicate(session: Session, seg: TranscriptSegment) -> bool:
    """True if this segment was already emitted by a neighbouring window.

    Overlapping windows re-transcribe the same speech, and Whisper often
    returns *different* segmentations of it (extended text, shifted
    timestamps).  A segment is a duplicate when:
      - it substantially overlaps an emitted one in time, or
      - it has identical text near in time, or
      - it overlaps in time AND its words are contained in / mostly match
        an emitted segment's words (same utterance, different split).
    """
    text_key = " ".join(seg.text.lower().split())
    seg_words = _words(seg.text)
    seg_dur = max(seg.end - seg.start, 1e-6)
    for s, e, t, t_words, _speaker_id in session.emitted:
        overlap = max(0.0, min(e, seg.end) - max(s, seg.start))
        if overlap / seg_dur > 0.6:
            return True
        if t == text_key and (overlap > 0 or abs(s - seg.start) < 1.5):
            return True
        if overlap > 0 and seg_words and t_words:
            # Same utterance re-segmented: one text contains the other,
            # or they share most of their words.
            if seg_words <= t_words or t_words <= seg_words:
                return True
            union = seg_words | t_words
            if union and len(seg_words & t_words) / len(union) >= 0.6:
                return True
    return False


def _token(text: str) -> str:
    return text.strip(".,!?;:'\"()[]{}-").lower()


def _remove_repeated_words(
    session: Session,
    seg: TranscriptSegment,
    speaker_id: Optional[str],
) -> Optional[TranscriptSegment]:
    previous_end = session.last_emitted_end.get(speaker_id, 0.0)

    if seg.words:
        kept_words: List[TranscriptWord] = []
        matched_previous: set[int] = set()
        previous_words = [
            word
            for word, previous_speaker in session.emitted_words
            if previous_speaker == speaker_id
        ]
        for word in seg.words:
            token = _token(word.text)
            if not token:
                continue
            duplicate_index = next(
                (
                    index
                    for index, previous in enumerate(previous_words)
                    if index not in matched_previous
                    and _token(previous.text) == token
                    and abs(
                        (previous.start + previous.end) / 2
                        - (word.start + word.end) / 2
                    ) <= 0.35
                ),
                None,
            )
            if duplicate_index is not None:
                matched_previous.add(duplicate_index)
                continue

            if word.end <= previous_end:
                continue
            if word.start < previous_end:
                word = TranscriptWord(word.text, previous_end, word.end)
            kept_words.append(word)

        if not kept_words:
            return None

        text = "".join(word.text for word in kept_words).strip()
        if len(kept_words) > 1 and all(
            not word.text[:1].isspace() for word in kept_words
        ):
            text = " ".join(word.text.strip() for word in kept_words)
        if not text:
            return None
        return TranscriptSegment(
            segment_id=seg.segment_id,
            text=text,
            start=kept_words[0].start,
            end=kept_words[-1].end,
            is_final=seg.is_final,
            confidence=seg.confidence,
            words=kept_words,
        )

    # Keep compatibility with ASR implementations that do not return word
    # timestamps by removing an exact repeated phrase prefix when possible.
    current_tokens = seg.text.split()
    previous_tokens: List[str] = []
    for start, end, text, _word_set, previous_speaker in session.emitted:
        if previous_speaker == speaker_id and start < seg.end and end > seg.start - 2.0:
            previous_tokens.extend(text.split())
    max_overlap = min(len(current_tokens), len(previous_tokens), 12)
    overlap_count = 0
    for count in range(max_overlap, 0, -1):
        if [_token(word) for word in previous_tokens[-count:]] == [
            _token(word) for word in current_tokens[:count]
        ]:
            overlap_count = count
            break
    remaining_tokens = current_tokens[overlap_count:]
    start = max(seg.start, previous_end)
    if not remaining_tokens or start >= seg.end:
        return None
    return TranscriptSegment(
        segment_id=seg.segment_id,
        text=" ".join(remaining_tokens),
        start=start,
        end=seg.end,
        is_final=seg.is_final,
        confidence=seg.confidence,
    )


def _has_overlap_speaker(
    activities: List[SpeakerActivity],
    speaker_id: Optional[str],
    seg_start: float,
    seg_end: float,
) -> bool:
    """True if a *different* speaker's activity overlaps this segment —
    i.e. simultaneous speech was detected."""
    if speaker_id is None:
        return False
    for act in activities:
        if act.speaker_id == speaker_id:
            continue
        if max(0.0, min(act.end, seg_end) - max(act.start, seg_start)) > 0.05:
            return True
    return False


async def _assign_speaker(
    ws: websockets.WebSocketServerProtocol,
    session: Session,
    audio: np.ndarray,
    window_start_s: float,
    seg: TranscriptSegment,
) -> Tuple[Optional[str], float, str]:
    """Identify the speaker for one transcript segment.

    Extracts an embedding from just the segment's audio slice (plus a
    small pad) — not the whole 4 s window — so two speakers talking in
    quick succession inside one window get separate identities.

    Returns (speaker_id, match_score, method) where method is one of
    "embedding", "overlap", "last", or "none".  speaker_id may be None.
    """
    session_id = session.session_id
    sr = SAMPLE_RATE
    win_dur = len(audio) / sr

    rel_start = max(0.0, seg.start - window_start_s - SEGMENT_PAD_S)
    rel_end = min(win_dur, seg.end - window_start_s + SEGMENT_PAD_S)
    i0 = int(rel_start * sr)
    i1 = int(rel_end * sr)

    speaker_id: Optional[str] = None
    match_score: float = -1.0

    if diar_module.SINGLE_SPEAKER_MODE:
        return diar_module.SOLE_SPEAKER_ID, 1.0, "single_mode"

    if i1 - i0 >= int(0.3 * sr) and diar_module.is_model_ready():
        slice_audio = audio[i0:i1]
        try:
            allow_new = (seg.end - seg.start) >= MIN_NEW_SPEAKER_SEG_S
            spk, is_new, score = session.diar.process_segment(
                slice_audio, seg.start, seg.end,
                allow_new_speaker=allow_new,
            )
            match_score = score
            if spk is not None:
                speaker_id = spk.speaker_id
                session.activities.append(
                    SpeakerActivity(
                        speaker_id=speaker_id,
                        start=round(seg.start, 3),
                        end=round(seg.end, 3),
                    )
                )
                if is_new:
                    logger.info(
                        "Speaker detected: %s in session %s",
                        speaker_id, session_id,
                    )
                    await ws.send(
                        _make_msg(
                            type="speaker.created",
                            session_id=session_id,
                            speaker_id=speaker_id,
                            display_name=spk.display_name,
                        )
                    )
        except Exception as exc:
            logger.error("Diarization error: %s", exc)

    # Fallback chain when the segment was too short/quiet to embed:
    # greatest-overlap activity → last identified speaker.
    if speaker_id is not None:
        return speaker_id, match_score, "embedding"
    speaker_id = _find_speaker(session.activities, seg.start, seg.end)
    if speaker_id is not None:
        return speaker_id, match_score, "overlap"
    if ALLOW_LAST_SPEAKER_FALLBACK:
        speaker_id = session.diar.last_speaker_id
        if speaker_id is not None:
            return speaker_id, match_score, "last"
    return None, match_score, "none"


# ---------------------------------------------------------------------------
# Core processing: one window → ASR + diarization → aligned events
# ---------------------------------------------------------------------------

async def _process_window(
    ws: websockets.WebSocketServerProtocol,
    session: Session,
    audio: np.ndarray,
    window_start_s: float,
    is_flush: bool = False,
) -> None:
    """Run ASR then diarization on `audio`, emit aligned transcript events."""
    session_id = session.session_id

    # ── ASR ────────────────────────────────────────────────────────────── #
    asr_segments: List[TranscriptSegment] = []
    if asr_module.is_model_ready():
        try:
            asr_segments = session.asr.transcribe_window(audio, window_start_s)
        except Exception as exc:
            logger.error("ASR error: %s", exc)
            await ws.send(_make_msg(type="pipeline.error", message=str(exc)))

    if not asr_segments:
        return  # nothing to emit

    # ── Align and emit ─────────────────────────────────────────────────── #
    for seg in sorted(asr_segments, key=lambda item: (item.start, item.end)):
        # Windows overlap for ASR context, but each segment belongs to the
        # window containing its midpoint. This prevents duplicate output.
        midpoint = (seg.start + seg.end) / 2
        if not is_flush and not (
            window_start_s <= midpoint < window_start_s + STEP_S
        ):
            continue

        # Drop a segment if this window only re-transcribed an earlier phrase.
        if _is_duplicate(session, seg):
            continue

        seg = _resolve_timeline_overlap(session, seg)
        if seg is None:
            continue

        # Per-segment diarization: embed only this segment's audio so
        # successive speakers inside one window keep separate identities.
        activity_count = len(session.activities)
        speaker_id, match_score, assign_method = await _assign_speaker(
            ws, session, audio, window_start_s, seg
        )
        original_start, original_end = seg.start, seg.end
        seg = _remove_repeated_words(session, seg, speaker_id)
        if seg is None:
            continue
        if seg.start != original_start or seg.end != original_end:
            if len(session.activities) > activity_count:
                activity = session.activities[-1]
                if (
                    activity.speaker_id == speaker_id
                    and activity.start == original_start
                    and activity.end == original_end
                ):
                    activity.start = seg.start
                    activity.end = seg.end

        # Simultaneous-speech detection: another speaker's activity
        # overlapping this segment's time range.
        overlap = _has_overlap_speaker(
            session.activities, speaker_id, seg.start, seg.end
        )
        if overlap:
            logger.info(
                "Overlapping speech suspected: '%s' [%.1f–%.1f] → %s",
                seg.text[:40], seg.start, seg.end, speaker_id or "unknown",
            )

        session.emitted.append(
            (
                seg.start,
                seg.end,
                " ".join(seg.text.lower().split()),
                _words(seg.text),
                speaker_id,
            )
        )
        if seg.words:
            session.emitted_words.extend((word, speaker_id) for word in seg.words)
        session.last_emitted_end[speaker_id] = max(
            session.last_emitted_end.get(speaker_id, 0.0), seg.end
        )

        if LOG_TRANSCRIPTS_CONSOLE:
            logger.info(
                "Transcript: '%s' [%.3f–%.3f] → %s (score=%.3f, via=%s%s)",
                seg.text[:60], seg.start, seg.end, speaker_id or "unknown",
                match_score, assign_method, ", overlap" if overlap else "",
            )
        session.last_timeline_emit_end = max(session.last_timeline_emit_end, seg.end)
        await ws.send(
            _make_msg(
                type="transcript.final" if (seg.is_final or is_flush) else "transcript.partial",
                session_id=session_id,
                segment_id=seg.segment_id,
                text=seg.text,
                speaker_id=speaker_id,
                start=seg.start,
                end=seg.end,
                is_final=seg.is_final or is_flush,
                confidence=seg.confidence,
                speaker_score=round(match_score, 3),
                overlap=overlap,
            )
        )


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------

async def _handle_start_session(
    ws: websockets.WebSocketServerProtocol, payload: dict
) -> None:
    session_id = payload.get("session_id") or str(uuid.uuid4())
    _sessions[session_id] = Session(session_id)
    logger.info("Session started: %s", session_id)
    if diar_module.SINGLE_SPEAKER_MODE:
        await ws.send(
            _make_msg(
                type="speaker.created",
                session_id=session_id,
                speaker_id=diar_module.SOLE_SPEAKER_ID,
                display_name=diar_module.SOLE_SPEAKER_DISPLAY,
            )
        )
    await ws.send(
        _make_msg(
            type="session.started",
            session_id=session_id,
            asr_ready=asr_module.is_model_ready(),
            diarization_ready=diar_module.is_model_ready(),
            asr_error=asr_module.get_model_error(),
            diarization_error=diar_module.get_model_error(),
            asr_config=asr_module.get_model_config(),
        )
    )


async def _handle_audio_chunk(
    ws: websockets.WebSocketServerProtocol,
    meta: dict,
    audio_bytes: bytes,
) -> None:
    session_id = meta.get("session_id", "default")
    sequence: int = meta.get("sequence", 0)

    session = _sessions.get(session_id)
    if session is None:
        session = Session(session_id)
        _sessions[session_id] = session
        logger.warning("Auto-creating session for id %s", session_id)

    # Sequence gap detection.
    if sequence != session.next_sequence:
        logger.warning(
            "Sequence gap — expected %d got %d", session.next_sequence, sequence
        )
    session.next_sequence = sequence + 1

    samples = _pcm16_to_float32(audio_bytes)
    if not len(samples):
        logger.warning("Ignoring empty audio chunk for session %s", session_id)
        return

    start_sample = meta.get("audio_start_sample")
    has_sample_offset = (
        isinstance(start_sample, int)
        and not isinstance(start_sample, bool)
        and start_sample >= 0
    )
    if has_sample_offset:
        expected_sample = session.audio_end_sample
        chunk_end_sample = start_sample + len(samples)
        if start_sample > expected_sample:
            gap_samples = start_sample - expected_sample
            logger.warning(
                "Audio timeline gap — inserting %.3fs of silence for session %s",
                gap_samples / SAMPLE_RATE, session_id,
            )
            session.buffer = np.concatenate(
                [session.buffer, np.zeros(gap_samples, dtype=np.float32)]
            )
        elif start_sample < expected_sample:
            overlap_samples = expected_sample - start_sample
            if overlap_samples >= len(samples):
                logger.warning("Ignoring duplicate audio chunk for session %s", session_id)
                return
            logger.warning(
                "Trimming %.3fs of overlapping audio for session %s",
                overlap_samples / SAMPLE_RATE, session_id,
            )
            samples = samples[overlap_samples:]
        session.audio_end_sample = max(expected_sample, chunk_end_sample)
    else:
        session.audio_end_sample += len(samples)
    session.total_received_s = session.audio_end_sample / SAMPLE_RATE

    # Append to unified buffer.
    session.buffer = np.concatenate([session.buffer, samples])

    # Process all complete windows available in the buffer.
    while len(session.buffer) >= WINDOW_SAMPLES:
        window = session.buffer[:WINDOW_SAMPLES]
        window_start_s = session.buffer_start_s

        await _process_window(ws, session, window, window_start_s)

        # Advance buffer by STEP_SAMPLES, keep the overlap tail.
        session.buffer = session.buffer[STEP_SAMPLES:]
        session.buffer_start_s += STEP_S


async def _handle_end_session(
    ws: websockets.WebSocketServerProtocol, payload: dict
) -> None:
    session_id = payload.get("session_id", "default")
    session = _sessions.get(session_id)
    if session is None:
        return

    logger.info("Flushing session: %s", session_id)

    # Process whatever remains in the buffer.
    if len(session.buffer) > 0:
        await _process_window(
            ws, session, session.buffer, session.buffer_start_s, is_flush=True
        )

    await ws.send(_make_msg(type="session.ended", session_id=session_id))

    # Speaker roster summary — makes it easy to verify at a glance whether
    # the session split one voice into many IDs or merged distinct voices.
    roster = ", ".join(
        f"{spk.speaker_id}(n={spk.sample_count})"
        for spk in session.diar.speakers
    )
    logger.info(
        "Session %s ended — %d speaker(s), %d activities, %d segments emitted: %s",
        session_id,
        len(session.diar.speakers),
        len(session.activities),
        len(session.emitted),
        roster or "none",
    )
    del _sessions[session_id]


# ---------------------------------------------------------------------------
# WebSocket connection handler
# ---------------------------------------------------------------------------

async def handle_connection(ws: websockets.WebSocketServerProtocol) -> None:
    remote = ws.remote_address
    logger.info("New connection from %s", remote)

    await ws.send(
        _make_msg(
            type="worker.status",
            asr_ready=asr_module.is_model_ready(),
            asr_error=asr_module.get_model_error(),
            diarization_ready=diar_module.is_model_ready(),
            diarization_error=diar_module.get_model_error(),
            asr_config=asr_module.get_model_config(),
        )
    )

    pending_audio_meta: Optional[dict] = None

    try:
        async for message in ws:
            if isinstance(message, bytes):
                if pending_audio_meta is not None:
                    await _handle_audio_chunk(ws, pending_audio_meta, message)
                    pending_audio_meta = None
                else:
                    logger.warning("Binary frame without metadata — ignoring")
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Non-JSON text frame: %r", message[:120])
                continue

            msg_type: str = payload.get("type", "")

            if msg_type == "audio.chunk":
                pending_audio_meta = payload
            elif msg_type == "session.start":
                await _handle_start_session(ws, payload)
            elif msg_type == "session.end":
                await _handle_end_session(ws, payload)
            elif msg_type == "ping":
                await ws.send(_make_msg(type="pong"))
            else:
                logger.debug("Unhandled message type: %s", msg_type)

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Connection closed: %s", remote)
    except websockets.exceptions.ConnectionClosedError as exc:
        logger.warning("Connection error: %s — %s", remote, exc)
    except Exception as exc:
        logger.error("Handler error: %s", exc, exc_info=True)
    finally:
        logger.info("Connection finished: %s", remote)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    logger.info("=== AI Worker starting ===")
    logger.info("Unified window pipeline: window=%.1fs step=%.1fs", WINDOW_S, STEP_S)
    if diar_module.SINGLE_SPEAKER_MODE:
        logger.info(
            "SINGLE_SPEAKER_MODE is on — all speech labeled %s; ECAPA not loaded",
            diar_module.SOLE_SPEAKER_ID,
        )

    logger.info("Loading ASR model…")
    asr_ok = asr_module.load_model()
    if not asr_ok:
        logger.error("ASR model failed to load.")

    logger.info("Loading diarization model…")
    diar_ok = diar_module.load_model()
    if not diar_ok:
        logger.warning("Diarization unavailable — speaker IDs will be null.")

    logger.info(
        "Worker ready — ASR: %s | Diarization: %s",
        "OK" if asr_ok else "FAILED",
        "OK" if diar_ok else "UNAVAILABLE",
    )

    async with websockets.serve(
        handle_connection,
        HOST,
        PORT,
        ping_interval=20,
        ping_timeout=30,
        max_size=10 * 1024 * 1024,
    ):
        logger.info("Listening on ws://%s:%d/ws", HOST, PORT)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
