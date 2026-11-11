/**
 * App.tsx — Real-Time Speaker Diarization Demo
 *
 * Architecture:
 *   Browser mic → Web Audio API → PCM16 chunks → WebSocket → Node.js gateway
 *   Node.js gateway → Python AI worker → transcript/speaker events → gateway → here
 *
 * Audio pipeline:
 *   1. getUserMedia at native rate
 *   2. ScriptProcessorNode (4096 sample buffer) collects Float32 frames
 *   3. Convert to mono, downsample to 16 kHz
 *   4. Pack as PCM16LE, send in two frames: JSON metadata + binary audio
 */

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TranscriptSegment {
  segment_id: string;
  text: string;
  speaker_id: string | null;
  start: number;
  end: number;
  is_final: boolean;
  confidence: number | null;
}

interface Speaker {
  speaker_id: string;
  display_name: string;
}

interface AsrConfig {
  model: string;
  language: string;
  device: string;
  compute_type: string;
  beam_size: number;
  backend?: string;
  cpu_threads?: number;
}

type BackendStatus = "disconnected" | "connecting" | "connected";
type WorkerStatus = "unknown" | "connected" | "disconnected";
type MicStatus = "idle" | "requesting" | "granted" | "denied" | "recording";

// Palette for up to 8 distinct speakers.
const SPEAKER_COLORS: string[] = [
  "#4f8ef7",
  "#e96b5a",
  "#4caf7d",
  "#f5a623",
  "#9b59b6",
  "#1abc9c",
  "#e74c3c",
  "#3498db",
];

function speakerColor(speakerId: string | null, speakers: Speaker[]): string {
  if (!speakerId) return "#888888";
  const idx = speakers.findIndex((s) => s.speaker_id === speakerId);
  return SPEAKER_COLORS[Math.max(0, idx) % SPEAKER_COLORS.length];
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
}

function formatTimelineTime(seconds: number): string {
  const totalMilliseconds = Math.max(0, Math.round(seconds * 1000));
  const hours = Math.floor(totalMilliseconds / 3_600_000);
  const minutes = Math.floor((totalMilliseconds % 3_600_000) / 60_000);
  const wholeSeconds = Math.floor((totalMilliseconds % 60_000) / 1000);
  const milliseconds = totalMilliseconds % 1000;
  return `${hours.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")}:${wholeSeconds.toString().padStart(2, "0")}.${milliseconds
    .toString()
    .padStart(3, "0")}`;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BACKEND_WS_URL: string =
  import.meta.env.VITE_BACKEND_WS ?? "ws://localhost:5000/ws";

const TARGET_SAMPLE_RATE = 16000;
// ScriptProcessor buffer size — 4096 samples ≈ 256 ms at 16 kHz.
const SCRIPT_PROCESSOR_BUFFER = 4096;
// How many native-rate samples to accumulate before resampling + sending.
// Lower = faster uplink (≈46 ms @ 44.1 kHz for 2048); no intentional playback delay.
const SEND_EVERY_SAMPLES = 2048;

const WS_RECONNECT_DELAY_MS = 3000;

// ---------------------------------------------------------------------------
// PCM helpers
// ---------------------------------------------------------------------------

/** Downsample float32 from srcRate to dstRate (simple linear interpolation). */
function resample(
  samples: Float32Array,
  srcRate: number,
  dstRate: number,
  outputLength = Math.floor(samples.length / (srcRate / dstRate))
): Float32Array {
  if (srcRate === dstRate) return samples;
  const ratio = srcRate / dstRate;
  const out = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const srcIdx = i * ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, samples.length - 1);
    const frac = srcIdx - lo;
    out[i] = samples[lo] * (1 - frac) + samples[hi] * frac;
  }
  return out;
}

/** Convert float32 [-1,1] to PCM signed 16-bit little-endian ArrayBuffer. */
function float32ToPcm16(samples: Float32Array): ArrayBuffer {
  const buf = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, clamped * 32767, /* littleEndian */ true);
  }
  return buf;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function App(): React.ReactElement {
  // --- Connection state ---
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("disconnected");
  const [workerStatus, setWorkerStatus] = useState<WorkerStatus>("unknown");
  const [workerAsrReady, setWorkerAsrReady] = useState<boolean>(false);
  const [workerDiarReady, setWorkerDiarReady] = useState<boolean>(false);
  const [workerAsrError, setWorkerAsrError] = useState<string | null>(null);
  const [workerDiarError, setWorkerDiarError] = useState<string | null>(null);
  const [workerAsrConfig, setWorkerAsrConfig] = useState<AsrConfig | null>(null);

  // --- Session ---
  const [sessionId, setSessionId] = useState<string | null>(null);

  // --- Microphone ---
  const [micStatus, setMicStatus] = useState<MicStatus>("idle");
  const [recordingSeconds, setRecordingSeconds] = useState<number>(0);

  // --- Transcript ---
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [speakers, setSpeakers] = useState<Speaker[]>([]);

  // --- Error / info messages ---
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // --- Refs ---
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sequenceRef = useRef<number>(0);
  const audioStartSampleRef = useRef<number>(0);
  const resampleRemainderRef = useRef<number>(0);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Audio pipeline refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sampleAccumRef = useRef<Float32Array>(new Float32Array(0));
  const nativeSampleRateRef = useRef<number>(44100);

  // Transcript scroll
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  // ---------------------------------------------------------------------------
  // WebSocket — connect / reconnect
  // ---------------------------------------------------------------------------

  const connectWs = useCallback(() => {
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    setBackendStatus("connecting");
    const ws = new WebSocket(BACKEND_WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setBackendStatus("connected");
      setErrorMsg(null);
    };

    ws.onclose = () => {
      setBackendStatus("disconnected");
      setWorkerStatus("disconnected");
      wsRef.current = null;
      // Reconnect automatically.
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(connectWs, WS_RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      setErrorMsg("WebSocket connection failed. Is the backend running?");
    };

    ws.onmessage = (evt: MessageEvent) => {
      if (typeof evt.data !== "string") return;
      let payload: Record<string, unknown>;
      try {
        payload = JSON.parse(evt.data);
      } catch {
        return;
      }
      handleServerMessage(payload);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    connectWs();
    return () => {
      wsRef.current?.close();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    };
  }, [connectWs]);

  // ---------------------------------------------------------------------------
  // Server message dispatcher
  // ---------------------------------------------------------------------------

  function handleServerMessage(payload: Record<string, unknown>): void {
    const type = payload.type as string;

    switch (type) {
      case "session.assigned":
        setSessionId(payload.session_id as string);
        if (payload.worker_connected) setWorkerStatus("connected");
        break;

      case "worker.status":
      case "session.started":
        setWorkerAsrReady(!!(payload.asr_ready));
        setWorkerDiarReady(!!(payload.diarization_ready));
        setWorkerAsrError((payload.asr_error as string | null) ?? null);
        setWorkerDiarError((payload.diarization_error as string | null) ?? null);
        if (payload.asr_config && typeof payload.asr_config === "object") {
          const config = payload.asr_config as Record<string, unknown>;
          if (
            typeof config.model === "string" &&
            typeof config.language === "string" &&
            typeof config.device === "string" &&
            typeof config.compute_type === "string" &&
            typeof config.beam_size === "number"
          ) {
            const backend =
              typeof config.backend === "string" ? config.backend : undefined;
            const cpu_threads =
              typeof config.cpu_threads === "number" ? config.cpu_threads : undefined;
            setWorkerAsrConfig({
              model: config.model,
              language: config.language,
              device: config.device,
              compute_type: config.compute_type,
              beam_size: config.beam_size,
              backend,
              cpu_threads,
            });
          }
        }
        break;

      case "worker.connected":
        setWorkerStatus("connected");
        break;

      case "worker.disconnected":
        setWorkerStatus("disconnected");
        break;

      case "speaker.created": {
        const newSpeaker: Speaker = {
          speaker_id: payload.speaker_id as string,
          display_name: payload.display_name as string,
        };
        setSpeakers((prev) =>
          prev.find((s) => s.speaker_id === newSpeaker.speaker_id)
            ? prev
            : [...prev, newSpeaker]
        );
        break;
      }

      case "transcript.partial":
      case "transcript.final": {
        const seg: TranscriptSegment = {
          segment_id: payload.segment_id as string,
          text: payload.text as string,
          speaker_id: (payload.speaker_id as string | null) ?? null,
          start: payload.start as number,
          end: payload.end as number,
          is_final: !!(payload.is_final),
          confidence: (payload.confidence as number | null) ?? null,
        };
        setSegments((prev) => {
          const idx = prev.findIndex((s) => s.segment_id === seg.segment_id);
          if (idx >= 0) {
            const updated = [...prev];
            updated[idx] = seg;
            return updated.sort((a, b) => a.start - b.start || a.end - b.end);
          }
          return [...prev, seg].sort((a, b) => a.start - b.start || a.end - b.end);
        });
        break;
      }

      case "pipeline.error":
        setErrorMsg(`AI Error: ${payload.message}`);
        break;

      default:
        break;
    }
  }

  // ---------------------------------------------------------------------------
  // Auto-scroll transcript
  // ---------------------------------------------------------------------------

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [segments]);

  // ---------------------------------------------------------------------------
  // Recording timer
  // ---------------------------------------------------------------------------

  function startRecordingTimer(): void {
    setRecordingSeconds(0);
    recordingTimerRef.current = setInterval(() => {
      setRecordingSeconds((s) => s + 1);
    }, 1000);
  }

  function stopRecordingTimer(): void {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Start recording
  // ---------------------------------------------------------------------------

  async function startRecording(): Promise<void> {
    setMicStatus("requesting");
    setErrorMsg(null);

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000, // request; actual rate may differ
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setMicStatus("denied");
      setErrorMsg(`Microphone access denied: ${msg}`);
      return;
    }

    setMicStatus("granted");
    streamRef.current = stream;
    audioStartSampleRef.current = 0;
    resampleRemainderRef.current = 0;
    sequenceRef.current = 0;

    // Capture at the browser's native rate, then resample the outgoing audio.
    const ctx = new AudioContext();
    audioContextRef.current = ctx;
    nativeSampleRateRef.current = ctx.sampleRate;

    const source = ctx.createMediaStreamSource(stream);
    sourceNodeRef.current = source;

    // ScriptProcessor is deprecated but still the most universally available
    // synchronous PCM access method in browsers.
    const processor = ctx.createScriptProcessor(SCRIPT_PROCESSOR_BUFFER, 1, 1);
    processorNodeRef.current = processor;

    processor.onaudioprocess = (evt: AudioProcessingEvent) => {
      const channelData = evt.inputBuffer.getChannelData(0); // Float32Array
      const chunk = new Float32Array(channelData);

      // Accumulate samples.
      const acc = new Float32Array(sampleAccumRef.current.length + chunk.length);
      acc.set(sampleAccumRef.current);
      acc.set(chunk, sampleAccumRef.current.length);
      sampleAccumRef.current = acc;

      // Send when enough samples have accumulated.
      if (sampleAccumRef.current.length >= SEND_EVERY_SAMPLES) {
        sendAudioChunk(sampleAccumRef.current);
        sampleAccumRef.current = new Float32Array(0);
      }
    };

    source.connect(processor);
    processor.connect(ctx.destination);

    setMicStatus("recording");
    sequenceRef.current = 0;
    startRecordingTimer();
  }

  // ---------------------------------------------------------------------------
  // Stop recording
  // ---------------------------------------------------------------------------

  function stopRecording(): void {
    if (sampleAccumRef.current.length > 0) {
      sendAudioChunk(sampleAccumRef.current);
    }

    // Disconnect audio nodes.
    processorNodeRef.current?.disconnect();
    sourceNodeRef.current?.disconnect();
    audioContextRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());

    processorNodeRef.current = null;
    sourceNodeRef.current = null;
    audioContextRef.current = null;
    streamRef.current = null;
    sampleAccumRef.current = new Float32Array(0);

    setMicStatus("idle");
    stopRecordingTimer();

    // Tell backend/worker the session has ended.
    if (wsRef.current?.readyState === WebSocket.OPEN && sessionId) {
      wsRef.current.send(JSON.stringify({ type: "session.end", session_id: sessionId }));
    }
  }

  // ---------------------------------------------------------------------------
  // Send audio chunk over WebSocket
  // ---------------------------------------------------------------------------

  function sendAudioChunk(native: Float32Array): void {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const exactOutputLength =
      (native.length * TARGET_SAMPLE_RATE) / nativeSampleRateRef.current +
      resampleRemainderRef.current;
    const outputLength = Math.floor(exactOutputLength);
    resampleRemainderRef.current = exactOutputLength - outputLength;
    const mono16k = resample(
      native,
      nativeSampleRateRef.current,
      TARGET_SAMPLE_RATE,
      outputLength
    );
    if (mono16k.length === 0) return;
    const pcm16 = float32ToPcm16(mono16k);

    const seq = sequenceRef.current++;
    const audioStartSample = audioStartSampleRef.current;
    audioStartSampleRef.current += mono16k.length;

    // Frame 1: JSON metadata.
    const meta = {
      type: "audio.chunk",
      version: 1,
      session_id: sessionId ?? "default",
      sequence: seq,
      audio_start_sample: audioStartSample,
      sample_rate: TARGET_SAMPLE_RATE,
      channels: 1,
      format: "pcm_s16le",
    };
    wsRef.current.send(JSON.stringify(meta));

    // Frame 2: binary PCM bytes.
    wsRef.current.send(pcm16);
  }

  // ---------------------------------------------------------------------------
  // Clear transcript
  // ---------------------------------------------------------------------------

  function clearTranscript(): void {
    setSegments([]);
    setSpeakers([]);
  }

  // ---------------------------------------------------------------------------
  // Start new session
  // ---------------------------------------------------------------------------

  function startNewSession(): void {
    clearTranscript();
    sequenceRef.current = 0;
    if (micStatus === "recording") {
      stopRecording();
    }
  }

  // ---------------------------------------------------------------------------
  // Download transcript
  // ---------------------------------------------------------------------------

  function downloadJson(): void {
    const data = {
      session_id: sessionId,
      exported_at: new Date().toISOString(),
      speakers,
      segments,
    };
    triggerDownload(
      JSON.stringify(data, null, 2),
      "transcript.json",
      "application/json"
    );
  }

  function downloadTxt(): void {
    const lines = segments.map((seg) => {
      const spkName =
        speakers.find((s) => s.speaker_id === seg.speaker_id)?.display_name ??
        (seg.speaker_id ?? "Unknown");
      return `[${formatTimelineTime(seg.start)} → ${formatTimelineTime(seg.end)}] ${spkName}: ${seg.text}`;
    });
    triggerDownload(lines.join("\n"), "transcript.txt", "text/plain");
  }

  function triggerDownload(content: string, filename: string, mimeType: string): void {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ---------------------------------------------------------------------------
  // Render helpers
  // ---------------------------------------------------------------------------

  function statusDot(active: boolean, warning = false): React.ReactElement {
    const color = active ? "#4caf7d" : warning ? "#f5a623" : "#e96b5a";
    return (
      <span
        style={{
          display: "inline-block",
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: color,
          marginRight: 6,
          flexShrink: 0,
        }}
      />
    );
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="app">
      {/* ------------------------------------------------------------------ */}
      {/* Header                                                               */}
      {/* ------------------------------------------------------------------ */}
      <header className="app-header">
        <div className="header-title">
          <h1>Real-Time Speaker Diarization</h1>
          <span className="cpu-badge">CPU Only</span>
        </div>
        <div className="header-status">
          <span className="status-item">
            {statusDot(backendStatus === "connected")}
            Backend: {backendStatus}
          </span>
          <span className="status-item">
            {statusDot(
              workerStatus === "connected",
              workerStatus === "unknown"
            )}
            AI Worker: {workerStatus}
          </span>
        </div>
      </header>

      <main className="app-main">
        {errorMsg && (
          <div className="error-banner" role="alert">
            ⚠ {errorMsg}
          </div>
        )}

        <div className="layout">
          {/* ---------------------------------------------------------------- */}
          {/* Left column                                                       */}
          {/* ---------------------------------------------------------------- */}
          <aside className="sidebar">
            {/* Microphone controls */}
            <section className="card">
              <h2>Microphone</h2>
              <div className="mic-status">
                {micStatus === "recording" ? (
                  <>
                    {statusDot(true)}
                    <span className="recording-label">Recording</span>
                    <span className="recording-timer">
                      {formatTime(recordingSeconds)}
                    </span>
                  </>
                ) : (
                  <>
                    {statusDot(false, micStatus === "requesting")}
                    <span>
                      {micStatus === "denied"
                        ? "Permission denied"
                        : micStatus === "requesting"
                        ? "Requesting…"
                        : "Idle"}
                    </span>
                  </>
                )}
              </div>
              <div className="btn-row">
                <button
                  className="btn btn-primary"
                  onClick={startRecording}
                  disabled={
                    micStatus === "recording" ||
                    micStatus === "requesting" ||
                    backendStatus !== "connected"
                  }
                >
                  ● Start Recording
                </button>
                <button
                  className="btn btn-danger"
                  onClick={stopRecording}
                  disabled={micStatus !== "recording"}
                >
                  ■ Stop Recording
                </button>
              </div>
              {backendStatus !== "connected" && (
                <p className="hint">Waiting for backend connection…</p>
              )}
            </section>

            {/* System status */}
            <section className="card">
              <h2>System Status</h2>
              <ul className="status-list">
                <li>
                  {statusDot(backendStatus === "connected")}
                  Backend gateway
                </li>
                <li>
                  {statusDot(workerStatus === "connected")}
                  AI Worker
                </li>
                <li>
                  {statusDot(workerAsrReady)}
                  ASR model (Faster-Whisper)
                  {workerAsrError && (
                    <span className="error-hint"> — {workerAsrError}</span>
                  )}
                </li>
                {workerAsrConfig && (
                  <li className="model-config">
                    {workerAsrConfig.model} · {workerAsrConfig.language} ·{" "}
                    {workerAsrConfig.device}/{workerAsrConfig.compute_type} · beam{" "}
                    {workerAsrConfig.beam_size}
                    {workerAsrConfig.backend && (
                      <> · {workerAsrConfig.backend}</>
                    )}
                    {workerAsrConfig.cpu_threads != null &&
                      workerAsrConfig.cpu_threads > 0 && (
                        <> · {workerAsrConfig.cpu_threads} threads</>
                      )}
                  </li>
                )}
                <li>
                  {statusDot(
                    workerDiarReady,
                    !workerDiarReady && !workerDiarError
                  )}
                  Diarization (Nemotron-3)
                  {!workerDiarReady && workerDiarError && (
                    <span className="error-hint"> — {workerDiarError}</span>
                  )}
                  {!workerDiarReady && !workerDiarError && (
                    <span className="error-hint"> — not ready</span>
                  )}
                </li>
                <li>
                  {statusDot(micStatus === "recording", micStatus === "granted")}
                  Microphone
                </li>
              </ul>
              {sessionId && (
                <p className="session-id">Session: {sessionId.slice(0, 8)}…</p>
              )}
            </section>

            {/* Speakers */}
            <section className="card">
              <h2>Speakers ({speakers.length})</h2>
              {speakers.length === 0 ? (
                <p className="hint">No speakers detected yet.</p>
              ) : (
                <ul className="speaker-list">
                  {speakers.map((spk, i) => (
                    <li key={spk.speaker_id} className="speaker-item">
                      <span
                        className="speaker-dot"
                        style={{
                          background:
                            SPEAKER_COLORS[i % SPEAKER_COLORS.length],
                        }}
                      />
                      {spk.display_name}
                    </li>
                  ))}
                </ul>
              )}
            </section>

            {/* Controls */}
            <section className="card">
              <h2>Controls</h2>
              <div className="btn-col">
                <button className="btn btn-secondary" onClick={clearTranscript}>
                  Clear Transcript
                </button>
                <button className="btn btn-secondary" onClick={startNewSession}>
                  New Session
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={downloadJson}
                  disabled={segments.length === 0}
                >
                  ↓ Download JSON
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={downloadTxt}
                  disabled={segments.length === 0}
                >
                  ↓ Download TXT
                </button>
              </div>
            </section>
          </aside>

          {/* ---------------------------------------------------------------- */}
          {/* Transcript panel                                                  */}
          {/* ---------------------------------------------------------------- */}
          <section className="transcript-panel card">
            <h2>Live Transcript</h2>
            {segments.length === 0 ? (
              <div className="empty-transcript">
                <p>Transcript will appear here when you start recording.</p>
                {!workerAsrReady && workerStatus === "connected" && (
                  <p className="hint">Waiting for ASR model to load…</p>
                )}
              </div>
            ) : (
              <div className="segment-list" role="log" aria-live="polite">
                {segments.map((seg) => {
                  const spkName =
                    speakers.find((s) => s.speaker_id === seg.speaker_id)
                      ?.display_name ?? (seg.speaker_id ? seg.speaker_id : null);
                  const color = speakerColor(seg.speaker_id, speakers);
                  return (
                    <div
                      key={seg.segment_id}
                      className={`segment ${seg.is_final ? "final" : "provisional"}`}
                      style={{ borderLeftColor: color }}
                    >
                      <div className="segment-meta">
                        <span
                          className="segment-speaker"
                          style={{ color }}
                        >
                          {spkName ?? (seg.is_final ? "Unknown Speaker" : "…")}
                        </span>
                        <span className="segment-time">
                          {formatTimelineTime(seg.start)} → {formatTimelineTime(seg.end)}
                        </span>
                        {!seg.is_final && (
                          <span className="provisional-badge">provisional</span>
                        )}
                      </div>
                      <p className="segment-text">{seg.text}</p>
                    </div>
                  );
                })}
                <div ref={transcriptEndRef} />
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
