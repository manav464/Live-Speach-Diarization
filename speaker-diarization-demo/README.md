# Real-Time Speaker Diarization Demo

A local demo for real-time speech transcription with speaker diarization using [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization).

> **Note:** ASR stays on CPU (Faster-Whisper). Diarization uses PyTorch and **prefers GPU when CUDA is available**, otherwise CPU. No cloud APIs required.

---

## Architecture

```
Browser Microphone
      |
      | PCM16 @ 16 kHz (WebSocket binary frames)
      v
Node.js Express Gateway  :5000
      |
      | WebSocket (JSON + binary)
      v
Python AI Worker  :8000
      |
      +------------------+
      |                  |
      v                  v
 Faster-Whisper    Nemotron-3-Diarization
 (ASR / STT)     (streaming Sortformer)
      |                  |
      +--------+---------+
               |
          Timestamp Alignment
               |
          WebSocket Events
               |
               v
         React UI  :5173
```

### WebSocket Message Framing (Audio)

Audio is sent as **two consecutive WebSocket frames**:

1. **JSON text frame** — metadata: `{ "type": "audio.chunk", "sequence": N, "sample_rate": 16000, ... }`
2. **Binary frame** — raw PCM16LE bytes

All other messages (transcript events, speaker events, status) are JSON text frames.

---

## Technology Stack

| Layer               | Technology                                              |
| ------------------- | ------------------------------------------------------- |
| Frontend            | React 18 + Vite + TypeScript                            |
| Backend gateway     | Node.js 20 + Express + ws                               |
| AI worker           | Python 3.10+ + websockets + asyncio                     |
| Speech recognition  | Faster-Whisper via CTranslate2 (CPU / int8; no PyTorch) |
| Speaker diarization | Nemotron-3-Diarization via Transformers (GPU or CPU)    |
| Audio capture       | Browser Web Audio API (getUserMedia)                    |

---

## Prerequisites

- **Node.js** 18 or 20+ (`node --version`)
- **npm** 9+ (`npm --version`)
- **Python** 3.10 – 3.13 (`python --version`)
- **pip** (`pip --version`)
- Internet access for the **first model download** (models are cached locally afterwards)

---

## Installation

### 1 — Python AI Worker

```powershell
# Windows PowerShell
cd speaker-diarization-demo\ai-worker
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

```bash
# Ubuntu / WSL2
cd speaker-diarization-demo/ai-worker
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -c "import sys; print(sys.executable)"
```

> **First run:** Faster-Whisper downloads `small.en` (~460 MB). Nemotron-3-Diarization
> (~400 MB weights) is fetched from Hugging Face when diarization first loads. Models
> are cached under `ai-worker/pretrained_models/`.

Copy `.env.example` to `.env` and adjust settings if needed:

```powershell
Copy-Item .env.example .env
```

### 2 — Node.js Backend

```powershell
# Windows PowerShell
cd speaker-diarization-demo\backend
npm install
```

```bash
# Ubuntu / WSL2
cd speaker-diarization-demo/backend
npm install
```

`backend/.npmrc` sets `include=dev` so `typescript` and `ts-node` install even when `NODE_ENV=production`. If `npm run dev` says `'ts-node' is not recognized`, run `npm install --include=dev` in `backend/`.

### 3 — React Frontend

```powershell
# Windows PowerShell
cd speaker-diarization-demo\frontend
npm install
```

```bash
# Ubuntu / WSL2
cd speaker-diarization-demo/frontend
npm install
```

If `npm run build` fails with `'tsc' is not recognized` or missing `vite`, dev tools were skipped: your environment likely has `NODE_ENV=production`. This repo ships `frontend/.npmrc` with `include=dev` so `npm install` still pulls **TypeScript** and **Vite**. You can also run `npm install --include=dev` once.

---

## Running the Application

Open **three terminals** and start each service in order.

### Terminal 1 — Python AI Worker

```powershell
# Windows PowerShell
cd speaker-diarization-demo\ai-worker
.\.venv\Scripts\python.exe main.py
```

```bash
# Ubuntu / WSL2
cd speaker-diarization-demo/ai-worker
./.venv/bin/python main.py
```

Wait until you see:

```
Worker ready — ASR: OK | Diarization: OK
WebSocket server listening on ws://127.0.0.1:8000/ws
```

### Terminal 2 — Node.js Gateway

```powershell
# Windows PowerShell
cd speaker-diarization-demo\backend
npm run dev
```

```bash
# Ubuntu / WSL2
cd speaker-diarization-demo/backend
npm run dev
```

Wait until you see:

```
[gateway] HTTP + WebSocket server listening on port 5000
[gateway] Connected to AI worker.
```

### Terminal 3 — React Frontend

```powershell
# Windows PowerShell
cd speaker-diarization-demo\frontend
npm run dev
```

```bash
# Ubuntu / WSL2
cd speaker-diarization-demo/frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Local Addresses

| Service             | Address                      |
| ------------------- | ---------------------------- |
| Frontend            | http://localhost:5173        |
| Backend gateway     | http://localhost:5000        |
| Backend health      | http://localhost:5000/health |
| AI Worker WebSocket | ws://127.0.0.1:8000/ws       |

---

## Configuration

Edit `ai-worker/.env` to tune the AI worker:

```env
HOST=127.0.0.1
PORT=8000
DEVICE=cpu
COMPUTE_TYPE=int8
MODEL_SIZE=small.en       # use "small" for multilingual speech
LANGUAGE=en               # use "auto" to detect the language
BEAM_SIZE=5               # higher values can improve accuracy but use more CPU
SAMPLE_RATE=16000
CHANNELS=1
VAD_ENERGY_THRESHOLD=0.0015
VAD_THRESHOLD=0.4
VAD_MIN_SILENCE_MS=200
VAD_SPEECH_PAD_MS=120
DIAR_DEVICE=auto                 # auto | cpu | cuda — GPU when available
DIAR_PRECISION=auto              # auto | fp32 | fp16 | bf16
DIAR_BACKEND=transformers          # transformers | nemo | auto
DIAR_STREAMING_MODE=low_latency    # low_latency | very_low_latency | ultra_low_latency
SINGLE_SPEAKER_MODE=false
ALLOW_LAST_SPEAKER_FALLBACK=true
WINDOW_S=4.0
STEP_S=3.0
RESOLVE_TIMELINE_OVERLAP=true
MIN_EMIT_DURATION_S=0.05
# NEMOTRON_MODEL_ID=nvidia/Nemotron-3-Diarization
# DIAR_CACHE_DIR=pretrained_models/nemotron_diar
```

The worker reports its active ASR model, language, device, compute type, and
beam size in the app's System Status panel. Restart the Python worker after
changing `.env`. Transcript times are relative to the start of each recording;
the UI and text export display `HH:MM:SS.mmm`, while JSON keeps numeric seconds.
Audio chunk sample offsets keep those times aligned, including gaps and the
final partial audio chunk when recording stops.

---

## AI Models

### Dependencies

- **ASR:** Faster-Whisper runs Whisper weights through **CTranslate2** (CPU; no PyTorch for ASR).
- **Diarization:** **PyTorch + Transformers** load `nvidia/Nemotron-3-Diarization` (~100M parameters). Optional NeMo backend if `DIAR_BACKEND=nemo` and a RoPE-capable NeMo build is installed.

### Speech Recognition — Faster-Whisper

- Model: `small.en` (English only, ~460 MB)
- Device: CPU
- Compute type: int8 (quantized for speed)
- Beam size: 5 by default; configurable with `BEAM_SIZE`
- For multilingual support: set `MODEL_SIZE=small` and `LANGUAGE=auto`
- Models are downloaded on first use; set `WHISPER_MODEL_CACHE_DIR` in `.env` to keep them under `ai-worker/pretrained_models/` (see `.env.example`).
- Tune CPU use with `WHISPER_CPU_THREADS` (CTranslate2 `cpu_threads`; `0` = default).

### Speaker Diarization — Nemotron-3-Diarization

| Item | Value |
| --- | --- |
| **Model** | [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization) |
| **Backend** | Hugging Face **Transformers** (default) or NeMo when supported |
| **Device** | `DIAR_DEVICE=auto` → CUDA if available, else CPU |
| **Streaming** | Sortformer + speaker cache (`DIAR_STREAMING_MODE`, default `low_latency` ≈ 1 s buffer) |
| **Cache** | `ai-worker/pretrained_models/nemotron_diar/` (`DIAR_CACHE_DIR`) |

#### How the app uses the model

1. **Streaming PCM** is fed into Nemotron incrementally each ASR window (only **new** audio after the first window, to avoid double-counting overlap).
2. Nemotron returns **who spoke when** (up to 8 speakers, arrival-ordered channels mapped to `speaker_001`, …).
3. **ASR** segments are labeled by **time overlap** with Nemotron activities; `ALLOW_LAST_SPEAKER_FALLBACK` covers short gaps.
4. On session end, a **flush** pass scores any trailing audio still in the streaming buffer.

Set `SINGLE_SPEAKER_MODE=true` to skip loading Nemotron and label every line as `Speaker 1`.

#### Diarization `.env` knobs

| Variable | Role |
| --- | --- |
| `DIAR_DEVICE` | `auto`, `cpu`, or `cuda` |
| `DIAR_PRECISION` | `auto`, `fp32`, `fp16`, `bf16` |
| `DIAR_BACKEND` | `transformers` (recommended on Windows), `nemo`, or `auto` |
| `DIAR_STREAMING_MODE` | Latency preset: `low_latency`, `very_low_latency`, `ultra_low_latency`, `offline` |
| `SINGLE_SPEAKER_MODE` | Skip Nemotron; single label |
| `ALLOW_LAST_SPEAKER_FALLBACK` | Inherit last speaker when no overlap yet |

---

## Testing

### Single Speaker

1. Start all three services.
2. Open http://localhost:5173.
3. Click **Start Recording** and speak for 10–15 seconds.
4. Verify transcription appears with "Speaker 1" attribution.

### Two Speakers

1. Start all three services.
2. Have two people take turns speaking (at least 1–2 seconds each turn).
3. Verify that "Speaker 1" and "Speaker 2" are identified separately.
4. **1–2 s** per turn is still recommended so Nemotron has enough speech per window.

---

## CPU Performance Notes

- Transcription uses 4-second overlapping windows, advancing every 3 seconds
  (`WINDOW_S` / `STEP_S`).
- Nemotron streaming adds CPU/GPU load per ~1 s diarization chunk (see model card RTF).
- A larger ASR model or beam size can improve recognition at the cost of CPU
  speed and memory.

---

## Known Limitations

1. **Latency:** CPU inference is not real-time in the strict sense. Expect 3–6 second delays before transcription appears.
2. **Speaker diarization accuracy:** Nemotron handles overlap better than embedding-only heuristics, but CPU inference is slower than GPU.
3. **Short utterances / latency:** Streaming buffer (~1 s minimum with `low_latency`) means very early words may lack a speaker label until the buffer fills.
4. **Model download:** First run needs internet; ~460 MB ASR + ~400 MB Nemotron weights.
6. **Browser compatibility:** Tested on Chrome and Firefox. Safari's AudioContext sample rate handling may vary.
7. **Multilingual:** Uses `small.en` by default (English only). Use `MODEL_SIZE=small` and `LANGUAGE=auto` for multilingual speech.

---

## Troubleshooting

### "WebSocket connection failed"

- Ensure the Node.js gateway is running on port 5000.
- Check for firewall rules blocking port 5000.

### "ASR model failed to load"

- Ensure you have internet access for the first download.
- Check that `faster-whisper` is installed in the worker environment:
  `.\.venv\Scripts\python.exe -m pip show faster-whisper`
- Try a smaller model: `MODEL_SIZE=tiny.en`

### "Diarization model unavailable"

- Install `torch`, `transformers>=4.57`, and `huggingface-hub>=0.34` in the worker venv.
- Ensure network access for the first download of `nvidia/Nemotron-3-Diarization`.
- Check worker logs for `Nemotron-3-Diarization ready` or `Diarization backend failed`.
- On Windows, use `DIAR_BACKEND=transformers` (NeMo 3.0.x may fail to load this checkpoint).
- ASR-only with a fixed label: `SINGLE_SPEAKER_MODE=true`.

### "No transcript appears"

- Speak clearly for at least 3 seconds before stopping.
- Check that the AI worker terminal shows no errors.
- Verify the browser microphone permission is granted.

### Microphone permission denied

- Use HTTPS or localhost (browsers restrict microphone to secure origins).
- Check browser microphone settings.

### Python package conflicts

- Use a virtual environment: `python -m venv .venv`
- Upgrade pip: `pip install --upgrade pip`

---

## Future GPU / Production Architecture

This demo is the CPU-only foundation. The planned production architecture adds:

- Nemotron-3 streaming ASR (GPU required)
- Real-time NeMo diarization
- NVIDIA CUDA / cuDNN
- True streaming with word-level timestamps

These components are intentionally excluded from this demo to ensure it runs on any machine.
