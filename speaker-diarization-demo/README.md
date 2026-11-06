# Real-Time Speaker Diarization Demo

A local, CPU-only demo application for real-time speech transcription with speaker identification (diarization).

> **Note:** This is a demonstration prototype built to run on CPU hardware without any GPU, CUDA, cloud APIs, or paid services. It is not the final production architecture (which would use Nemotron-3 on GPU).

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
 Faster-Whisper       ECAPA (ONNX)
 (ASR / STT)        (Speaker Embeddings)
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
| Speaker diarization | ECAPA-TDNN ONNX embeddings (ONNX Runtime; no PyTorch)   |
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

> **First run:** Faster-Whisper will download the `small.en` model (~460 MB) and
> the default ECAPA speaker ONNX (~80 MB; fetched from Hugging Face when
> diarization first loads). Both are cached locally and reused on subsequent runs.

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
SPEAKER_SIMILARITY_THRESHOLD=0.42   # raise if different people merge; lower if one splits
SPEAKER_CONTINUITY_FLOOR=0.18   # below this, no speaker matches; weak band uses margin vs 2nd-best
SPEAKER_MATCH_MARGIN=0.06
WEAK_MATCH_LONE_SPEAKER_RELAXATION=false   # true = legacy 75%-of-threshold merge when only Speaker 1 exists
LONE_SPEAKER_WEAK_MERGE_MIN=0.30         # solo weak-band merge floor; lower if one voice splits into speaker_002
SINGLE_SPEAKER_MODE=false                  # true = force one label (disables ECAPA for that run)
LOG_DIAR_SCORES=false                      # true = log similarity vs every speaker (debug)
ALLOW_LAST_SPEAKER_FALLBACK=true           # short clips inherit last speaker (less Unknown)
MIN_NEW_SPEAKER_SEG_S=1.2           # shorter utterances won't spawn a new ID
MIN_EMBED_SPEECH_S=0.3
MIN_WINDOW_RMS=0.015
SEGMENT_PAD_S=0.15
WINDOW_S=4.0
STEP_S=3.0
RESOLVE_TIMELINE_OVERLAP=true    # clip timestamps so lines do not overlap
MIN_EMIT_DURATION_S=0.05
EMBEDDING_SMOOTHING_ALPHA=0.08
```

The worker reports its active ASR model, language, device, compute type, and
beam size in the app's System Status panel. Restart the Python worker after
changing `.env`. Transcript times are relative to the start of each recording;
the UI and text export display `HH:MM:SS.mmm`, while JSON keeps numeric seconds.
Audio chunk sample offsets keep those times aligned, including gaps and the
final partial audio chunk when recording stops.

---

## AI Models

### Dependencies (no PyTorch in the worker)

- **ASR:** Faster-Whisper runs the converted OpenAI Whisper weights through **CTranslate2** (same stack as upstream `faster-whisper`).
- **Diarization:** **ONNX Runtime** loads the ECAPA speaker embedding network only.
- PyTorch / SpeechBrain are **not** used by the application code. Do not install `torch` in the worker venv unless you add optional tooling yourself.

### Speech Recognition — Faster-Whisper

- Model: `small.en` (English only, ~460 MB)
- Device: CPU
- Compute type: int8 (quantized for speed)
- Beam size: 5 by default; configurable with `BEAM_SIZE`
- For multilingual support: set `MODEL_SIZE=small` and `LANGUAGE=auto`
- Models are downloaded on first use; set `WHISPER_MODEL_CACHE_DIR` in `.env` to keep them under `ai-worker/pretrained_models/` (see `.env.example`).
- Tune CPU use with `WHISPER_CPU_THREADS` (CTranslate2 `cpu_threads`; `0` = default).

### Speaker Diarization — ECAPA-TDNN (ONNX)

- Default file: `ecapa-any-length.onnx` from [kenrunio/transcriber-speaker-onnx](https://huggingface.co/kenrunio/transcriber-speaker-onnx) on Hugging Face (~84 MB)
- Cached under `ai-worker/pretrained_models/ecapa_onnx/` (or override with `.env`; see `.env.example`)
- Uses cosine similarity on 192-dim embeddings (same idea as the original SpeechBrain ECAPA)
- Configurable similarity threshold

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
4. The system needs at least ~0.8 seconds of speech per speaker to extract a reliable embedding.

---

## CPU Performance Notes

- Transcription uses 4-second overlapping windows, advancing every 3 seconds
  (`WINDOW_S` / `STEP_S`).
- Speaker embedding extraction adds ~100–300 ms per speech segment.
- A larger ASR model or beam size can improve recognition at the cost of CPU
  speed and memory.

---

## Known Limitations

1. **Latency:** CPU inference is not real-time in the strict sense. Expect 3–6 second delays before transcription appears.
2. **Speaker diarization accuracy:** The energy-based VAD + embedding approach works for clearly separated speakers. Overlapping speech is not handled.
3. **Short utterances:** Speech segments shorter than ~0.8 seconds may not generate speaker embeddings.
4. **First-speaker cold start:** The very first utterance is transcribed without a speaker label until the embedding is extracted.
5. **Model download:** First run requires internet access and about 540 MB for the default ASR and diarization models.
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

- Usually missing `onnxruntime`, a bad `ECAPA_ONNX_PATH`, or no network for the first Hugging Face download.
- Transcription still works without diarization; speaker labels will be `null`.
- Check worker logs; install `onnxruntime` and ensure `ECAPA_ONNX_REPO_ID` / `ECAPA_ONNX_PATH` in `.env` are valid.

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
