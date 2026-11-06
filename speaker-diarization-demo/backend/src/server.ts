/**
 * server.ts — Node.js Express + WebSocket gateway
 *
 * Responsibilities:
 *   • Serve HTTP health endpoint on GET /health
 *   • Accept WebSocket connections from the React frontend (/ws)
 *   • Maintain a persistent WebSocket connection to the Python AI worker
 *   • Forward binary PCM audio frames from browser → Python worker
 *   • Forward JSON transcript/speaker events from Python worker → browser
 *   • Assign session IDs and propagate them through the pipeline
 *
 * WebSocket framing protocol (audio):
 *   Frame 1: JSON text  — { "type": "audio.chunk", "session_id": "...", "sequence": N, ... }
 *   Frame 2: binary     — raw PCM16LE bytes for the chunk described in Frame 1
 *
 * All other messages are JSON text frames.
 */

import * as dotenv from "dotenv";
dotenv.config();

import express, { Request, Response } from "express";
import http from "http";
import { WebSocket, WebSocketServer } from "ws";
import { v4 as uuidv4 } from "uuid";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const PORT = parseInt(process.env.PORT ?? "5000", 10);
const AI_WORKER_URL = process.env.AI_WORKER_URL ?? "ws://127.0.0.1:8000/ws";

// How long to wait before retrying the worker connection (ms).
const WORKER_RECONNECT_DELAY_MS = 3000;

// ---------------------------------------------------------------------------
// Express app + HTTP server
// ---------------------------------------------------------------------------

const app = express();
app.use(express.json());

app.get("/health", (_req: Request, res: Response) => {
  res.json({
    status: "ok",
    service: "speaker-diarization-gateway",
    workerConnected: workerSocket?.readyState === WebSocket.OPEN,
    connectedClients: frontendClients.size,
    timestamp: new Date().toISOString(),
  });
});

const httpServer = http.createServer(app);

// ---------------------------------------------------------------------------
// Frontend WebSocket server  (/ws)
// ---------------------------------------------------------------------------

const wss = new WebSocketServer({ server: httpServer, path: "/ws" });

/** All currently connected frontend clients, keyed by session ID. */
const frontendClients = new Map<string, WebSocket>();

wss.on("connection", (clientWs: WebSocket) => {
  const sessionId = uuidv4();
  frontendClients.set(sessionId, clientWs);

  console.log(`[gateway] Frontend connected — session ${sessionId}`);

  // Tell the frontend its session ID.
  send(clientWs, {
    type: "session.assigned",
    session_id: sessionId,
    worker_connected: workerSocket?.readyState === WebSocket.OPEN,
  });

  // If a worker is already connected, tell it about this new session.
  sendToWorker({ type: "session.start", session_id: sessionId });

  // ------------------------------------------------------------------
  // State: track whether the NEXT binary frame is audio payload
  //        (preceded by an "audio.chunk" JSON metadata frame).
  // ------------------------------------------------------------------
  let pendingAudioMeta: Record<string, unknown> | null = null;

  clientWs.on("message", (data: Buffer | ArrayBuffer | Buffer[], isBinary: boolean) => {
    const rawBuffer = Buffer.isBuffer(data)
      ? data
      : Buffer.concat(Array.isArray(data) ? data : [Buffer.from(data)]);

    if (isBinary) {
      // Binary frame — PCM audio bytes.
      if (pendingAudioMeta !== null) {
        forwardAudioToWorker(pendingAudioMeta, rawBuffer);
        pendingAudioMeta = null;
      } else {
        console.warn("[gateway] Binary audio frame received without metadata — ignoring");
      }
      return;
    }

    // Text frame — parse JSON.
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(rawBuffer.toString("utf8"));
    } catch {
      console.warn("[gateway] Non-JSON text frame from client — ignoring");
      return;
    }

    const msgType = payload.type as string | undefined;

    if (msgType === "audio.chunk") {
      // Inject the gateway-assigned session ID.
      payload.session_id = sessionId;
      pendingAudioMeta = payload;
      // Also forward the metadata frame to the worker.
      sendToWorker(payload);
    } else if (msgType === "session.end") {
      payload.session_id = sessionId;
      sendToWorker(payload);
    } else if (msgType === "ping") {
      send(clientWs, { type: "pong" });
    } else {
      // Forward any other control messages.
      payload.session_id = sessionId;
      sendToWorker(payload);
    }
  });

  clientWs.on("close", () => {
    console.log(`[gateway] Frontend disconnected — session ${sessionId}`);
    frontendClients.delete(sessionId);
    sendToWorker({ type: "session.end", session_id: sessionId });
  });

  clientWs.on("error", (err: Error) => {
    console.error(`[gateway] Frontend WebSocket error (session ${sessionId}):`, err.message);
  });
});

// ---------------------------------------------------------------------------
// Python AI worker WebSocket connection
// ---------------------------------------------------------------------------

let workerSocket: WebSocket | null = null;
let workerReconnectTimer: ReturnType<typeof setTimeout> | null = null;

function connectToWorker(): void {
  if (workerSocket && workerSocket.readyState === WebSocket.OPEN) {
    return; // Already connected.
  }

  console.log(`[gateway] Connecting to AI worker at ${AI_WORKER_URL}…`);
  const ws = new WebSocket(AI_WORKER_URL);
  workerSocket = ws;

  ws.on("open", () => {
    console.log("[gateway] Connected to AI worker.");
    broadcastToFrontend({ type: "worker.connected" });
  });

  ws.on("message", (data: Buffer | string) => {
    // All worker messages are JSON text.
    const text = typeof data === "string" ? data : data.toString("utf8");
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(text);
    } catch {
      console.warn("[gateway] Non-JSON message from worker:", text.slice(0, 200));
      return;
    }

    const sessionId = payload.session_id as string | undefined;

    if (sessionId) {
      // Route to the specific session client if present.
      const clientWs = frontendClients.get(sessionId);
      if (clientWs && clientWs.readyState === WebSocket.OPEN) {
        send(clientWs, payload);
        return;
      }
    }

    // Otherwise broadcast to all connected clients (e.g. worker.status).
    broadcastToFrontend(payload);
  });

  ws.on("close", (code: number, reason: Buffer) => {
    console.warn(
      `[gateway] AI worker disconnected (code=${code} reason=${reason.toString()}) — retrying in ${WORKER_RECONNECT_DELAY_MS}ms`
    );
    workerSocket = null;
    broadcastToFrontend({ type: "worker.disconnected" });
    scheduleWorkerReconnect();
  });

  ws.on("error", (err: Error) => {
    console.error("[gateway] AI worker WebSocket error:", err.message);
    // close event will fire after error and trigger reconnect.
  });
}

function scheduleWorkerReconnect(): void {
  if (workerReconnectTimer) return;
  workerReconnectTimer = setTimeout(() => {
    workerReconnectTimer = null;
    connectToWorker();
  }, WORKER_RECONNECT_DELAY_MS);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function send(ws: WebSocket, payload: Record<string, unknown>): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  }
}

function broadcastToFrontend(payload: Record<string, unknown>): void {
  const text = JSON.stringify(payload);
  for (const [, clientWs] of frontendClients) {
    if (clientWs.readyState === WebSocket.OPEN) {
      clientWs.send(text);
    }
  }
}

function sendToWorker(payload: Record<string, unknown>): void {
  if (workerSocket?.readyState === WebSocket.OPEN) {
    workerSocket.send(JSON.stringify(payload));
  }
}

/**
 * Forward an audio chunk to the Python worker.
 * Sends the JSON metadata frame first, then the binary PCM frame.
 */
function forwardAudioToWorker(
  meta: Record<string, unknown>,
  audioBytes: Buffer
): void {
  if (!workerSocket || workerSocket.readyState !== WebSocket.OPEN) {
    console.warn("[gateway] Worker not connected — dropping audio chunk");
    return;
  }
  // Frame 1: metadata (already sent when we received it from the client).
  // Frame 2: binary audio bytes.
  workerSocket.send(audioBytes, { binary: true });
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

httpServer.listen(PORT, () => {
  console.log(`[gateway] HTTP + WebSocket server listening on port ${PORT}`);
  console.log(`[gateway] Health: http://localhost:${PORT}/health`);
  console.log(`[gateway] WebSocket: ws://localhost:${PORT}/ws`);
});

connectToWorker();
