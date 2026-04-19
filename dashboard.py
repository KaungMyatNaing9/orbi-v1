"""
Orbi Dashboard — real-time web UI served from Jetson, viewed on Mac.
Mac browser captures mic → streams PCM to Jetson → Whisper processes it.
Open http://<jetson-ip>:8080 in your browser after starting orbi.py.
"""

import asyncio
import base64
import json
import queue
import threading

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ── Frontend ───────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Orbi Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #e0e0e0; font-family: 'Courier New', monospace;
         height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  #topbar { background: #111; border-bottom: 1px solid #222; padding: 10px 20px;
            display: flex; align-items: center; gap: 14px; flex-shrink: 0; }
  #topbar h1 { font-size: 15px; font-weight: bold; letter-spacing: 3px; color: #a78bfa; }
  #status-badge { padding: 3px 12px; border-radius: 20px; font-size: 11px;
                  font-weight: bold; letter-spacing: 1px; transition: all 0.2s; }
  .status-idle      { background: #1a1a1a; color: #555; }
  .status-listening { background: #052e16; color: #4ade80; border: 1px solid #4ade80; }
  .status-thinking  { background: #1e1b4b; color: #a78bfa; border: 1px solid #a78bfa; }
  .status-speaking  { background: #1c0f00; color: #fb923c; border: 1px solid #fb923c; }

  #mic-btn { padding: 3px 12px; border-radius: 20px; font-size: 11px; font-weight: bold;
             letter-spacing: 1px; cursor: pointer; border: 1px solid #333;
             background: #1a1a1a; color: #555; transition: all 0.2s; }
  #mic-btn.active { background: #052e16; color: #4ade80; border-color: #4ade80; }

  #main { display: flex; flex: 1; overflow: hidden; }
  #log-panel { flex: 1; display: flex; flex-direction: column; border-right: 1px solid #1e1e1e; }
  #log-panel h2 { padding: 8px 14px; font-size: 10px; color: #444; letter-spacing: 2px;
                  border-bottom: 1px solid #1a1a1a; flex-shrink: 0; }
  #log { flex: 1; overflow-y: auto; padding: 10px 14px; font-size: 12.5px; line-height: 1.9; }
  #log .entry { display: flex; gap: 10px; }
  #log .ts  { color: #2e2e2e; min-width: 56px; font-size: 10px; padding-top: 3px; flex-shrink: 0; }
  .msg-user   { color: #60a5fa; }
  .msg-orbi   { color: #c4b5fd; }
  .msg-vision { color: #34d399; }
  .msg-move   { color: #fbbf24; }
  .msg-sys    { color: #3a3a3a; font-style: italic; }

  #right { width: 340px; display: flex; flex-direction: column; flex-shrink: 0; }
  #cam-panel { flex: 1; display: flex; flex-direction: column; border-bottom: 1px solid #1e1e1e; }
  #cam-panel h2 { padding: 8px 14px; font-size: 10px; color: #444; letter-spacing: 2px;
                  border-bottom: 1px solid #1a1a1a; flex-shrink: 0; }
  #cam-wrap { flex: 1; display: flex; align-items: center; justify-content: center; padding: 14px; }
  #cam-img  { max-width: 100%; max-height: 220px; border-radius: 6px;
              border: 1px solid #222; display: none; }
  #cam-ph   { color: #2a2a2a; font-size: 12px; }
  #cam-desc { padding: 6px 14px 10px; font-size: 11px; color: #555; font-style: italic;
              min-height: 32px; flex-shrink: 0; }
  #footer { padding: 6px 14px; font-size: 10px; border-top: 1px solid #1a1a1a;
            display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
  .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
  .green { background: #4ade80; } .red { background: #f87171; }
</style>
</head>
<body>

<div id="topbar">
  <h1>ORBI</h1>
  <div id="status-badge" class="status-idle">IDLE</div>
  <button id="mic-btn" onclick="toggleMic()">🎙 ENABLE MIC</button>
</div>

<div id="main">
  <div id="log-panel">
    <h2>LOG</h2>
    <div id="log"></div>
  </div>
  <div id="right">
    <div id="cam-panel">
      <h2>VISION</h2>
      <div id="cam-wrap">
        <img id="cam-img" alt="Camera feed">
        <div id="cam-ph">No image yet</div>
      </div>
      <div id="cam-desc"></div>
    </div>
    <div id="footer">
      <span class="dot red" id="dot"></span>
      <span id="conn-txt">Disconnected</span>
    </div>
  </div>
</div>

<script>
const logEl   = document.getElementById('log');
const badge   = document.getElementById('status-badge');
const camImg  = document.getElementById('cam-img');
const camPh   = document.getElementById('cam-ph');
const camDesc = document.getElementById('cam-desc');
const dot     = document.getElementById('dot');
const connTxt = document.getElementById('conn-txt');
const micBtn  = document.getElementById('mic-btn');

function ts() { return new Date().toTimeString().slice(0, 8); }

function addLog(text, cls) {
  const e = document.createElement('div');
  e.className = 'entry';
  e.innerHTML = `<span class="ts">${ts()}</span><span class="${cls}">${escHtml(text)}</span>`;
  logEl.appendChild(e);
  logEl.scrollTop = logEl.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setStatus(state) {
  badge.className = 'status-' + state;
  badge.textContent = state.toUpperCase();
}

// ── Playback audio (Orbi's voice) ────────────────────────────────────────────
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

function playMp3(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  audioCtx.resume().then(() => {
    audioCtx.decodeAudioData(buf.buffer, decoded => {
      const src = audioCtx.createBufferSource();
      src.buffer = decoded;
      src.connect(audioCtx.destination);
      src.start(0);
    }, err => console.warn('audio decode:', err));
  });
}

// ── Mic capture → stream to Jetson ───────────────────────────────────────────
let micActive = false;
let micProcessor = null;
let micSource = null;
let micCtx = null;
let micWs = null;

async function toggleMic() {
  if (micActive) {
    stopMic();
  } else {
    await startMic();
  }
}

async function startMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    // Request 16kHz mono — matches Whisper's expected format
    micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    micSource = micCtx.createMediaStreamSource(stream);
    micProcessor = micCtx.createScriptProcessor(4096, 1, 1);

    micWs = new WebSocket(`ws://${location.host}/ws/mic`);
    micWs.binaryType = 'arraybuffer';

    micWs.onopen = () => {
      addLog('🎙 Mac mic streaming to Orbi', 'msg-sys');
    };
    micWs.onclose = () => {
      addLog('🎙 mic stream closed', 'msg-sys');
      stopMic();
    };

    micProcessor.onaudioprocess = (e) => {
      if (!micWs || micWs.readyState !== WebSocket.OPEN) return;
      const f32 = e.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      micWs.send(i16.buffer);
    };

    micSource.connect(micProcessor);
    micProcessor.connect(micCtx.destination);

    micActive = true;
    micBtn.textContent = '🎙 MIC ON';
    micBtn.classList.add('active');
    audioCtx.resume();  // also unlock playback
  } catch (err) {
    addLog(`mic error: ${err.message}`, 'msg-sys');
  }
}

function stopMic() {
  if (micProcessor) { micProcessor.disconnect(); micProcessor = null; }
  if (micSource)    { micSource.disconnect();    micSource = null; }
  if (micCtx)       { micCtx.close();            micCtx = null; }
  if (micWs)        { micWs.close();             micWs = null; }
  micActive = false;
  micBtn.textContent = '🎙 ENABLE MIC';
  micBtn.classList.remove('active');
}

// ── Event WebSocket ───────────────────────────────────────────────────────────
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onopen = () => {
    dot.className = 'dot green';
    connTxt.textContent = 'Connected';
    addLog('dashboard connected', 'msg-sys');
  };

  ws.onclose = () => {
    dot.className = 'dot red';
    connTxt.textContent = 'Disconnected — retrying…';
    setTimeout(connect, 2000);
  };

  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    switch (msg.type) {
      case 'log':
        const cls = msg.text.startsWith('👤') ? 'msg-user'
                  : msg.text.startsWith('🤖') ? 'msg-orbi'
                  : msg.text.startsWith('🔍') ? 'msg-vision'
                  : msg.text.startsWith('🛞') ? 'msg-move'
                  : 'msg-sys';
        addLog(msg.text, cls);
        break;
      case 'status': setStatus(msg.state); break;
      case 'vision':
        camImg.src = 'data:image/jpeg;base64,' + msg.image;
        camImg.style.display = 'block';
        camPh.style.display  = 'none';
        camDesc.textContent  = msg.description || '';
        break;
      case 'audio': playMp3(msg.data); break;
    }
  };
}

connect();
</script>
</body>
</html>"""

# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI()
_queue: queue.Queue = queue.Queue()
_mic_queue: queue.Queue = queue.Queue()
_connections: list = []


@app.get("/")
async def index():
    return HTMLResponse(_HTML)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _connections.append(websocket)
    try:
        while True:
            await asyncio.sleep(3600)
    except (WebSocketDisconnect, Exception):
        if websocket in _connections:
            _connections.remove(websocket)


@app.websocket("/ws/mic")
async def mic_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            _mic_queue.put(data)
    except (WebSocketDisconnect, Exception):
        pass


async def _broadcaster():
    while True:
        batch = []
        try:
            while True:
                batch.append(_queue.get_nowait())
        except queue.Empty:
            pass
        if batch and _connections:
            dead = []
            for ws in list(_connections):
                try:
                    for ev in batch:
                        await ws.send_text(json.dumps(ev))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in _connections:
                    _connections.remove(ws)
        await asyncio.sleep(0.04)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_broadcaster())


# ── Public API (called from orbi.py) ──────────────────────────────────────────

class Dashboard:
    def _push(self, event: dict):
        _queue.put(event)

    def log(self, text: str):
        self._push({"type": "log", "text": text})

    def vision(self, image_bytes: bytes, description: str = ""):
        self._push({"type": "vision",
                    "image": base64.b64encode(image_bytes).decode(),
                    "description": description})

    def audio(self, mp3_bytes: bytes):
        self._push({"type": "audio",
                    "data": base64.b64encode(mp3_bytes).decode()})

    def status(self, state: str):
        self._push({"type": "status", "state": state})

    def get_audio_chunk(self, timeout: float = 0.5) -> bytes | None:
        try:
            return _mic_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def start(self, host: str = "0.0.0.0", port: int = 8080):
        def _run():
            uvicorn.run(app, host=host, port=port, log_level="error")
        threading.Thread(target=_run, daemon=True).start()
        print(f"✓ Dashboard → http://<jetson-ip>:{port}", flush=True)


dashboard = Dashboard()
