"""
Orbi Dashboard — real-time web UI served from Jetson, viewed on Mac.
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

// Audio via Web Audio API (plays on Mac browser speaker)
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
document.addEventListener('click',  () => audioCtx.resume(), { once: true });
document.addEventListener('keydown', () => audioCtx.resume(), { once: true });

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
      case 'status':
        setStatus(msg.state);
        break;
      case 'vision':
        camImg.src = 'data:image/jpeg;base64,' + msg.image;
        camImg.style.display = 'block';
        camPh.style.display  = 'none';
        camDesc.textContent  = msg.description || '';
        break;
      case 'audio':
        playMp3(msg.data);
        break;
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

    def start(self, host: str = "0.0.0.0", port: int = 8080):
        def _run():
            uvicorn.run(app, host=host, port=port, log_level="error")
        threading.Thread(target=_run, daemon=True).start()
        print(f"✓ Dashboard → http://<jetson-ip>:{port}", flush=True)


dashboard = Dashboard()
