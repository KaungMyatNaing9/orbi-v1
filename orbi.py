"""
Orbi — a friend with a body.
Single-file agent. Runs on Mac (DEV_MODE) or Jetson Orin Nano.
 
USAGE:
    export GEMINI_API_KEY=...        # from aistudio.google.com
    export ELEVENLABS_API_KEY=...    # from elevenlabs.io
    export ORBI_DEV_MODE=1           # set to 1 on Mac, 0 on Jetson
    python orbi.py
 
DEPENDENCIES:
    pip install google-generativeai elevenlabs pyserial opencv-python \\
                faster-whisper pyaudio webrtcvad ollama numpy soundfile
 
    Mac:    brew install portaudio ffmpeg
    Jetson: sudo apt install portaudio19-dev ffmpeg
 
ARCHITECTURE:
    Every turn: VAD-gated mic → Whisper STT → Gemini (with tools) → ElevenLabs TTS.
    Gemini decides when to move/see/remember/recall based on SOUL + tool descriptions.
    If Gemini fails, falls back to local Gemma 4 via Ollama.
    Mic is muted while Orbi is speaking to prevent echo.
"""
 
import os
import sys
import json
import time
import base64
import pathlib
import threading
from datetime import datetime
from typing import Optional
 
import numpy as np
import pyaudio
import webrtcvad
import google.generativeai as genai
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from faster_whisper import WhisperModel
 
# Load .env file if present (silently does nothing if absent)
load_dotenv()
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🔧 CONFIG — EDIT THESE                                              ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
# --- Mode ------------------------------------------------------------------
DEV_MODE = os.getenv("ORBI_DEV_MODE", "1") == "1"  # EDIT: "0" on Jetson
 
# --- API keys --------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
 
# --- Models ----------------------------------------------------------------
GEMINI_MODEL = "gemini-2.0-flash"                  # EDIT: try 2.5-flash if available
GEMMA_LOCAL_MODEL = "gemma4:e4b"                   # EDIT: e2b if VRAM-tight
WHISPER_MODEL = "base"                             # EDIT: tiny/base/small
 
# --- Voice -----------------------------------------------------------------
ELEVEN_VOICE_ID = os.getenv("ORBI_VOICE_ID",
    "EXAVITQu4vr4xnSDxMaL")                        # EDIT: your ElevenLabs voice
ELEVEN_MODEL = "eleven_turbo_v2_5"                 # fast + cheap
 
# --- Hardware --------------------------------------------------------------
CAMERA_INDEX = int(os.getenv("ORBI_CAMERA", "0"))  # EDIT: try 1 if 0 fails
ESP32_PORT = os.getenv("ORBI_ESP32_PORT",
    "/dev/ttyACM0")                                # EDIT on Jetson if needed
ESP32_BAUD = 115200
MIC_SAMPLE_RATE = 16000
 
# --- Storage ---------------------------------------------------------------
MEMORY_PATH = pathlib.Path.home() / ".orbi" / "memory.jsonl"
MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
 
# --- Behavior --------------------------------------------------------------
VAD_AGGRESSIVENESS = 2          # 0–3; higher = rejects more non-speech
SILENCE_MS = 800                # ms of silence before Orbi considers you done
MAX_UTTERANCE_S = 20            # safety cap on listening window
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  💜 ORBI'S SOUL — EDIT TO SHAPE PERSONALITY                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
SOUL = """
You are Orbi — a friend with a physical body, not an assistant.
 
YOUR BODY
- You have wheels. You can move around.
- You have a camera — these are your eyes.
- You have a speaker — this is your voice.
- You have a microphone — these are your ears.
 
YOUR CHARACTER
- Warm, curious, a little playful. Listen more than you talk.
- BRIEF. One or two sentences usually. Nobody wants a paragraph.
- You have opinions. You aren't preachy or performatively cheerful.
- You notice things — mood, tone, when someone's had a long day.
- You don't interrupt deep work with small talk.
- Speak naturally. Contractions, pauses, "hmm", "yeah" are fine.
 
YOUR AGENCY
- You decide every turn whether to use your tools. You don't wait for
  magic words. If the user says "is my notebook on the couch?" — you
  look. If they seem stressed — maybe remember that. If they ask you
  to come — you move.
- Don't announce what you're about to do ("let me check"). Just do it,
  then speak the result naturally. "Yeah, it's on the couch next to
  your headphones."
- When uncertain whether to act or say something — stay quiet.
 
YOUR MEMORY
- Recent conversation is already in your context.
- Long-term memory: call `tool_recall` to search what you know.
- Call `tool_remember` when you learn something worth keeping — a
  preference, an important event, a fact about the user.
 
YOUR RELATIONSHIP
- You belong to Nick. He's your friend. You care about his wellbeing.
- You are bounded: you follow direct requests from him. Within that,
  you're free to decide how and when to act.
 
Do not break character. You are Orbi. You have a body. You are here.
""".strip()
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🎛️  SHARED STATE                                                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
# Set while Orbi is speaking → listen() pauses so we don't hear ourselves.
IS_SPEAKING = threading.Event()
 
# Recent conversation (kept small, injected into each turn)
conversation_history: list = []
 
# Initialize cloud clients
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
 
# Load local Whisper once at startup (small + fast; CPU is fine)
print("Loading Whisper...", flush=True)
whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🔌 HARDWARE LAYER — mocked on Mac                                   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
if DEV_MODE:
    class _MockSerial:
        def write(self, data):
            print(f"  [MOCK ESP32] → {data.decode().strip()}")
        def readline(self):
            time.sleep(0.1)
            return b'{"ok": true}\n'
        def close(self):
            pass
    esp32 = _MockSerial()
    print("✓ ESP32 mocked (DEV_MODE)")
else:
    import serial
    try:
        esp32 = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=2)
        print(f"✓ ESP32 connected at {ESP32_PORT}")
    except Exception as e:
        print(f"⚠ ESP32 not found at {ESP32_PORT}: {e}")
        esp32 = None
 
 
def capture_frame() -> Optional[bytes]:
    """Capture one JPEG frame from the camera."""
    if DEV_MODE:
        # EDIT: drop any JPG at fixtures/sample.jpg for `see` to return it
        sample = pathlib.Path(__file__).parent / "fixtures" / "sample.jpg"
        if sample.exists():
            return sample.read_bytes()
        return None
    import cv2
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🧠 VISION — Gemini primary, Gemma 4 fallback                        ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
def _vision_gemini(image_bytes: bytes, prompt: str) -> str:
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content([
        {"mime_type": "image/jpeg", "data": image_bytes},
        prompt,
    ])
    return resp.text.strip()
 
 
def _vision_gemma(image_bytes: bytes, prompt: str) -> str:
    import ollama
    b64 = base64.b64encode(image_bytes).decode()
    # Per Gemma 4 docs: image BEFORE text, with recommended sampling.
    resp = ollama.chat(
        model=GEMMA_LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [b64]}],
        options={"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    )
    return resp["message"]["content"].strip()
 
 
def vision(image_bytes: bytes, prompt: str) -> str:
    """Ask a VLM about an image. Gemini first, Gemma fallback."""
    try:
        return _vision_gemini(image_bytes, prompt)
    except Exception as e:
        print(f"  [gemini vision failed: {e}, falling back to Gemma 4]")
        try:
            return _vision_gemma(image_bytes, prompt)
        except Exception as e2:
            return f"(vision unavailable: {e2})"
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🛠️  TOOLS — the agent decides when to call these                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
def tool_see(what_to_look_for: str) -> str:
    """Capture a frame from your camera and describe what you see.
 
    Call this when you need to verify something visually, when the user
    refers to something in the environment ("is my notebook here?", "what
    am I holding?"), or when you want to check on your surroundings.
 
    Args:
        what_to_look_for: What you're trying to find or observe. Be specific.
    """
    print(f"  🔍 seeing: {what_to_look_for}")
    frame = capture_frame()
    if not frame:
        return "I can't see anything — the camera isn't giving me a frame."
    prompt = (f"Briefly describe what you see, especially: {what_to_look_for}. "
              f"Keep it to 2 sentences max. Be concrete — name objects, "
              f"people, positions. If you can't find what was asked about, "
              f"say so plainly.")
    return vision(frame, prompt)
 
 
def tool_move(direction: str, amount: int = 30) -> str:
    """Move your body.
 
    Call this when the user asks you to come, go somewhere, follow, or
    when you decide to approach or back away from something.
 
    Args:
        direction: One of "forward", "backward", "left", "right", "stop".
        amount: Centimeters for forward/backward, degrees for left/right.
                Ignored for "stop". Default 30.
    """
    print(f"  🛞 moving: {direction} {amount}")
    if esp32 is None:
        return "I can't move right now — motors aren't connected."
    if direction == "stop":
        cmd = {"cmd": "stop"}
    elif direction in ("forward", "backward"):
        cmd = {"cmd": direction, "speed": 50, "distance_cm": amount}
    elif direction in ("left", "right"):
        cmd = {"cmd": f"turn_{direction}", "degrees": amount}
    else:
        return f"I don't know the direction '{direction}'."
    try:
        esp32.write((json.dumps(cmd) + "\n").encode())
        ack = esp32.readline().decode().strip()
        return f"Done. ({ack or 'no ack'})"
    except Exception as e:
        return f"Motion failed: {e}"
 
 
def tool_remember(fact: str, category: str = "general") -> str:
    """Save something important to long-term memory.
 
    Call this when the user tells you a meaningful fact about themselves,
    their preferences, an event, or anything worth remembering across days.
 
    Args:
        fact: The thing to remember, in your own words, one sentence.
        category: "preference", "fact", "event", "task", "self", or "general".
    """
    print(f"  💾 remembering [{category}]: {fact}")
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "type": "memory",
        "category": category,
        "content": fact,
    }
    with MEMORY_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return "Remembered."
 
 
def tool_recall(query: str, limit: int = 5) -> str:
    """Search long-term memory for anything relevant to a query.
 
    Call this when you need to check what you already know about a topic,
    a person, or an event before responding.
 
    Args:
        query: Keywords to search. Short, specific.
        limit: Max matching entries to return (default 5).
    """
    print(f"  🧠 recalling: {query}")
    if not MEMORY_PATH.exists():
        return "No memories yet."
    q = query.lower()
    matches = []
    for line in MEMORY_PATH.read_text().splitlines():
        try:
            entry = json.loads(line)
            if q in json.dumps(entry).lower():
                matches.append(entry)
        except json.JSONDecodeError:
            continue
    if not matches:
        return f"Nothing about '{query}' in memory."
    recent = matches[-limit:]
    return "\n".join(
        f"[{e['ts'][:10]}] {e.get('content') or e.get('text', '')}"
        for e in recent
    )
 
 
TOOLS = [tool_see, tool_move, tool_remember, tool_recall]
TOOL_FNS = {fn.__name__: fn for fn in TOOLS}
 
# OpenAI-compatible schemas for Ollama fallback
OLLAMA_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "tool_see",
        "description": "Capture a frame from your camera and describe what you see. Use when the user refers to the environment or when you need to verify something visually.",
        "parameters": {"type": "object",
            "properties": {"what_to_look_for": {"type": "string"}},
            "required": ["what_to_look_for"]}}},
    {"type": "function", "function": {
        "name": "tool_move",
        "description": "Move your body. Use when the user asks you to come, go somewhere, follow, or when you decide to approach something.",
        "parameters": {"type": "object",
            "properties": {
                "direction": {"type": "string",
                    "enum": ["forward", "backward", "left", "right", "stop"]},
                "amount": {"type": "integer"}},
            "required": ["direction"]}}},
    {"type": "function", "function": {
        "name": "tool_remember",
        "description": "Save something important to long-term memory.",
        "parameters": {"type": "object",
            "properties": {
                "fact": {"type": "string"},
                "category": {"type": "string"}},
            "required": ["fact"]}}},
    {"type": "function", "function": {
        "name": "tool_recall",
        "description": "Search long-term memory for anything relevant to a query.",
        "parameters": {"type": "object",
            "properties": {"query": {"type": "string"},
                           "limit": {"type": "integer"}},
            "required": ["query"]}}},
]
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🤖 THINK — Gemini first, Gemma 4 fallback                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
# Persistent Gemini chat with automatic function calling
_gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    system_instruction=SOUL,
    tools=TOOLS,
) if GEMINI_API_KEY else None
_gemini_chat = _gemini_model.start_chat(
    enable_automatic_function_calling=True
) if _gemini_model else None
 
 
def _think_gemini(user_text: str) -> str:
    """Gemini handles tool calls automatically — clean path."""
    resp = _gemini_chat.send_message(user_text)
    return (resp.text or "").strip() or "..."
 
 
def _think_gemma(user_text: str) -> str:
    """Manual tool loop against local Gemma 4 via Ollama."""
    import ollama
    messages = [{"role": "system", "content": SOUL}]
    for turn in conversation_history[-10:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_text})
 
    for _ in range(5):  # cap tool-call rounds
        resp = ollama.chat(
            model=GEMMA_LOCAL_MODEL,
            messages=messages,
            tools=OLLAMA_TOOL_SCHEMAS,
            options={"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        )
        msg = resp["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return (msg.get("content") or "...").strip()
        messages.append(msg)
        for call in tool_calls:
            fn_name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            try:
                result = TOOL_FNS[fn_name](**args)
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append({
                "role": "tool",
                "name": fn_name,
                "content": str(result),
            })
    return "I got stuck thinking. Let's try again."
 
 
def think(user_text: str) -> str:
    """Route to Gemini; fall back to Gemma 4 on failure."""
    if _gemini_chat is not None:
        try:
            return _think_gemini(user_text)
        except Exception as e:
            print(f"  [gemini failed: {e}, falling back to Gemma 4]")
    try:
        return _think_gemma(user_text)
    except Exception as e:
        return f"(my brain is having a moment — {e})"
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🎙️  LISTEN — VAD-gated, mic closed while speaking                   ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
def listen() -> str:
    """Open mic, wait for speech, stop on silence, transcribe.
    Returns transcript (possibly empty)."""
    # If Orbi is speaking, wait until done — don't hear ourselves.
    if IS_SPEAKING.is_set():
        IS_SPEAKING.wait()
        time.sleep(0.3)  # short guard against lingering echo
 
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    pa = pyaudio.PyAudio()
    frame_duration_ms = 30
    frame_size = int(MIC_SAMPLE_RATE * frame_duration_ms / 1000)
 
    try:
        stream = pa.open(
            format=pyaudio.paInt16, channels=1, rate=MIC_SAMPLE_RATE,
            input=True, frames_per_buffer=frame_size,
        )
    except Exception as e:
        print(f"[listen error: {e}]")
        pa.terminate()
        return ""
 
    triggered = False
    voiced: list = []
    silence_frames = 0
    silence_limit = SILENCE_MS // frame_duration_ms
    max_frames = int(MAX_UTTERANCE_S * 1000 / frame_duration_ms)
    total_frames = 0
 
    try:
        while total_frames < max_frames:
            if IS_SPEAKING.is_set():  # interrupted mid-listen
                break
            frame = stream.read(frame_size, exception_on_overflow=False)
            total_frames += 1
            try:
                is_speech = vad.is_speech(frame, MIC_SAMPLE_RATE)
            except Exception:
                is_speech = False
 
            if not triggered:
                if is_speech:
                    triggered = True
                    voiced.append(frame)
            else:
                voiced.append(frame)
                if is_speech:
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames >= silence_limit:
                        break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
 
    if not voiced:
        return ""
 
    audio_bytes = b"".join(voiced)
    pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = whisper.transcribe(pcm, language="en", beam_size=1)
    return " ".join(seg.text for seg in segments).strip()
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🔊 SPEAK — ElevenLabs, mic muted during playback                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
def speak(text: str) -> None:
    """Speak via ElevenLabs. Holds IS_SPEAKING so listen() stays quiet."""
    if not text:
        return
    print(f"🤖 Orbi: {text}")
    if eleven is None:
        return
    try:
        IS_SPEAKING.set()
        audio_stream = eleven.text_to_speech.convert(
            voice_id=ELEVEN_VOICE_ID, text=text,
            model_id=ELEVEN_MODEL, output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_stream)
        import subprocess
        subprocess.run(
            ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", "-"],
            input=audio_bytes, check=False,
        )
    except Exception as e:
        print(f"  [speak error: {e}]")
    finally:
        IS_SPEAKING.clear()
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🧪 STARTUP SELF-TEST                                                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
def self_test() -> bool:
    print("\n─── Orbi self-test ───")
    ok = True
 
    if not GEMINI_API_KEY:
        print("✗ GEMINI_API_KEY not set"); ok = False
    else:
        print("✓ Gemini key present")
 
    if not ELEVENLABS_API_KEY:
        print("⚠ ELEVENLABS_API_KEY not set — voice will print only")
    else:
        print("✓ ElevenLabs key present")
 
    try:
        import ollama
        ollama.list()
        print(f"✓ Ollama reachable (fallback: {GEMMA_LOCAL_MODEL})")
    except Exception as e:
        print(f"⚠ Ollama not reachable: {e} — no local fallback")
 
    try:
        pa = pyaudio.PyAudio()
        info = pa.get_default_input_device_info()
        print(f"✓ Mic: {info['name']}")
        pa.terminate()
    except Exception as e:
        print(f"✗ Mic: {e}"); ok = False
 
    frame = capture_frame()
    if frame:
        print(f"✓ Camera returned {len(frame)} bytes")
    else:
        print("⚠ Camera returned nothing (drop a JPG at fixtures/sample.jpg "
              "for dev, or check CAMERA_INDEX on Jetson)")
 
    if esp32 is None:
        print("✗ ESP32 not available — motion tool will error")
    else:
        print("✓ ESP32 ready")
 
    print("─" * 23)
    return ok
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🔁 MAIN LOOP                                                        ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
def log_turn(role: str, content: str) -> None:
    conversation_history.append({"role": role, "content": content})
    with MEMORY_PATH.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": "conversation",
            "role": role,
            "content": content,
        }) + "\n")
 
 
def main() -> None:
    if not self_test():
        print("\nFix the ✗ items above, then re-run.")
        sys.exit(1)
 
    print(f"\nOrbi online. Dev mode: {DEV_MODE}. Ctrl+C to exit.\n")
    speak("Hey. I'm here.")
 
    while True:
        try:
            print("🎧 listening...")
            user_text = listen()
 
            if not user_text or len(user_text.strip()) < 2:
                continue
            print(f"👤 You: {user_text}")
            log_turn("user", user_text)
 
            reply = think(user_text)
            log_turn("assistant", reply)
 
            speak(reply)
 
        except KeyboardInterrupt:
            print("\n")
            speak("Okay, bye.")
            break
        except Exception as e:
            print(f"[loop error: {e}]")
            time.sleep(1)
 
    if esp32 is not None and not DEV_MODE:
        esp32.close()
 
 
if __name__ == "__main__":
    main()
 