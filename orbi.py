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
import ctypes
import ctypes.util
from datetime import datetime
from typing import Optional

# Suppress ALSA's noisy "Unknown PCM" warnings — they are harmless probe failures.
# _ALSA_HANDLER must be kept alive at module level; if it's a temporary, the pointer
# gets garbage collected and ALSA calls freed memory → segfault.
try:
    _asound = ctypes.cdll.LoadLibrary(ctypes.util.find_library("asound"))
    _ALSA_CB_TYPE = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_char_p,
    )
    def _alsa_noop(*_): pass
    _ALSA_HANDLER = _ALSA_CB_TYPE(_alsa_noop)   # module-level ref keeps it alive
    _asound.snd_lib_error_set_handler(_ALSA_HANDLER)
except Exception:
    pass
 
import numpy as np
import pyaudio
import webrtcvad
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from faster_whisper import WhisperModel
from dashboard import dashboard
 
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
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
 
# --- LLM backend -----------------------------------------------------------
# 1 = Gemini (cloud), 2 = Gemma 4 via Ollama (local, no internet needed)
LLM_BACKEND = int(os.getenv("ORBI_LLM", "1"))

# --- Models ----------------------------------------------------------------
GEMINI_MODEL = "gemini-2.0-flash"
GEMMA_LOCAL_MODEL = os.getenv("ORBI_LOCAL_MODEL", "gemma4:e2b")
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
MIC_CARD = int(os.getenv("ORBI_MIC_CARD", "-1"))  # -1 = system default

# --- Motor timing (tune these on real hardware) ----------------------------
MS_PER_CM     = 20   # ms of motor run per cm  (forward / backward)
MS_PER_DEGREE = 8    # ms of motor run per degree (turning)
 
# --- Storage ---------------------------------------------------------------
MEMORY_PATH = pathlib.Path.home() / ".orbi" / "memory.jsonl"
MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
 
# --- Behavior --------------------------------------------------------------
VAD_AGGRESSIVENESS = 2          # 0–3; 2 is safe for USB webcam mics
SILENCE_MS = 900                # ms of silence before utterance is considered done
MAX_UTTERANCE_S = 20            # safety cap on listening window
MIN_WORDS = 2                   # ignore transcripts shorter than this
LLM_COOLDOWN_S = 4.0            # min seconds between LLM calls (avoids 429s)
 
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
_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

# Load local Whisper — try GPU, fall back to CPU if CTranslate2 lacks CUDA
print("Loading Whisper...", flush=True)
_whisper_device = "cpu" if DEV_MODE else "cuda"
_whisper_compute = "int8" if DEV_MODE else "float16"
try:
    whisper = WhisperModel(WHISPER_MODEL, device=_whisper_device, compute_type=_whisper_compute)
    print(f"  Whisper on {_whisper_device}", flush=True)
except ValueError:
    print("  CUDA not available for Whisper, falling back to CPU", flush=True)
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
    resp = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )
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
    result = vision(frame, prompt)
    dashboard.vision(frame, result)
    return result
 
 
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
        cmd = {"cmd": direction, "duration_ms": amount * MS_PER_CM}
    elif direction in ("left", "right"):
        cmd = {"cmd": f"turn_{direction}", "duration_ms": amount * MS_PER_DEGREE}
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
_gemini_chat = _gemini_client.chats.create(
    model=GEMINI_MODEL,
    config=genai_types.GenerateContentConfig(
        system_instruction=SOUL,
        tools=TOOLS,
    ),
) if _gemini_client else None
 
 
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
 
 
def _think_openai(user_text: str) -> str:
    """OpenAI GPT with manual tool loop (same schema format as Ollama)."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [{"role": "system", "content": SOUL}]
    for turn in conversation_history[-10:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_text})

    for _ in range(5):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=OLLAMA_TOOL_SCHEMAS,
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            return (msg.content or "...").strip()
        messages.append(msg)
        for call in tool_calls:
            fn_name = call.function.name
            args = json.loads(call.function.arguments)
            try:
                result = TOOL_FNS[fn_name](**args)
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })
    return "I got stuck thinking. Let's try again."


def _think_claude(user_text: str) -> str:
    """Claude Haiku as last-resort fallback. Basic conversation, no tools."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = []
    for turn in conversation_history[-10:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_text})
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=SOUL,
        messages=messages,
    )
    return resp.content[0].text.strip()


def think(user_text: str) -> str:
    """Cloud chain: Gemini → OpenAI → Claude → local Gemma (ORBI_LLM=2 skips cloud)."""
    dashboard.status("thinking")

    if LLM_BACKEND == 2:
        try:
            return _think_gemma(user_text)
        except Exception as e:
            return f"(my brain is having a moment — {e})"

    if GEMINI_API_KEY and _gemini_chat:
        try:
            return _think_gemini(user_text)
        except Exception as e:
            msg = f"  [gemini: {e}] → OpenAI"
            print(msg); dashboard.log(msg)

    if OPENAI_API_KEY:
        try:
            return _think_openai(user_text)
        except Exception as e:
            msg = f"  [openai: {e}] → Claude"
            print(msg); dashboard.log(msg)

    if ANTHROPIC_API_KEY:
        try:
            return _think_claude(user_text)
        except Exception as e:
            msg = f"  [claude: {e}] → local Gemma"
            print(msg); dashboard.log(msg)

    try:
        return _think_gemma(user_text)
    except Exception as e:
        return f"(all backends failed — {e})"
 
 
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
        time.sleep(0.3)

    dashboard.status("listening")
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    pa = pyaudio.PyAudio()
    frame_duration_ms = 30
    frame_size = int(MIC_SAMPLE_RATE * frame_duration_ms / 1000)
 
    try:
        mic_kwargs = dict(
            format=pyaudio.paInt16, channels=2, rate=MIC_SAMPLE_RATE,
            input=True, frames_per_buffer=frame_size,
        )
        if MIC_CARD >= 0:
            mic_kwargs["input_device_index"] = MIC_CARD
        stream = pa.open(**mic_kwargs)
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
            # Downmix stereo to mono for VAD (take left channel)
            mono_frame = np.frombuffer(frame, dtype=np.int16)[::2].tobytes()
            try:
                is_speech = vad.is_speech(mono_frame, MIC_SAMPLE_RATE)
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
    # Downmix stereo to mono for Whisper (average both channels)
    stereo = np.frombuffer(audio_bytes, dtype=np.int16).reshape(-1, 2)
    pcm = stereo.mean(axis=1).astype(np.float32) / 32768.0
    segments, _ = whisper.transcribe(pcm, language="en", beam_size=1)
    return " ".join(seg.text for seg in segments).strip()
 
 
# ═══════════════════════════════════════════════════════════════════════════
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃  🔊 SPEAK — ElevenLabs, mic muted during playback                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ═══════════════════════════════════════════════════════════════════════════
 
def speak(text: str) -> None:
    """Speak via ElevenLabs. Streams audio to dashboard (Mac browser plays it)."""
    if not text:
        return
    print(f"🤖 Orbi: {text}")
    if eleven is None:
        dashboard.status("idle")
        return
    try:
        IS_SPEAKING.set()
        dashboard.status("speaking")
        audio_stream = eleven.text_to_speech.convert(
            voice_id=ELEVEN_VOICE_ID, text=text,
            model_id=ELEVEN_MODEL, output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_stream)
        dashboard.audio(audio_bytes)
        # Block while browser plays (~0.45s per word)
        time.sleep(max(1.5, len(text.split()) * 0.45))
    except Exception as e:
        print(f"  [speak error: {e}]")
    finally:
        IS_SPEAKING.clear()
        dashboard.status("idle")
 
 
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
 
# Whisper often hallucinates these when it hears background noise
_NOISE_PHRASES = {
    "you", "the", "a", "uh", "um", "hmm", "hm", "okay", "ok",
    "yeah", "yes", "no", "thanks", "thank you", "bye", "goodbye",
    "thank you.", "okay.", "yes.", "no.", "hmm.", "uh-huh",
}

def _is_noise(text: str) -> bool:
    """Return True if the transcript is too short or a known hallucination."""
    cleaned = text.lower().strip().rstrip(".,!?")
    if cleaned in _NOISE_PHRASES:
        return True
    if len(cleaned.split()) < MIN_WORDS:
        return True
    return False


def log_turn(role: str, content: str) -> None:
    conversation_history.append({"role": role, "content": content})
    if role == "user":
        dashboard.log(f"👤 You: {content}")
    elif role == "assistant":
        dashboard.log(f"🤖 Orbi: {content}")
    with MEMORY_PATH.open("a") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": "conversation",
            "role": role,
            "content": content,
        }) + "\n")
 
 
def main() -> None:
    dashboard.start()
    if not self_test():
        print("\nFix the ✗ items above, then re-run.")
        sys.exit(1)
 
    print(f"\nOrbi online. Dev mode: {DEV_MODE}. Ctrl+C to exit.\n")
    speak("Hey. I'm here.")

    _last_llm_call = 0.0
    _last_heartbeat = 0.0
    while True:
        try:
            print("🎧 listening...")
            user_text = listen()

            # Heartbeat to dashboard every 10s so it shows the system is alive
            now = time.time()
            if now - _last_heartbeat > 10:
                dashboard.log("🎧 listening...")
                dashboard.status("listening")
                _last_heartbeat = now

            if not user_text or _is_noise(user_text):
                if user_text:
                    print(f"  [filtered: {user_text!r}]")
                    dashboard.log(f"  [filtered: {user_text!r}]")
                continue

            # Enforce cooldown to avoid 429s
            elapsed = time.time() - _last_llm_call
            if elapsed < LLM_COOLDOWN_S:
                wait = LLM_COOLDOWN_S - elapsed
                print(f"  [cooldown {wait:.1f}s]")
                dashboard.log(f"  [cooldown {wait:.1f}s]")
                time.sleep(wait)

            print(f"👤 You: {user_text}")
            log_turn("user", user_text)

            reply = think(user_text)
            _last_llm_call = time.time()
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
 