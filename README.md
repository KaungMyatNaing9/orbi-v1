# Orbi

**Give your hardware a brain. Let it speak, think, see, and move — driven by language, not code.**

Orbi is not a robot that follows a program. It is a physical agent: it listens to you, understands context, decides what to do, uses its tools (camera, wheels, memory), and responds with a voice. You talk to it like a person. It acts like one.

---

## The idea

Traditional robotics is built on CV pipelines, state machines, and hand-coded decision trees. Every behaviour has to be explicitly programmed. It works, but it doesn't *think*.

Orbi replaces all of that with a single LLM agent loop:

```
Mic → Whisper STT → LLM (with tools) → ElevenLabs TTS → Speaker
                         ↓
               see / move / remember / recall
```

The LLM decides — on every turn — whether to look through the camera, drive the wheels, save a memory, or just talk. You don't write behaviours. You write a soul.

---

## What Orbi v1 can do

- **Hear you** — VAD-gated microphone captures speech, Whisper transcribes it locally
- **Think** — multi-LLM cloud chain (Gemini → OpenAI → Claude) with local Gemma fallback
- **See** — captures a frame from the webcam and describes it using a vision model
- **Move** — drives an ESP32-controlled wheeled robot via serial JSON commands
- **Remember** — stores long-term memories to disk, recalls them by keyword search
- **Speak** — ElevenLabs voice streamed to a browser dashboard (no speaker on the robot needed)
- **Show you everything** — real-time web dashboard: logs, camera feed, status, audio

---

## The future — Orbi OS

Orbi v1 is a proof of concept running on a Jetson Nano. The goal is **Orbi OS**: a downloadable agent runtime anyone can flash onto their hardware to give it life.

**What Orbi OS will be:**
- A deployable software layer — download, configure, run
- Hardware-agnostic — works on any board that can run Python (Jetson, Raspberry Pi, x86)
- Full agent capabilities out of the box: web search, browser control, file system, API calls
- A marketplace of tools and personalities (souls) the community can publish and install
- Local-first — runs offline with local LLMs, cloud APIs optional
- Open source

The shift is philosophical: instead of programming a robot to do things, you give it the ability to figure out what to do. The hardware becomes a body. The LLM becomes the mind.

---

## Hardware (v1)

| Part | Notes |
|---|---|
| Jetson Orin Nano 8 GB | JetPack 6.x (Ubuntu 22.04) |
| ESP32-S3 | USB to Jetson, runs motor firmware |
| USB-C webcam with mic | Plug-and-play |
| Wheeled chassis + motor driver | L298N or similar |

---

## Setup on Jetson

### 1. System dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y portaudio19-dev ffmpeg python3-pip python3-venv \
                    build-essential v4l-utils alsa-utils
```

### 2. Permissions (reboot after)

```bash
sudo usermod -a -G dialout,audio,video $USER
sudo reboot
```

### 3. Find ESP32 port

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

ESP32-S3 with native USB → `/dev/ttyACM0`. CP2102/CH340 chip → `/dev/ttyUSB0`.

### 4. Ollama + local model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:e2b
```

### 5. Clone and install

```bash
git clone https://github.com/your-username/orbi-v1.git
cd orbi-v1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 6. Configure

```bash
cp env.example .env
nano .env
```

Key variables:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Primary LLM (free tier available) |
| `OPENAI_API_KEY` | Fallback LLM |
| `ANTHROPIC_API_KEY` | Last-resort LLM fallback |
| `ELEVENLABS_API_KEY` | Voice synthesis |
| `ORBI_LLM` | `1` = cloud chain, `2` = local Gemma only |
| `ORBI_LOCAL_MODEL` | Default `gemma4:e2b` |
| `ORBI_DEV_MODE` | `0` on Jetson, `1` on Mac |
| `ORBI_ESP32_PORT` | Default `/dev/ttyACM0` |
| `ORBI_CAMERA` | Default `0` |

### 7. Flash ESP32

Open `esp32_firmware/orbi_motors.ino` in Arduino IDE.
Install **ArduinoJson** via Library Manager. Select board **ESP32S3 Dev Module**. Flash.

### 8. Run

```bash
source venv/bin/activate
python orbi.py
```

Then open **`http://<jetson-ip>:8080`** on your Mac to see the live dashboard.
Click anywhere on the page once to unlock audio — after that Orbi's voice plays through your Mac speakers automatically.

### 9. Auto-start on boot (optional)

```bash
sudo nano /etc/systemd/system/orbi.service
```

```ini
[Unit]
Description=Orbi Agent
After=network.target

[Service]
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/orbi-v1
EnvironmentFile=/home/YOUR_USERNAME/orbi-v1/.env
ExecStart=/home/YOUR_USERNAME/orbi-v1/venv/bin/python orbi.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable orbi
sudo systemctl start orbi
```

---

## LLM backend

Orbi tries each backend in order, falling back automatically:

```
Gemini  →  OpenAI (gpt-4o-mini)  →  Claude (Haiku)  →  local Gemma
```

Set `ORBI_LLM=2` in `.env` to skip cloud entirely and run fully offline.

---

## Tuning motor timing

After flashing the ESP32, test movement and adjust in `orbi.py`:

```python
MS_PER_CM     = 20   # ms per cm  (forward/backward)
MS_PER_DEGREE = 8    # ms per degree (turning)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied /dev/ttyACM0` | Not in `dialout` group — see step 2 |
| No audio in browser | Click the page once to unlock Web Audio API |
| `[Errno -9996]` mic error | Check `arecord -l`, set correct ALSA card |
| Whisper loads on CPU | CTranslate2 not CUDA-built — see faster-whisper docs |
| All cloud APIs fail | Set `ORBI_LLM=2` to use local Gemma |
| Motors don't move | Wrong serial port or ESP32 not flashed |
