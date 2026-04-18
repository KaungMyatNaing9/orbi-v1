# Orbi v1

Agentic voice robot — VAD mic → Whisper STT → Gemini (with tools) → ElevenLabs TTS.
Runs on a Jetson Nano (8 GB) with an ESP32-S3 motor controller and a USB-C webcam.

---

## Hardware

| Part | Notes |
|---|---|
| Jetson Orin Nano 8 GB | JetPack 6.x (Ubuntu 22.04) |
| ESP32-S3 | Connected via USB to Jetson; runs motor firmware |
| USB-C webcam with built-in mic | Plug-and-play; shows up as `/dev/video0` and a ALSA capture device |
| Speaker | USB or 3.5 mm via USB audio adapter |

---

## 1. Flash JetPack

Use the [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) on a host Ubuntu machine to flash **JetPack 6.x** onto the Jetson. After first boot, complete the initial setup wizard.

---

## 2. System dependencies

```bash
sudo apt update && sudo apt upgrade -y

# Audio, serial, build tools
sudo apt install -y \
    portaudio19-dev ffmpeg \
    python3-pip python3-venv python3-dev \
    build-essential libssl-dev libffi-dev \
    v4l-utils alsa-utils

# Verify webcam is detected
v4l2-ctl --list-devices

# Verify mic is detected (look for your USB-C webcam)
arecord -l
```

---

## 3. User permissions (one-time, then reboot)

```bash
sudo usermod -a -G dialout $USER   # ESP32 serial access
sudo usermod -a -G audio   $USER   # mic and speaker
sudo usermod -a -G video   $USER   # webcam
sudo reboot
```

---

## 4. Find the ESP32-S3 serial port

After plugging in the ESP32-S3 via USB:

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

ESP32-S3 with native USB typically appears as `/dev/ttyACM0`. If it shows as `/dev/ttyUSB0`, set the env var accordingly (see step 7).

---

## 5. Install Ollama + Gemma fallback model

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Pull the fallback model (uses ~6 GB of the 8 GB)
ollama pull gemma4:e4b

# Verify it loads
ollama run gemma4:e4b "hello"
```

Ollama runs as a background service automatically after install.

---

## 6. Python environment

```bash
cd ~/orbi-v1

python3 -m venv venv
source venv/bin/activate

# Core deps
pip install --upgrade pip
pip install \
    google-generativeai \
    elevenlabs \
    pyserial \
    opencv-python-headless \
    faster-whisper \
    pyaudio \
    webrtcvad-wheels \
    ollama \
    numpy \
    python-dotenv

# Verify CUDA is visible to Python
python3 -c "import torch; print(torch.cuda.is_available())"
# faster-whisper uses CTranslate2, not torch — verify separately:
python3 -c "from faster_whisper import WhisperModel; m = WhisperModel('tiny', device='cuda'); print('CUDA Whisper OK')"
```

> If `faster-whisper` CUDA fails, install the JetPack-matched CTranslate2 wheel from
> [https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) releases.

---

## 7. Environment variables

Copy the example and fill in your keys:

```bash
cp env.example .env
nano .env
```

`.env` contents:

```
GEMINI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
ORBI_DEV_MODE=0
ORBI_VOICE_ID=EXAVITQu4vr4xnSDxMaL
ORBI_CAMERA=0
ORBI_ESP32_PORT=/dev/ttyACM0
```

Adjust `ORBI_CAMERA` if `v4l2-ctl --list-devices` shows your webcam on a different index.
Adjust `ORBI_ESP32_PORT` to `/dev/ttyUSB0` if that's what appeared in step 4.

---

## 8. ESP32-S3 firmware

The Python code sends JSON commands over serial and expects a JSON ack:

| Command sent | Meaning |
|---|---|
| `{"cmd": "forward", "speed": 50, "distance_cm": 30}` | Drive forward |
| `{"cmd": "backward", "speed": 50, "distance_cm": 30}` | Drive backward |
| `{"cmd": "turn_left", "degrees": 30}` | Turn left |
| `{"cmd": "turn_right", "degrees": 30}` | Turn right |
| `{"cmd": "stop"}` | Stop all motors |

Expected ack from ESP32: any JSON line, e.g. `{"ok": true}`.

Flash your ESP32-S3 with firmware that reads `Serial` at **115200 baud**, parses these JSON payloads, drives your motor driver (L298N, DRV8833, etc.), and writes an ack line back.

---

## 9. Test audio devices

```bash
# Test speaker — you should hear a beep
speaker-test -t wav -c 2

# Test mic recording (5 seconds)
arecord -d 5 -f cd test.wav && aplay test.wav

# If wrong device, list cards and set defaults in /etc/asound.conf
aplay -l
arecord -l
```

If the USB-C webcam mic is not the default, find its card number from `arecord -l` and add to `/etc/asound.conf`:

```
defaults.pcm.card 1
defaults.ctl.card 1
```

Replace `1` with the actual card number of your USB device.

---

## 10. Run

```bash
source venv/bin/activate
python orbi.py
```

On first run you will see a self-test output. All items should show `✓`. Fix any `✗` before using.

---

## 11. Auto-start on boot (optional)

```bash
sudo nano /etc/systemd/system/orbi.service
```

```ini
[Unit]
Description=Orbi Robot Agent
After=network.target sound.target

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
sudo journalctl -u orbi -f   # follow logs
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[Errno -9996] Invalid input device` | Wrong mic index or missing audio group — run `arecord -l` and check permissions |
| `[Errno 13] Permission denied /dev/ttyACM0` | Not in `dialout` group — repeat step 3 and reboot |
| `ffplay: command not found` | `sudo apt install ffmpeg` |
| Whisper loads on CPU even on Jetson | CTranslate2 not built with CUDA — see step 6 note |
| ESP32 connected but no movement | Firmware not flashed or wrong serial port — see step 4 and 8 |
| Ollama times out | Run `ollama serve` manually and check `ollama list` |
