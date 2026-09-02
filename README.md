# Vocal

Local, CPU-only voice daemon with two sides:

- **Input (dictation)** — speak and text appears in the active window. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with streaming voice activity detection.
- **Output (speech)** — send text and it is read aloud. Local TTS via [Piper](https://github.com/OHF-Voice/piper1-gpl) or [Kokoro](https://github.com/thewh1teagle/kokoro-onnx), with a localhost HTTP hook so any process can make Vocal talk.

No cloud, no GPU, no latency surprises. A system tray icon shows what it is doing.

```
 mic ──► input ──► Whisper ──► text ──► active window
                     ▲ paused while speaking
 curl / `vocal say` ──► output ──► Piper/Kokoro ──► speakers
```

## Platform

- **OS**: Linux (X11 and Wayland), macOS (experimental), Windows (untested)
- **Python**: 3.10+
- **CPU**: Any x86_64 — runs int8 quantised by default
- **Audio**: Any ALSA/PulseAudio/PipeWire device
- **Desktop**: System tray required (see [Tray support](#tray-support))

## Installation

### 1. System dependencies

```bash
# X11
sudo apt install xdotool xclip portaudio19-dev \
    python3-gi gir1.2-ayatanaappindicator3-0.1 libnotify-bin
# Wayland: replace xdotool xclip with
sudo apt install wtype wl-clipboard
# Optional: OS text-to-speech fallback
sudo apt install espeak-ng
```

Or run the setup script:

```bash
./scripts/setup-permissions.sh
```

This also adds your user to the `input` group (required for global hotkeys via evdev). Log out and back in after.

### 2. Install Vocal

```bash
python -m venv .venv
source .venv/bin/activate
pip install '.[tts-piper]'          # dictation + Piper speech (recommended)
# pip install '.[tts]'              # Piper + Kokoro
# pip install .                     # dictation only; speech falls back to OS TTS
```

## Quick start

```bash
# Run the daemon: live dictation + speech server + tray icon
vocal

# Hotkey dictation instead of always-on
vocal --hotkey

# Make it talk (from any shell, while the daemon runs)
vocal say "Build finished."
echo "Or pipe text in." | vocal say
```

First run downloads the Whisper model (~500 MB for `small.en`) and, on the first `say`, the default Piper voice (~65 MB). Subsequent runs start in seconds.

Tray icon: green when listening, grey when paused, amber when transcribing or speaking.

---

## Input — dictation

### Modes

| Mode | Command | How it works |
|------|---------|-------------|
| **Live** (default) | `vocal` | Always-on; VAD detects speech boundaries automatically |
| **Hotkey** | `vocal --hotkey` | Press hotkey to record, press again to transcribe |
| **Push-to-talk** | `vocal --mode ptt` | Hold hotkey to record, release to transcribe |

Passing `--mode` or `--duck` implies `--hotkey`. Add `--live` explicitly to combine them with live mode, where the hotkey pauses/resumes listening (hold-to-mute in PTT mode).

### Volume ducking while recording

In hotkey mode, `--duck` lowers the system output volume while you record so playback does not bleed into the mic, then ramps it back over ~300 ms. `--duck-amount` is a relative cut: at 50 (the default), 80% volume drops to 40%. Uses `pactl`, `wpctl`, or `amixer` on Linux and `osascript` on macOS. Live mode never ducks.

```bash
vocal --hotkey --mode ptt --duck --duck-amount 70
```

### Text injection

| Method | Flag | Notes |
|--------|------|-------|
| **Clipboard** (default) | `--output clipboard` | Pastes via Ctrl+V, restores clipboard after |
| **xdotool** | `--output xdotool` | Simulates typing (wtype on Wayland), slower but works everywhere |

### Whisper model

```bash
vocal --model small.en           # default, good balance
vocal --model tiny.en            # fastest, lower accuracy
vocal --model medium.en          # slower, higher accuracy
vocal --benchmark                # find the best model for this machine
```

### Phrasebook

Teach Vocal your vocabulary in `~/.config/vocal/phrasebook.toml`:

```toml
[replacements]
"Cooper Netties" = "Kubernetes"
"pie torch" = "PyTorch"
```

| Flag | What it does |
|------|-------------|
| `--phrasebook` | Seeds Whisper's decoder with your vocabulary |
| `--phrasebook-replace` | Applies find/replace corrections after transcription |

---

## Output — speech

### Voices

```bash
vocal models list                          # what's available / downloaded
vocal models download piper-en-lessac-medium
vocal models remove piper-en-lessac-medium
```

| Voice | Backend | Size | Notes |
|-------|---------|------|-------|
| `piper-en-lessac-medium` | Piper | 65 MB | **Default.** Lowest latency, good quality |
| `piper-en-amy-low` | Piper | 30 MB | Smallest/fastest |
| `piper-en-gb-alan-medium` | Piper | 65 MB | British English |
| `kokoro-af_sarah` / `am_adam` / `bf_emma` | Kokoro | 330 MB (shared) | Noticeably more natural, ~0.5–1 s to first audio on CPU |
| `system` | OS | — | espeak-ng / macOS `say` / Windows SAPI. No download; robotic |

Models are stored under `~/.cache/vocal/models/<backend>/` (macOS `~/Library/Caches/vocal/models`, or `$VOCAL_MODELS_DIR`). They download automatically the first time a voice is used; set `auto_download = false` to require the explicit command.

Piper voices come from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) on Hugging Face. Kokoro files (`kokoro-v1.0.onnx` + `voices-v1.0.bin`) come from the [kokoro-onnx GitHub release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0) — the Hugging Face `onnx-community` export is a different, incompatible format.

**Manual / offline install:** put the files anywhere and point `output.speech.model_path` at them — for Piper the `.onnx` file (with its `.onnx.json` alongside), for Kokoro the directory holding the `.onnx` and `voices*.bin`. `voice` still selects the backend (and the Kokoro speaker); the registry and downloader are bypassed.

### Speaking

```bash
vocal say "Hello there."             # via the daemon; falls back to in-process if none
vocal say -i "Stop everything and say this."   # interrupt: flush queue, speak now
vocal say --voice kokoro-af_sarah "A different voice, just this once."
vocal stop                           # halt and clear the queue
vocal status                         # {"speaking": ..., "queue": ..., "voice": ..., "backend": ...}
```

Text is split at sentence boundaries and synthesized sentence-by-sentence, so long passages start playing after the first sentence. Requests queue FIFO unless `--interrupt` is given.

### HTTP hook (for other processes)

While the daemon runs it listens on `127.0.0.1:47821` (or an ephemeral port if that's taken). The port and a per-run bearer token live in a `0600` runtime file:

- Linux: `$XDG_RUNTIME_DIR/vocal/server.json`
- macOS: `~/Library/Application Support/vocal/server.json`
- Windows: `%LOCALAPPDATA%\vocal\server.json`
- override: `$VOCAL_RUNTIME_FILE`

```bash
RT=${XDG_RUNTIME_DIR}/vocal/server.json
PORT=$(jq -r .port "$RT"); TOKEN=$(jq -r .token "$RT")

curl -s -X POST "http://127.0.0.1:$PORT/say" \
     -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"text": "Deploy complete.", "interrupt": false}'
```

| Route | Body | Response |
|-------|------|----------|
| `POST /say` | `{"text": str, "interrupt"?: bool, "voice"?: str}` | `202 {"ok": true, "queue": n}` |
| `POST /stop` | — | `200 {"ok": true}` |
| `GET /status` | — | `{"speaking", "queue", "voice", "backend"}` |
| `GET /health` | — | `{"ok": true}` |

Missing or wrong token → `401`. Disable the server with `--no-server` or `[output.server] enabled = false`.

### What happens to dictation while speaking

- **Live mode** stops consuming the microphone from the first audio frame until playback ends plus `pause_input_tail_ms`, so it never transcribes the speaker. VAD state is reset on resume.
- **Hotkey mode**: pressing the hotkey while speech is playing cuts the speech first, then records.
- **Other apps are ducked** by `duck_amount` percent — per-stream via `pactl set-sink-input-volume` (PulseAudio / PipeWire), so the speech itself stays at full volume. Streams that start mid-utterance are not ducked. Not available on macOS/ALSA-only systems (a warning is logged; set `duck = false` to silence it).

---

## Command reference

```
vocal [flags]                       run the daemon (default when no command given)
vocal say [-i] [--voice V] [TEXT…]  speak TEXT, or stdin if omitted / '-'
vocal stop
vocal status
vocal models [list | download NAME | remove NAME]

Dictation:
  --live / --hotkey         Engine (default: live)
  --mode {toggle,ptt}       Hotkey behaviour (implies --hotkey)
  --key KEY                 Hotkey name: PAUSE, F18, SCROLLLOCK, …
  --hotkey-backend {auto,evdev,pynput}
  --duck / --duck-amount    Duck system volume while recording (hotkey mode)
  --output {clipboard,xdotool}
  --silence-ms MS           Min silence before ending an utterance (live, default 600)
  --model / --compute-type / --beam-size
  --phrasebook / --phrasebook-replace

Speech:
  --voice NAME              Default TTS voice for the daemon
  --tts-backend {piper,kokoro,system}
  --no-server               Don't start the HTTP server

Utilities:
  --list-devices            Audio input + output devices
  --benchmark [--benchmark-mic] [--latency-target S]

General:
  --config PATH             Config TOML (default: ~/.config/vocal/config.toml)
  --log-level LEVEL
```

## Configuration file

`~/.config/vocal/config.toml` (macOS: `~/Library/Application Support/vocal/`). CLI flags override it.

```toml
log_level = "INFO"

# ── Input: dictation ──────────────────────────────────────────
[input.model]
size = "small.en"
compute_type = "int8"
beam_size = 3

[input.audio]
# device = "pulse"          # name substring or index from --list-devices

[input.hotkey]
key = "PAUSE"
mode = "toggle"             # or "ptt"
backend = "auto"            # auto | evdev | pynput
duck = false                # duck system volume while recording
duck_amount = 50

[input.inject]
method = "clipboard"        # or "xdotool"
xdotool_delay = 8

[input.vad]
threshold = 0.5

[input.live]
min_silence_duration_ms = 600
min_speech_duration_ms = 250
max_speech_duration_s = 30.0

[input.postprocess]
capitalize_first = true
remove_filler_words = true
remove_hallucinations = true

# ── Output: speech ────────────────────────────────────────────
[output.speech]
backend = "piper"           # piper | kokoro | system (used with model_path)
voice = "piper-en-lessac-medium"
# model_path = "~/voices/en_US-lessac-medium.onnx"   # manual install; bypasses download
auto_download = true
speed = 1.0
volume = 100                # digital gain 0-100
# device = "USB Audio"      # output device, name substring or index
pause_input = true          # stop dictation while speaking
pause_input_tail_ms = 300
duck = true                 # duck other apps' streams while speaking (pactl)
duck_amount = 50

[output.server]
enabled = true
host = "127.0.0.1"
port = 47821
```

### Migrating from 0.2

Config tables moved under `input` / `output`. Vocal refuses to start with the old layout and prints the mapping:

| 0.2 | 0.3 |
|-----|-----|
| `[model]` | `[input.model]` |
| `[audio]` | `[input.audio]` |
| `[hotkey]` | `[input.hotkey]` |
| `[output]` (text injection) | `[input.inject]` |
| `[vad]` / `[live]` / `[postprocess]` | `[input.vad]` / `[input.live]` / `[input.postprocess]` |

## Tray support

The tray icon shows the current state and offers: Pause/Resume, Audio Device, Model, Mode, **Voice** (pick a TTS voice; undownloaded ones are marked), **Stop speaking**, Edit Phrasebook, Quit.

| Desktop | Status |
|---------|--------|
| KDE Plasma | Works out of the box |
| XFCE | Works out of the box |
| Cinnamon | Works out of the box |
| GNOME | Requires the [AppIndicator and KStatusNotifierItem Support](https://extensions.gnome.org/extension/615/appindicator-support/) extension |
| Sway / i3 | Requires a status bar with StatusNotifierItem support (e.g. waybar) |

## Run at login

```bash
cp packaging/vocal-autostart.desktop ~/.config/autostart/     # autostart
cp packaging/vocal.desktop ~/.local/share/applications/       # app menu
```

## Logs

Rotating log at `~/.local/state/vocal/vocal.log` (Linux) or `~/Library/Logs/vocal/vocal.log` (macOS); 1 MB × 5 backups. Errors also go to stderr when run from a terminal.
