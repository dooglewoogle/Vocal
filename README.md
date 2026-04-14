# Vocal

Local, CPU-only dictation. Speak and text appears in the active window. No cloud, no GPU, no latency surprises.

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with streaming voice activity detection and a system tray icon for always-on operation.

## Platform

- **OS**: Linux (X11/Xorg), macOS (experimental)
- **Python**: 3.10+
- **CPU**: Any x86_64 — runs int8 quantised by default
- **Audio**: Any ALSA/PulseAudio/PipeWire input device
- **Desktop**: System tray required (see [Tray support](#tray-support))

## Installation

### 1. System dependencies

```bash
sudo apt install xdotool xclip portaudio19-dev \
    python3-gi gir1.2-ayatanaappindicator3-0.1 libnotify-bin
```

Or run the setup script:

```bash
./scripts/setup-permissions.sh
```

This also adds your user to the `input` group (required for global hotkeys via evdev). Log out and back in after.

### 2. Install Vocal

```bash
pip install .
```

Or in a venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Quick start

```bash
# Default: live mode — always listening, auto-detects speech
vocal

# Hotkey mode — press Pause key to start/stop recording
vocal --hotkey
```

The first run downloads the Whisper model (~500 MB for `small.en`). Subsequent runs start in seconds.

A tray icon appears: green when listening, grey when paused, amber when transcribing.

## Usage

### Dictation modes

| Mode | Command | How it works |
|------|---------|-------------|
| **Live** (default) | `vocal` | Always-on; VAD detects speech boundaries automatically |
| **Hotkey** | `vocal --hotkey` | Press hotkey to record, press again to transcribe |
| **Push-to-talk** | `vocal --hotkey --mode ptt` | Hold hotkey to record, release to transcribe |

In live mode the hotkey pauses/resumes listening (hold-to-mute in PTT mode).

### Text injection

| Method | Flag | Notes |
|--------|------|-------|
| **Clipboard** (default) | `--output clipboard` | Pastes via Ctrl+V, restores clipboard after |
| **xdotool** | `--output xdotool` | Simulates typing, works everywhere but slower |

### Model selection

```bash
vocal --model small.en           # default, good balance
vocal --model tiny.en            # fastest, lower accuracy
vocal --model medium.en          # slower, higher accuracy
vocal --model base.en            # between tiny and small
```

Find the best model for your hardware:

```bash
vocal --benchmark
vocal --benchmark --benchmark-mic   # test with real mic input
vocal --benchmark --latency-target 1.5
```

### Phrasebook

Teach Vocal your domain-specific vocabulary. Create `~/.config/vocal/phrasebook.toml`:

```toml
[replacements]
"Cooper Netties" = "Kubernetes"
"pie torch" = "PyTorch"
```

Two independent flags control how the phrasebook is used:

| Flag | What it does |
|------|-------------|
| `--phrasebook` | Seeds Whisper's decoder with your vocabulary (biases it toward correct terms) |
| `--phrasebook-replace` | Applies find/replace corrections after transcription |

```bash
# Both layers (recommended)
vocal --phrasebook --phrasebook-replace
```

### All CLI flags

```
Dictation:
  --live                    Live VAD-driven mode (the default)
  --hotkey                  Hotkey-driven mode (press to record)
  --mode {toggle,ptt}       Hotkey mode (default: toggle)
  --key KEY                 Hotkey name: PAUSE, F18, SCROLLLOCK, etc.
  --hotkey-backend {evdev,pynput}
  --output {clipboard,xdotool}
  --silence-ms MS           Min silence before ending utterance (live mode, default: 600)

Model:
  --model MODEL             Whisper model size (default: small.en)
  --compute-type TYPE       int8 or float32 (default: int8)
  --beam-size N             1=greedy, 3=default, 5=thorough

Phrasebook:
  --phrasebook              Seed decoder with known vocabulary
  --phrasebook-replace      Apply replacement rules after transcription

Utilities:
  --list-devices            List audio input devices and exit
  --benchmark               Benchmark all model sizes
  --benchmark-mic           Use live mic for benchmark
  --latency-target SECS     Max acceptable latency for recommendation (default: 2.0)

General:
  --config PATH             Path to config TOML (default: ~/.config/vocal/config.toml)
  --log-level LEVEL         DEBUG, INFO, WARNING, ERROR
```

### Configuration file

All CLI flags can be set permanently in `~/.config/vocal/config.toml`:

```toml
log_level = "INFO"

[model]
size = "small.en"
compute_type = "int8"
beam_size = 3

[audio]
# device = "pulse"       # or device index from --list-devices

[hotkey]
key = "PAUSE"
mode = "toggle"          # or "ptt"
backend = "evdev"        # or "pynput"

[output]
method = "clipboard"     # or "xdotool"

[vad]
threshold = 0.5

[live]
min_silence_duration_ms = 600
min_speech_duration_ms = 250
max_speech_duration_s = 30.0

[postprocess]
capitalize_first = true
remove_filler_words = true
remove_hallucinations = true
```

CLI flags override config file values.

## Tray support

Vocal runs as a system tray application. The tray icon shows the current state and provides a menu to pause/resume or quit.

| Desktop | Status |
|---------|--------|
| KDE Plasma | Works out of the box |
| XFCE | Works out of the box |
| Cinnamon | Works out of the box |
| GNOME | Requires the [AppIndicator and KStatusNotifierItem Support](https://extensions.gnome.org/extension/615/appindicator-support/) extension |
| Sway / i3 | Requires a status bar with StatusNotifierItem support (e.g. waybar) |

## Run at login

Copy the autostart desktop entry:

```bash
cp packaging/vocal-autostart.desktop ~/.config/autostart/
```

Or add to your app menu:

```bash
cp packaging/vocal.desktop ~/.local/share/applications/
```

## Logs

Vocal writes a rotating log file to `~/.local/state/vocal/vocal.log` (Linux) or `~/Library/Logs/vocal/vocal.log` (macOS). The log rotates at 1 MB with 5 backups (~6 MB max). Errors also appear on stderr if running from a terminal.
