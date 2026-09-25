# Vocal

Local, CPU-only voice daemon with two sides:

- **Input (dictation)** — speak and text appears in the active window. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with streaming voice activity detection.
- **Output (speech)** — send text and it is read aloud. Local TTS via [Piper](https://github.com/OHF-Voice/piper1-gpl) or [Kokoro](https://github.com/thewh1teagle/kokoro-onnx), with a localhost HTTP hook so any process can make Vocal talk.

No cloud, no GPU, no latency surprises. A settings window is the main control surface; a system tray icon shows what it is doing. Everything is also reachable from the command line and a TOML file for headless use.

```
 mic ──► input ──► Whisper ──► text ──► active window
                     ▲ paused while speaking
 curl / `vocal say` ──► output ──► Piper/Kokoro ──► speakers
```

## Platform

- **OS**: Linux (X11 and Wayland), macOS (experimental), Windows (untested)
- **Python**: 3.10 – 3.13 (kokoro-onnx has no 3.14 build yet)
- **CPU**: Any x86_64 — runs int8 quantised by default
- **Audio**: Any ALSA/PulseAudio/PipeWire device
- **Desktop**: Tk for the settings window (`python3-tk`), system tray optional (see [Tray support](#tray-support))

## Installation

```bash
git clone https://github.com/dooglewoogle/Vocal.git
cd Vocal
./install.sh
```

That is the whole install. The script installs the system packages (asks for `sudo` once), adds you to the `input` group for the global hotkey, creates a virtual environment under `~/.local/share/vocal`, installs Vocal with both speech engines and the hotkey backend, links `vocal` into `~/.local/bin`, adds an app-menu entry and starts Vocal at login. Log out and back in once afterwards so the `input` group applies.

The script first lists exactly what it will do and asks for confirmation, then prints every command as it runs. Options: `--no-autostart` (don't start at login), `--no-system` (you already installed the system packages, or have no `sudo`), `--dev` (editable install for hacking on Vocal), `--yes` (skip the confirmation), `--agents` / `--no-agents` (see below). Linux with `apt` and macOS with Homebrew are supported; other distros get a list of packages to install by hand. If the default `python3` is too new (3.14 on current macOS), the script picks an installed 3.10–3.13 interpreter, or on a Mac installs Homebrew's `python@3.13` alongside PortAudio.

First run downloads the Whisper model (~500 MB for `small.en`) and, on the first `say`, the Kokoro model behind the default voice `kokoro-bf_emma` (~354 MB). Subsequent runs start in seconds.

### AI coding agents

If Claude Code, OpenAI Codex CLI or Gemini CLI is installed (their `~/.claude`, `~/.codex`, `~/.gemini` directory or command exists), the installer offers a seventh, opt-in step that makes them talk. It is off unless you answer **y** to its own prompt or pass `--agents`; `--yes` alone never enables it. For each detected agent it:

- registers `python -m vocal.hooks.say_hook` (this venv's interpreter, by absolute path) as a hook on the event that carries the assistant's text: Claude Code `MessageDisplay` in `~/.claude/settings.json`, Codex `Stop` in `~/.codex/hooks.json`, Gemini `AfterAgent` in `~/.gemini/settings.json`. Existing hooks and settings are kept; the entry is merged in.
- appends a marked block (`<!-- vocal:begin -->` … `<!-- vocal:end -->`) to the agent's global instructions file (`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, `~/.gemini/GEMINI.md`) telling the model to wrap a one-sentence spoken summary in `<say>…</say>` at the start and end of each reply.

The hook posts every completed `<say>` span to the running daemon's `/say` route from a detached process, so the agent never waits on it; spans inside code blocks are ignored, and nothing happens when the daemon is not running. Start a new agent session after installing.

A span can pick its voice with `<say voice="am_adam">…</say>` (any name `vocal models list` shows; an unknown one speaks in the default voice). The installed instructions don't mention it, so agents keep to the default unless your own instructions ask for a voice, e.g. one per agent.

```bash
vocal install-agents               # all detected agents
vocal install-agents claude codex  # just these
vocal install-agents --list        # detected / installed state and the files involved
vocal install-agents --uninstall   # remove the hook entries and the instruction block
```

<details>
<summary>Manual install (what the script does)</summary>

```bash
# Debian / Ubuntu
sudo apt install python3-venv python3-dev python3-tk python3-gi gir1.2-ayatanaappindicator3-0.1 \
    portaudio19-dev libnotify-bin espeak-ng xdotool xclip wtype wl-clipboard
sudo usermod -aG input $USER           # global hotkey reads /dev/input; re-login afterwards

python3 -m venv --system-site-packages .venv   # needs Python 3.10–3.13; system-site-packages lets the tray see python3-gi
.venv/bin/pip install '.[hotkey]'              # drop [hotkey] to skip the compiled evdev backend
.venv/bin/vocal install-desktop --autostart    # app-menu entry + start at login (optional)
```

`--system-site-packages` is a convenience: Vocal also finds a system PyGObject on its own when it was built for the same Python version. Without either, the window still runs but there is no tray icon.

### Global hotkey backends

| Backend | Platforms | Install | Notes |
|---------|-----------|---------|-------|
| **evdev** | Linux, X11 and Wayland | `[hotkey]` extra | Reads `/dev/input`; your user must be in the `input` group |
| **pynput** | X11, macOS, Windows | included on macOS/Windows; part of `[hotkey]` on Linux | Cannot capture keys under Wayland |

On Linux both backends need the `evdev` C extension (pynput depends on it), which has no prebuilt wheels: `python3-dev` and a compiler must be present when installing `[hotkey]`. Without the extra, Linux has **no global hotkey**: live dictation works, but hotkey mode cannot record and hold-to-mute does nothing. Vocal logs which backend it picked. Text insertion never depends on either; it uses `xdotool`/`xclip` or `wtype`/`wl-clipboard`.

### macOS permissions

macOS grants these to the app you start Vocal from (iTerm, Terminal, VS Code), not to Python. Switch that app on in **System Settings → Privacy & Security**:

| Permission | Needed for |
|---|---|
| **Input Monitoring** | the global hotkey |
| **Accessibility** | typing or pasting the transcript |
| **Automation → System Events** | typing or pasting (macOS asks on the first insertion) |
| **Microphone** | recording (macOS asks on the first recording) |

- Quit that app completely (⌘Q) and reopen it afterwards: macOS only applies grants to newly started apps.
- `vocal permissions` shows what is missing and opens the right panes; the Status tab shows a banner while the hotkey or typing is blocked.
- Don't try to grant the venv's Python: it is a symlink into Homebrew, whose path changes with every `brew upgrade`.
- Secure Keyboard Entry (a terminal menu option) and password fields hide keystrokes from the hotkey.

</details>

## Quick start

```bash
# Run Vocal: settings window + live dictation + speech server + tray icon
vocal

# Tray only, no window (servers, SSH, scripts)
vocal --headless

# Hotkey dictation instead of always-on (also settable in the window)
vocal --hotkey

# Make it talk (from any shell, while the daemon runs)
vocal say "Build finished."
echo "Or pipe text in." | vocal say
```

Tray icon: green when listening, grey when paused, amber when loading, transcribing or speaking.

---

## The window

`vocal` opens a window with four tabs. Closing it hides Vocal to the tray; **Open Vocal** in the tray menu brings it back.

| Tab | What it does |
|-----|--------------|
| **Status** | Current state (Loading / Listening / Recording / Transcribing / Paused, plus Speaking), a log of recent transcriptions, Pause/Resume and Stop speaking |
| **Dictation** | Whisper model grid (cached models marked, pick the current one), then Settings (mode, hotkey, sentence silence, recording ducking) and a collapsed **Advanced** block (model tuning, microphone, text insertion, post-processing, voice detection) |
| **Speech** | Enable speech + host/port on the top row, the voice tree grouped by engine and language with a filter box (download, remove, test, set the default), then Settings (speed, volume, speaker, speaking ducking) and a collapsed **Advanced** block (duck amount, pause-while-speaking, manual model path, log level) |
| **Phrasebook** | Edit mishearing → correction rules and whether they seed recognition / correct output; saving applies them without a model reload |

**Save & Apply** on a tab writes `config.toml` and applies the change live: speech settings take effect immediately, dictation settings restart the dictation engine (a few seconds while the Whisper model reloads).

Values passed on the command line are shown in the window and, if you save, written to the file. Saving rewrites `config.toml` without any comments you had added by hand.

If Tk is not installed Vocal prints a hint and runs headless. If the tray is unavailable, the window runs alone and closing it quits.

---

## Input — dictation

### Modes

| Mode | Command | How it works |
|------|---------|-------------|
| **Live** (default) | `vocal` | Always-on; VAD detects speech boundaries automatically. Hold the hotkey to mute. |
| **Hotkey** | `vocal --hotkey` | Hold the hotkey to record, release to transcribe |

The mode persists as `input.engine` in the config file (or the Dictation tab). Passing `--duck` implies `--hotkey`; add `--live` explicitly to combine them.

### Volume ducking while recording

In hotkey mode, `--duck` lowers the system output volume while you record so playback does not bleed into the mic, then ramps it back over ~300 ms. `--duck-amount` is a relative cut: at 50 (the default), 80% volume drops to 40%. Uses `pactl`, `wpctl`, or `amixer` on Linux and `osascript` on macOS. Live mode never ducks.

```bash
vocal --hotkey --duck --duck-amount 70
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

| Flag | Config key | What it does |
|------|-----------|-------------|
| `--phrasebook` | `input.phrasebook.seed` | Seeds Whisper's decoder with your vocabulary |
| `--phrasebook-replace` | `input.phrasebook.replace` | Applies find/replace corrections after transcription |

The Phrasebook tab in the window edits the same file and applies changes to the running engine immediately.

---

## Output — speech

### Voices

```bash
vocal models list                     # every voice, grouped by engine and language
vocal models list en_GB               # only voices whose name / language / description contains this
vocal models download piper-en_US-lessac-medium
vocal models remove piper-en_US-lessac-medium
```

Every upstream voice is listed: the 54 [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) speakers (US and British English, Spanish, French, Hindi, Italian, Brazilian Portuguese, Japanese, Mandarin) and the 177 [Piper](https://huggingface.co/rhasspy/piper-voices) voices in 53 languages, plus `system`.

| Voices | Size | Notes |
|--------|------|-------|
| `kokoro-<speaker>`, e.g. `kokoro-bf_emma` (**default**), `kokoro-af_sarah` | 354 MB, shared by all | Noticeably more natural, ~0.5–1 s to first audio on CPU. First letter = language, second = gender. Japanese and Mandarin go through espeak and are rough |
| `piper-<lang>-<name>-<quality>`, e.g. `piper-en_US-lessac-medium` | 20–140 MB each | Lowest latency. Multi-speaker models (e.g. `piper-en_GB-vctk-medium`, 109 speakers) take a speaker as `#N`: `piper-en_GB-vctk-medium#12` |
| `system` | — | espeak-ng / macOS `say` / Windows SAPI. No download; robotic |

Names are case-insensitive and the engine prefix is optional: `af_sarah` and `en_us-lessac-medium` work too. The **default voice** (Speech tab, or `output.speech.voice`) speaks whenever a request names no voice or one Vocal doesn't know; if the configured default itself is unknown, `kokoro-bf_emma` is used. Switching between Kokoro speakers, or Piper speakers of one model, doesn't reload anything; Vocal keeps one Kokoro and one Piper model in memory.

Models are stored under `~/.cache/vocal/models/<backend>/` (macOS `~/Library/Caches/vocal/models`, or `$VOCAL_MODELS_DIR`). They download automatically the first time a voice is used (the request waits for it); with `auto_download = false` a voice that isn't downloaded falls back to the default. Every download is checked against a pinned checksum.

Piper voices come from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) on Hugging Face; the catalogue is a snapshot in `src/vocal/output/piper_voices.json`, refreshed with `scripts/gen_piper_voices.py`. Kokoro files (`kokoro-v1.0.onnx` + `voices-v1.0.bin`) come from the [kokoro-onnx GitHub release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0) — the Hugging Face `onnx-community` export is a different, incompatible format.

**Manual / offline install:** put the files anywhere and point `output.speech.model_path` at them — for Piper the `.onnx` file (with its `.onnx.json` alongside), for Kokoro the directory holding the `.onnx` and `voices*.bin`. `voice` selects the backend (and the Kokoro speaker); the registry and downloader are bypassed. `model_path` applies only to the default voice; voices named in requests still use the registry.

### Speaking

```bash
vocal say "Hello there."             # via the daemon; falls back to in-process if none
vocal say -i "Stop everything and say this."   # interrupt: flush queue, speak now
vocal say --voice af_sarah "A different voice, just this once."   # unknown name: default voice + a warning
vocal stop                           # halt and clear the queue
vocal status                         # {"speaking": ..., "queue": ..., "default_voice": ..., "backend": ...}
```

Text is split at sentence boundaries; the next sentence, and any queued request, is synthesized while the current one plays, so long passages start after the first sentence and there is no synthesis gap between sentences. Requests queue FIFO unless `--interrupt` is given.

### HTTP hook (for other processes)

While the daemon runs it listens on `127.0.0.1:47821` (or an ephemeral port if that's taken). No credentials — it is loopback-only. The actual port is in a runtime file so scripts don't have to hard-code it:

- Linux: `$XDG_RUNTIME_DIR/vocal/server.json`
- macOS: `~/Library/Application Support/vocal/server.json`
- Windows: `%LOCALAPPDATA%\vocal\server.json`
- override: `$VOCAL_RUNTIME_FILE`

```bash
curl -s -X POST http://127.0.0.1:47821/say \
     -H 'Content-Type: application/json' \
     -d '{"text": "Deploy complete.", "voice": "bf_emma"}'

# or read the port from the runtime file
PORT=$(jq -r .port "$XDG_RUNTIME_DIR/vocal/server.json")
```

| Route | Body | Response |
|-------|------|----------|
| `POST /say` | `{"text": str, "interrupt"?: bool, "voice"?: str}` | `202 {"ok": true, "queue": n, "voice": str, "fallback": str \| null}` |
| `POST /stop` | — | `200 {"ok": true}` |
| `GET /status` | — | `{"speaking", "queue", "default_voice", "backend"}` |

`voice` in the `/say` reply is the voice that will speak. An unknown `voice` is not an error: the reply names the default voice and `fallback` says why (`"unknown voice 'x'"`). A voice that is known but fails to load at synthesis time (download failed, `auto_download` off) also falls back to the default, and that is only logged.
| `GET /health` | — | `{"ok": true}` |

Requests carrying an `Origin` header get `403`: browsers add one to every cross-origin request, so a web page can't drive your speaker, while curl and scripts (which send none) are unaffected. Disable the server with `--no-server` or `[output.server] enabled = false`.

### What happens to dictation while speaking

- **Live mode** stops consuming the microphone from the first audio frame until playback ends plus `pause_input_tail_ms`, so it never transcribes the speaker. VAD state is reset on resume.
- **Hotkey mode**: pressing the hotkey while speech is playing cuts the speech first, then records.
- **Other apps are ducked** by `duck_amount` percent — per-stream via `pactl set-sink-input-volume` (PulseAudio / PipeWire), so the speech itself stays at full volume. Streams that start mid-utterance are not ducked. Not available on macOS/ALSA-only systems (a warning is logged; set `duck = false` to silence it).

---

## Command reference

Flags override the configuration file for that run.

```
vocal [flags]                       run Vocal: window + daemon (default when no command given)
vocal say [-i] [--voice V] [TEXT…]  speak TEXT, or stdin if omitted / '-'; --voice for this utterance only
vocal stop
vocal status
vocal models [list [FILTER] | download NAME | remove NAME]
vocal install-desktop [--autostart | --no-autostart | --uninstall]
vocal install-agents [claude|codex|gemini …] [--uninstall | --list]
vocal permissions                   macOS: check privacy permissions, open Settings for missing ones

General:
  --headless                Tray icon only, no settings window
  --config PATH             Config TOML (default: ~/.config/vocal/config.toml)
  --log-level LEVEL

Dictation:
  --live / --hotkey         Engine (default: live)
  --key KEY                 Hotkey name: PAUSE, F18, SCROLLLOCK, … (hold to record / hold to mute)
  --hotkey-backend {auto,evdev,pynput}
  --duck / --duck-amount    Duck system volume while recording (hotkey mode)
  --output {clipboard,xdotool}
  --silence-ms MS           Min silence before ending an utterance (live, default 600)
  --model / --compute-type / --beam-size
  --phrasebook / --phrasebook-replace

Speech:
  --voice NAME              Default TTS voice for the daemon
  --no-server               Don't start the HTTP server

Utilities:
  --list-devices            Audio input + output devices
  --benchmark [--benchmark-mic] [--latency-target S]
```

## Configuration file

`~/.config/vocal/config.toml` (macOS: `~/Library/Application Support/vocal/`). CLI flags override it; the Settings tab writes it.

```toml
log_level = "INFO"

# ── Input: dictation ──────────────────────────────────────────
[input]
engine = "live"             # live | hotkey

[input.phrasebook]
seed = false                # bias Whisper toward phrasebook terms
replace = false             # apply phrasebook corrections after transcription

[input.model]
size = "small.en"
compute_type = "int8"
beam_size = 3

[input.audio]
# device = "pulse"          # name substring or index from --list-devices

[input.hotkey]
key = "PAUSE"               # hold to record (hotkey mode) or hold to mute (live mode)
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
voice = "kokoro-bf_emma"   # default voice; also decides the backend (piper / kokoro / system)
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

The tray icon shows the current state and offers **Open Vocal**, **Pause/Resume**, **Stop speaking** and **Quit**. Everything else lives in the window. Without a tray (or on macOS in window mode) closing the window quits Vocal; `--headless` still requires a tray.

| Desktop | Status |
|---------|--------|
| KDE Plasma | Works out of the box |
| XFCE | Works out of the box |
| Cinnamon | Works out of the box |
| GNOME | Requires the [AppIndicator and KStatusNotifierItem Support](https://extensions.gnome.org/extension/615/appindicator-support/) extension |
| Sway / i3 | Requires a status bar with StatusNotifierItem support (e.g. waybar) |

## App menu and run at login

```bash
vocal install-desktop               # app-menu entry + icon, pointing at this venv's vocal
vocal install-desktop --autostart   # …and start Vocal when you log in
vocal install-desktop --uninstall
```

`install.sh` runs this for you; the **Start Vocal at login** checkbox on the Status tab does the same. Entries are written to `~/.local/share/applications/` and `~/.config/autostart/` with the absolute path of the installed executable, so the venv does not need to be on your PATH.

## Logs

Rotating log at `~/.local/state/vocal/vocal.log` (Linux) or `~/Library/Logs/vocal/vocal.log` (macOS); 1 MB × 5 backups. Errors also go to stderr when run from a terminal.
