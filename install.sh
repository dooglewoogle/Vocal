#!/usr/bin/env bash
# Vocal installer — one command from a clone to a running app.
#
#   ./install.sh                 full install (system packages, venv, PATH link, app menu, autostart)
#   ./install.sh --no-system     skip apt/brew and the input group (already done, or no sudo)
#   ./install.sh --no-autostart  don't start Vocal at login
#   ./install.sh --dev           editable install (developers)
#
# Environment overrides: VOCAL_HOME (default ~/.local/share/vocal), VOCAL_BIN (default ~/.local/bin),
# PYTHON (interpreter to use; default: the system python3 whose PyGObject the tray needs).
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOCAL_HOME="${VOCAL_HOME:-$HOME/.local/share/vocal}"
VOCAL_BIN="${VOCAL_BIN:-$HOME/.local/bin}"
VENV="$VOCAL_HOME/venv"
DO_SYSTEM=1
DO_AUTOSTART=1
DEV=0
for arg in "$@"; do
    case "$arg" in
        --no-system) DO_SYSTEM=0 ;;
        --no-autostart) DO_AUTOSTART=0 ;;
        --dev) DEV=1 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }

OS="$(uname -s)"
NEED_RELOGIN=0

# ── 1. System packages ─────────────────────────────────────────────────
if [ "$DO_SYSTEM" = 1 ]; then
    case "$OS" in
        Linux)
            if command -v apt-get >/dev/null; then
                say "Installing system packages (sudo)"
                # X11 and Wayland tools both: a few MB, and it removes a question.
                sudo apt-get install -y \
                    python3-venv python3-dev python3-tk python3-gi gir1.2-ayatanaappindicator3-0.1 \
                    portaudio19-dev libnotify-bin espeak-ng \
                    xdotool xclip wtype wl-clipboard
            else
                say "Non-apt distro: install these yourself, then re-run with --no-system"
                note "python3 venv+dev headers, tkinter, PyGObject, AyatanaAppIndicator3 typelib,"
                note "portaudio, libnotify, espeak-ng, xdotool, xclip, wtype, wl-clipboard"
                exit 1
            fi
            if ! id -nG "$USER" | grep -qw input; then
                say "Adding $USER to the 'input' group (global hotkey reads /dev/input)"
                sudo usermod -aG input "$USER"
                NEED_RELOGIN=1
            fi
            ;;
        Darwin)
            say "Installing PortAudio (Homebrew)"
            if command -v brew >/dev/null; then
                brew list portaudio >/dev/null 2>&1 || brew install portaudio
            else
                note "Homebrew not found; install portaudio manually, then re-run with --no-system"
                exit 1
            fi
            ;;
        *)
            echo "Unsupported platform: $OS (Linux and macOS are supported)" >&2
            exit 1
            ;;
    esac
fi

# ── 2. Virtual environment ─────────────────────────────────────────────
# The tray needs the distro's PyGObject; use the same interpreter and let the
# venv see system site-packages. Pick python3 unless PYTHON says otherwise.
PYTHON="${PYTHON:-python3}"
say "Creating virtual environment at $VENV ($("$PYTHON" --version 2>&1))"
mkdir -p "$VOCAL_HOME"
"$PYTHON" -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip

# ── 3. Vocal ───────────────────────────────────────────────────────────
EXTRA=""
[ "$OS" = Linux ] && EXTRA="[hotkey]"
say "Installing Vocal${EXTRA} into the venv"
if [ "$DEV" = 1 ]; then
    "$VENV/bin/pip" install --quiet --no-warn-conflicts -e "${SRC_DIR}${EXTRA}"
else
    "$VENV/bin/pip" install --quiet --no-warn-conflicts "${SRC_DIR}${EXTRA}"
fi

# ── 4. PATH link ───────────────────────────────────────────────────────
mkdir -p "$VOCAL_BIN"
ln -sf "$VENV/bin/vocal" "$VOCAL_BIN/vocal"
say "Linked $VOCAL_BIN/vocal"
case ":$PATH:" in
    *":$VOCAL_BIN:"*) ;;
    *) note "$VOCAL_BIN is not on your PATH yet; most shells add it after a re-login, or add it to your shell rc." ;;
esac

# ── 5. Desktop entry + autostart (Linux) ───────────────────────────────
if [ "$OS" = Linux ]; then
    if [ "$DO_AUTOSTART" = 1 ]; then
        "$VENV/bin/vocal" install-desktop --autostart
    else
        "$VENV/bin/vocal" install-desktop
    fi
fi

# ── Done ───────────────────────────────────────────────────────────────
say "Vocal is installed"
note "Run it:   $VOCAL_BIN/vocal        (or 'vocal' once PATH is updated)"
note "First run downloads the Whisper model (~500 MB) and the default voice (~65 MB)."
if [ "$NEED_RELOGIN" = 1 ]; then
    note "Log out and back in once so the 'input' group applies (needed for the global hotkey)."
fi
if [ "$OS" = Darwin ]; then
    note "macOS: grant Accessibility permission to your terminal for the global hotkey"
    note "(System Settings → Privacy & Security → Accessibility)."
fi
