#!/usr/bin/env bash
# Vocal installer — one command from a clone to a running app.
#
#   ./install.sh                 full install (system packages, venv, PATH link, app menu, autostart)
#   ./install.sh --no-system     skip apt/brew and the input group (already done, or no sudo)
#   ./install.sh --no-autostart  don't start Vocal at login
#   ./install.sh --dev           editable install (developers)
#   ./install.sh --yes           don't ask for confirmation
#
# Environment overrides: VOCAL_HOME (default ~/.local/share/vocal), VOCAL_BIN (default ~/.local/bin),
# PYTHON (interpreter to use; default: the system python3 whose PyGObject the tray needs).
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOCAL_HOME="${VOCAL_HOME:-$HOME/.local/share/vocal}"
VOCAL_BIN="${VOCAL_BIN:-$HOME/.local/bin}"
VENV="$VOCAL_HOME/venv"
PYTHON="${PYTHON:-python3}"
DO_SYSTEM=1
DO_AUTOSTART=1
DEV=0
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --no-system) DO_SYSTEM=0 ;;
        --no-autostart) DO_AUTOSTART=0 ;;
        --dev) DEV=1 ;;
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

bold() { printf '\033[1m%s\033[0m' "$*"; }
step() { printf '\n%s %s\n' "$(bold "[$1/$TOTAL]")" "$(bold "$2")"; }
run()  { printf '    $ %s\n' "$*"; "$@"; }
note() { printf '    %s\n' "$*"; }
skip() { printf '\n%s %s — skipped (%s)\n' "$(bold "[$1/$TOTAL]")" "$2" "$3"; }

OS="$(uname -s)"
case "$OS" in
    Linux|Darwin) ;;
    *) echo "Unsupported platform: $OS (Linux and macOS are supported)" >&2; exit 1 ;;
esac
if [ "$OS" = Linux ] && [ "$DO_SYSTEM" = 1 ] && ! command -v apt-get >/dev/null; then
    echo "This distro has no apt-get. Install these yourself, then re-run with --no-system:" >&2
    echo "  python3 venv + dev headers, tkinter, PyGObject, AyatanaAppIndicator3 typelib," >&2
    echo "  portaudio, libnotify, espeak-ng, xdotool, xclip, wtype, wl-clipboard" >&2
    exit 1
fi
if ! command -v "$PYTHON" >/dev/null; then
    echo "Python interpreter not found: $PYTHON (set PYTHON=/path/to/python3)" >&2
    exit 1
fi

APT_PKGS="python3-venv python3-dev python3-tk python3-gi gir1.2-ayatanaappindicator3-0.1 portaudio19-dev libnotify-bin espeak-ng xdotool xclip wtype wl-clipboard"
EXTRA=""
[ "$OS" = Linux ] && EXTRA="[hotkey]"
PIP_MODE="install"
[ "$DEV" = 1 ] && PIP_MODE="install --editable (developer mode)"
NEED_GROUP=0
if [ "$OS" = Linux ] && ! id -nG "$USER" | grep -qw input; then
    NEED_GROUP=1
fi
TOTAL=6

# ── The plan ───────────────────────────────────────────────────────────
echo
bold "Vocal installer"; echo
echo
echo "This will:"
if [ "$DO_SYSTEM" = 1 ]; then
    if [ "$OS" = Linux ]; then
        echo "  1. Install system packages with sudo apt-get:"
        echo "       $APT_PKGS"
        if [ "$NEED_GROUP" = 1 ]; then
            echo "  2. Add $USER to the 'input' group (sudo usermod), so the global hotkey can read /dev/input."
            echo "     You will need to log out and back in afterwards."
        else
            echo "  2. Input group: nothing to do, $USER is already in 'input'."
        fi
    else
        echo "  1. Install PortAudio with Homebrew (brew install portaudio)."
        echo "  2. Input group: not needed on macOS."
    fi
else
    echo "  1. System packages: skipped (--no-system)."
    echo "  2. Input group: skipped (--no-system)."
fi
echo "  3. Create a Python virtual environment at $VENV"
echo "       using $PYTHON ($("$PYTHON" --version 2>&1)), with access to system site-packages."
echo "  4. pip $PIP_MODE Vocal${EXTRA} from $SRC_DIR into that venv."
echo "  5. Link $VOCAL_BIN/vocal -> $VENV/bin/vocal so 'vocal' is on your PATH."
if [ "$OS" = Linux ]; then
    if [ "$DO_AUTOSTART" = 1 ]; then
        echo "  6. Add Vocal to the app menu and start it at login"
        echo "       (writes ~/.local/share/applications/vocal.desktop, ~/.config/autostart/vocal.desktop and an icon)."
    else
        echo "  6. Add Vocal to the app menu (no autostart, --no-autostart)."
    fi
else
    echo "  6. Desktop entry: not applicable on macOS."
fi
echo
echo "Nothing outside $VOCAL_HOME, $VOCAL_BIN and the desktop-entry files above is written,"
echo "except the system packages in step 1."
echo
if [ "$ASSUME_YES" != 1 ]; then
    read -r -p "Proceed? [Y/n] " answer
    case "${answer:-Y}" in
        [Yy]|[Yy][Ee][Ss]) ;;
        *) echo "Aborted. Nothing was changed."; exit 0 ;;
    esac
fi

NEED_RELOGIN=0

# ── 1. System packages ─────────────────────────────────────────────────
if [ "$DO_SYSTEM" = 1 ]; then
    if [ "$OS" = Linux ]; then
        step 1 "Installing system packages"
        # shellcheck disable=SC2086
        run sudo apt-get install -y $APT_PKGS
    else
        step 1 "Installing PortAudio"
        if ! command -v brew >/dev/null; then
            echo "    Homebrew not found. Install portaudio manually, then re-run with --no-system." >&2
            exit 1
        fi
        if brew list portaudio >/dev/null 2>&1; then
            note "portaudio is already installed"
        else
            run brew install portaudio
        fi
    fi
else
    skip 1 "System packages" "--no-system"
fi

# ── 2. Input group ─────────────────────────────────────────────────────
if [ "$DO_SYSTEM" = 1 ] && [ "$OS" = Linux ]; then
    step 2 "Input group"
    if [ "$NEED_GROUP" = 1 ]; then
        run sudo usermod -aG input "$USER"
        NEED_RELOGIN=1
        note "added; takes effect after you log out and back in"
    else
        note "$USER is already in the 'input' group"
    fi
elif [ "$OS" = Darwin ]; then
    skip 2 "Input group" "not needed on macOS"
else
    skip 2 "Input group" "--no-system"
fi

# ── 3. Virtual environment ─────────────────────────────────────────────
step 3 "Creating virtual environment"
run mkdir -p "$VOCAL_HOME"
if [ -x "$VENV/bin/python" ]; then
    note "$VENV already exists; reusing it"
else
    run "$PYTHON" -m venv --system-site-packages "$VENV"
fi
run "$VENV/bin/pip" install --quiet --upgrade pip
note "pip $("$VENV/bin/pip" --version | awk '{print $2}')"

# ── 4. Vocal ───────────────────────────────────────────────────────────
step 4 "Installing Vocal${EXTRA}"
if [ "$DEV" = 1 ]; then
    run "$VENV/bin/pip" install --quiet --no-warn-conflicts -e "${SRC_DIR}${EXTRA}"
else
    run "$VENV/bin/pip" install --quiet --no-warn-conflicts "${SRC_DIR}${EXTRA}"
fi
note "installed vocal $("$VENV/bin/python" -c 'import vocal; print(vocal.__version__)')"
if [ "$OS" = Linux ]; then
    note "hotkey backends available: $("$VENV/bin/python" -c 'from vocal.input.hotkey import available_backends as a; print(", ".join(a()) or "none")')"
    TRAY_MISSING="$("$VENV/bin/python" -c 'from vocal.utils import check_tray_dependencies as c; print("; ".join(c()))')"
    if [ -n "$TRAY_MISSING" ]; then
        note "tray icon: NOT available — $TRAY_MISSING"
    else
        note "tray icon: available"
    fi
fi

# ── 5. PATH link ───────────────────────────────────────────────────────
step 5 "Linking $VOCAL_BIN/vocal"
run mkdir -p "$VOCAL_BIN"
run ln -sf "$VENV/bin/vocal" "$VOCAL_BIN/vocal"
case ":$PATH:" in
    *":$VOCAL_BIN:"*) note "$VOCAL_BIN is on your PATH" ;;
    *) note "$VOCAL_BIN is not on your PATH yet; most shells add it after a re-login, or add it to your shell rc." ;;
esac

# ── 6. Desktop entry + autostart ───────────────────────────────────────
if [ "$OS" = Linux ]; then
    if [ "$DO_AUTOSTART" = 1 ]; then
        step 6 "App-menu entry and start at login"
        run "$VENV/bin/vocal" install-desktop --autostart
    else
        step 6 "App-menu entry"
        run "$VENV/bin/vocal" install-desktop
    fi
else
    skip 6 "Desktop entry" "not applicable on macOS"
fi

# ── Done ───────────────────────────────────────────────────────────────
echo
bold "Vocal is installed."; echo
note "Run it:   $VOCAL_BIN/vocal        (or 'vocal' once PATH is updated)"
note "First run downloads the Whisper model (~500 MB) and the default voice (~65 MB)."
if [ "$NEED_RELOGIN" = 1 ]; then
    note "Log out and back in once so the 'input' group applies (needed for the global hotkey)."
fi
if [ "$OS" = Darwin ]; then
    note "macOS: grant Accessibility permission to your terminal for the global hotkey"
    note "(System Settings → Privacy & Security → Accessibility)."
fi
