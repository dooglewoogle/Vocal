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
# PYTHON (interpreter to use, 3.10-3.13; default: first supported of python3, python3.13 ... python3.10).
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOCAL_HOME="${VOCAL_HOME:-$HOME/.local/share/vocal}"
VOCAL_BIN="${VOCAL_BIN:-$HOME/.local/bin}"
VENV="$VOCAL_HOME/venv"
PYTHON_OVERRIDE="${PYTHON:-}"
PYTHON=""
# Python minors Vocal can run on, newest first. Must match requires-python in pyproject.toml
# (tests/test_install_sh.py enforces this). The ceiling comes from kokoro-onnx, which has no 3.14 release.
SUPPORTED_PYTHONS="3.13 3.12 3.11 3.10"
NEED_PYTHON=""   # set to a minor when step 1 must brew-install python@X.Y first
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
# Interpreter probes use only `--version` so a broken shim (macOS without CLT) is skipped, not fatal.
py_version() { "$1" --version 2>/dev/null | sed -nE 's/^Python ([0-9][0-9.]*).*/\1/p' || true; }
py_minor()   { py_version "$1" | sed -E 's/^([0-9]+\.[0-9]+).*/\1/'; }
py_supported() {
    local m; m="$(py_minor "$1")"
    [ -n "$m" ] || return 1
    case " $SUPPORTED_PYTHONS " in *" $m "*) return 0 ;; esac
    return 1
}

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

# ── Interpreter ────────────────────────────────────────────────────────
PY_NEWEST="${SUPPORTED_PYTHONS%% *}"
PY_RANGE="${SUPPORTED_PYTHONS##* }–$PY_NEWEST"
PY_WHY="kokoro-onnx has no 3.14 build yet"
BREW_PREFIX=""
if [ "$OS" = Darwin ] && command -v brew >/dev/null; then
    BREW_PREFIX="$(brew --prefix)"
fi
SYS_PY="$(py_version python3)"; SYS_PY="${SYS_PY:-not found}"
if [ -n "$PYTHON_OVERRIDE" ]; then
    if ! command -v "$PYTHON_OVERRIDE" >/dev/null; then
        echo "Python interpreter not found: $PYTHON_OVERRIDE (from PYTHON=...)" >&2
        exit 1
    fi
    if ! py_supported "$PYTHON_OVERRIDE"; then
        echo "PYTHON=$PYTHON_OVERRIDE is Python $(py_version "$PYTHON_OVERRIDE"); Vocal needs $PY_RANGE ($PY_WHY)." >&2
        exit 1
    fi
    PYTHON="$PYTHON_OVERRIDE"
else
    CANDIDATES="python3"
    for v in $SUPPORTED_PYTHONS; do
        CANDIDATES="$CANDIDATES python$v"
        if [ -n "$BREW_PREFIX" ]; then
            CANDIDATES="$CANDIDATES $BREW_PREFIX/opt/python@$v/bin/python$v"
        fi
    done
    PROBED=""
    for c in $CANDIDATES; do
        command -v "$c" >/dev/null 2>&1 || continue
        if py_supported "$c"; then
            PYTHON="$c"
            break
        fi
        ver="$(py_version "$c")"
        PROBED="$PROBED  $c = ${ver:-not a working Python}"$'\n'
    done
    if [ -z "$PYTHON" ]; then
        if [ "$OS" = Darwin ] && [ -n "$BREW_PREFIX" ] && [ "$DO_SYSTEM" = 1 ]; then
            NEED_PYTHON="$PY_NEWEST"
            PYTHON="$BREW_PREFIX/opt/python@$NEED_PYTHON/bin/python$NEED_PYTHON"
        else
            {
                echo "Vocal needs Python $PY_RANGE ($PY_WHY)."
                echo "Found:"
                printf '%s' "${PROBED:-  no python3 on PATH}"
                echo "Install one and re-run, or point PYTHON at one:"
                echo "  Debian/Ubuntu: sudo apt-get install python$PY_NEWEST python$PY_NEWEST-venv    (or the deadsnakes PPA)"
                echo "  macOS:         brew install python@$PY_NEWEST"
                echo "  then:          PYTHON=python$PY_NEWEST ./install.sh"
            } >&2
            exit 1
        fi
    fi
fi
if [ -n "$NEED_PYTHON" ]; then
    PY_DESC="python$NEED_PYTHON (installed in step 1)"
else
    PY_DESC="$PYTHON (Python $(py_version "$PYTHON"))"
fi
# An existing venv on an unsupported interpreter (e.g. left by a failed run) is replaced, not reused.
VENV_STALE=""
if [ -x "$VENV/bin/python" ] && ! py_supported "$VENV/bin/python"; then
    VENV_STALE="$(py_version "$VENV/bin/python")"
    : "${VENV_STALE:=unknown}"
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
        if [ -n "$NEED_PYTHON" ]; then
            echo "  1. Install PortAudio and Python $NEED_PYTHON with Homebrew (brew install portaudio python@$NEED_PYTHON)."
            echo "       python3 here is $SYS_PY; Vocal needs $PY_RANGE because $PY_WHY."
        else
            echo "  1. Install PortAudio with Homebrew (brew install portaudio)."
        fi
        echo "  2. Input group: not needed on macOS."
    fi
else
    echo "  1. System packages: skipped (--no-system)."
    echo "  2. Input group: skipped (--no-system)."
fi
if [ -n "$VENV_STALE" ]; then
    echo "  3. Replace the virtual environment at $VENV (it uses Python $VENV_STALE, unsupported)"
else
    echo "  3. Create a Python virtual environment at $VENV"
fi
echo "       using $PY_DESC, with access to system site-packages."
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
        if [ -n "$NEED_PYTHON" ]; then
            step 1 "Installing PortAudio and Python $NEED_PYTHON"
        else
            step 1 "Installing PortAudio"
        fi
        if ! command -v brew >/dev/null; then
            echo "    Homebrew not found. Install portaudio manually, then re-run with --no-system." >&2
            exit 1
        fi
        if brew list portaudio >/dev/null 2>&1; then
            note "portaudio is already installed"
        else
            run brew install portaudio
        fi
        if [ -n "$NEED_PYTHON" ]; then
            if brew list "python@$NEED_PYTHON" >/dev/null 2>&1; then
                note "python@$NEED_PYTHON is already installed"
            else
                run brew install "python@$NEED_PYTHON"
            fi
            if [ ! -x "$PYTHON" ]; then
                echo "    Expected $PYTHON after installing python@$NEED_PYTHON, but it is not there." >&2
                exit 1
            fi
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
if [ -n "$VENV_STALE" ]; then
    note "$VENV uses Python $VENV_STALE (unsupported); replacing it"
    run rm -rf "$VENV"
fi
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
