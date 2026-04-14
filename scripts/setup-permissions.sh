#!/usr/bin/env bash
set -euo pipefail

echo "=== Vocal Setup ==="
echo

case "$(uname -s)" in
    Linux)
        # Check system dependencies
        MISSING=()
        for cmd in xdotool xclip; do
            if ! command -v "$cmd" &>/dev/null; then
                MISSING+=("$cmd")
            fi
        done

        if ! dpkg -l portaudio19-dev &>/dev/null; then
            MISSING+=("portaudio19-dev")
        fi

        # Tray icon dependencies (pystray / AppIndicator)
        if ! python3 -c "import gi" &>/dev/null; then
            MISSING+=("python3-gi")
        fi
        if ! dpkg -l gir1.2-ayatanaappindicator3-0.1 &>/dev/null 2>&1; then
            MISSING+=("gir1.2-ayatanaappindicator3-0.1")
        fi
        if ! command -v notify-send &>/dev/null; then
            MISSING+=("libnotify-bin")
        fi

        if [ ${#MISSING[@]} -gt 0 ]; then
            echo "Installing missing system packages: ${MISSING[*]}"
            sudo apt-get install -y "${MISSING[@]}"
        else
            echo "✓ System packages OK"
        fi

        echo

        # Check input group for evdev
        if groups "$USER" | grep -qw input; then
            echo "✓ User '$USER' is in the 'input' group"
        else
            echo "Adding '$USER' to the 'input' group (for global hotkey via evdev)..."
            sudo usermod -aG input "$USER"
            echo "⚠  You must log out and back in for group changes to take effect."
        fi
        ;;

    Darwin)
        echo "macOS detected"
        echo

        # PortAudio (required by sounddevice)
        if command -v brew &>/dev/null; then
            if brew list portaudio &>/dev/null 2>&1; then
                echo "✓ portaudio already installed (Homebrew)"
            else
                echo "Installing portaudio via Homebrew..."
                brew install portaudio
            fi
        else
            echo "⚠  Homebrew not found. Install portaudio manually:"
            echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "   brew install portaudio"
        fi

        echo
        echo "Accessibility Permissions (required for global hotkey via pynput):"
        echo "  System Settings → Privacy & Security → Accessibility"
        echo "  Add your terminal app (Terminal.app, iTerm2, Alacritty, etc.)"
        ;;

    *)
        echo "⚠  Unsupported platform: $(uname -s)"
        echo "   Vocal currently supports Linux and macOS."
        exit 1
        ;;
esac

echo
echo "=== Setup complete ==="
