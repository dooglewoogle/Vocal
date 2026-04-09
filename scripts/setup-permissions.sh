#!/usr/bin/env bash
set -euo pipefail

echo "=== Vocal Setup ==="
echo

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

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Installing missing system packages: ${MISSING[*]}"
    sudo apt-get install -y "${MISSING[@]}"
else
    echo "✓ System packages OK (xdotool, xclip, portaudio19-dev)"
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

echo
echo "=== Setup complete ==="
