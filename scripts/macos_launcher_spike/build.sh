#!/usr/bin/env bash
# Build ~/Applications/VocalSpike.app around launcher.c. macOS only; needs the
# Xcode command line tools (Homebrew already requires them).
#   VOCAL_PYTHON=...      venv python to run (default: the install.sh venv)
#   VOCAL_SPIKE_SIGN=...  codesign identity (default "-", ad-hoc)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${VOCAL_PYTHON:-$HOME/.local/share/vocal/venv/bin/python}"
APP="$HOME/Applications/VocalSpike.app"
LOG="$HOME/Library/Logs/VocalSpike.log"
SIGN="${VOCAL_SPIKE_SIGN:--}"

[ "$(uname)" = Darwin ] || { echo "macOS only" >&2; exit 1; }
[ -x "$PY" ] || { echo "No python at $PY; set VOCAL_PYTHON" >&2; exit 1; }

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$(dirname "$LOG")"
clang -O2 -Wall -o "$APP/Contents/MacOS/VocalSpike" "$HERE/launcher.c" \
    -DPYTHON_PATH="\"$PY\"" -DPROBE_PATH="\"$HERE/probe.py\"" -DLOG_PATH="\"$LOG\""

cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>dev.vocal.spike</string>
    <key>CFBundleName</key><string>VocalSpike</string>
    <key>CFBundleExecutable</key><string>VocalSpike</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSUIElement</key><true/>
    <key>NSMicrophoneUsageDescription</key><string>VocalSpike records two seconds to test microphone access.</string>
</dict>
</plist>
EOF

codesign --force --sign "$SIGN" "$APP"
echo "Built $APP (signed with '$SIGN')"
codesign -dv "$APP" 2>&1 | grep -E '^(Identifier|CDHash)='
echo "Log: $LOG"
