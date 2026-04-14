#!/usr/bin/env python3
"""Generate placeholder tray icons for Vocal.

Produces three 64x64 PNGs under src/vocal/assets/:
  vocal-awake.png — green, actively listening
  vocal-sleep.png — grey, paused
  vocal-busy.png  — amber, recording / transcribing

Regenerate after tweaking colours or glyph; commit the PNG outputs.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


SIZE = 64
OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "vocal" / "assets"


def _draw_mic(draw: ImageDraw.ImageDraw, fg: tuple[int, int, int, int]) -> None:
    """Draw a simple microphone glyph centered on an SIZE x SIZE canvas."""
    # Mic capsule — rounded rectangle, upper 2/3 vertically
    capsule_w = SIZE * 0.28
    capsule_h = SIZE * 0.42
    cx = SIZE / 2
    cy = SIZE * 0.42
    x0 = cx - capsule_w / 2
    y0 = cy - capsule_h / 2
    x1 = cx + capsule_w / 2
    y1 = cy + capsule_h / 2
    draw.rounded_rectangle((x0, y0, x1, y1), radius=capsule_w / 2, fill=fg)

    # Yoke arc — small semicircle under the capsule
    arc_w = capsule_w + SIZE * 0.12
    arc_h = SIZE * 0.22
    ax0 = cx - arc_w / 2
    ay0 = y1 - arc_h / 2
    ax1 = cx + arc_w / 2
    ay1 = y1 + arc_h / 2
    draw.arc((ax0, ay0, ax1, ay1), start=0, end=180, fill=fg, width=max(2, SIZE // 24))

    # Stand — vertical line from yoke bottom to base
    stand_top = ay1 - SIZE * 0.03
    stand_bot = SIZE * 0.82
    draw.line((cx, stand_top, cx, stand_bot), fill=fg, width=max(2, SIZE // 20))

    # Base — short horizontal line
    base_w = SIZE * 0.22
    draw.line(
        (cx - base_w / 2, stand_bot, cx + base_w / 2, stand_bot),
        fill=fg,
        width=max(2, SIZE // 20),
    )


def _make_icon(bg: tuple[int, int, int, int], fg: tuple[int, int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((2, 2, SIZE - 2, SIZE - 2), fill=bg)
    _draw_mic(draw, fg)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    white = (255, 255, 255, 255)

    targets = {
        "vocal-awake.png": ((46, 160, 67, 255), white),    # green
        "vocal-sleep.png": ((90, 90, 90, 255), (200, 200, 200, 255)),  # grey + lighter grey
        "vocal-busy.png":  ((224, 158, 34, 255), white),   # amber
    }
    for name, (bg, fg) in targets.items():
        img = _make_icon(bg, fg)
        path = OUT_DIR / name
        img.save(path, "PNG")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
