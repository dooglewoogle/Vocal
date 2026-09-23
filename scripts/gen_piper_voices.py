#!/usr/bin/env python3
"""Regenerate src/vocal/output/piper_voices.json from the Piper voice catalogue.

Reads ``voices.json`` from rhasspy/piper-voices (or a local copy given as the
first argument) and keeps only what the registry needs: key, language, quality,
speaker count and the .onnx/.onnx.json files with their sizes and md5 digests.
Run after upstream adds voices; commit the JSON output.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
OUT = Path(__file__).resolve().parent.parent / "src" / "vocal" / "output" / "piper_voices.json"


def main() -> None:
    if len(sys.argv) > 1:
        data = json.loads(Path(sys.argv[1]).read_text())
    else:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            data = json.load(resp)

    voices = []
    for key, v in sorted(data.items()):
        lang = v["language"]
        files = [[path, f["size_bytes"], f["md5_digest"]]
                 for path, f in sorted(v["files"].items()) if path.endswith((".onnx", ".onnx.json"))]
        voices.append({
            "key": key,
            "lang": lang["code"],
            "language": f"{lang['name_english']} ({lang['country_english']})",
            "quality": v["quality"],
            "speakers": v["num_speakers"],
            "files": files,
        })

    # One voice per line: compact, and upstream additions diff cleanly.
    lines = ",\n".join(json.dumps(v, ensure_ascii=False) for v in voices)
    OUT.write_text(f"[\n{lines}\n]\n", encoding="utf-8")
    print(f"Wrote {len(voices)} voices to {OUT}")


if __name__ == "__main__":
    main()
