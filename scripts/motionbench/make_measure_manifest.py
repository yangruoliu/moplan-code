#!/usr/bin/env python3
"""Manifest for measure_clips.py from a generation root laid out as <root>/<model>/s<seed>/<sid>.mp4 (+ <sid>.json
with fps written next to each clip, e.g. by scripts/s2eval/miq_export.py)."""
import argparse, glob, json, os, time
from pathlib import Path


def fps_for(mp4: Path, default: float) -> float:
    js = mp4.with_suffix(".json")
    if js.is_file():
        try:
            return float(json.load(open(js)).get("fps", default))
        except Exception:
            pass
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--path-prefix", default="", help="replace the local root by this prefix (manifest consumed on another host)")
    ap.add_argument("--default-fps", type=float, default=16.0)
    ap.add_argument("--min-age-s", type=float, default=180.0,
                    help="skip clips modified less than this many seconds ago (a generator or a relay may still be writing them)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    now = time.time()
    root = Path(a.gen_root)
    rows = []
    for mp4 in sorted(list(root.glob("*/s*/*.mp4")) + list(root.glob("*/s*/*.npy"))):
        model, seed_dir = mp4.parts[-3], mp4.parts[-2]
        seed = int(seed_dir[1:])
        sid = mp4.stem
        if sid.endswith("_raw") or sid.endswith(".tmp"):
            continue
        if now - mp4.stat().st_mtime < a.min_age_s:
            continue
        path = str(mp4) if not a.path_prefix else str(mp4).replace(str(root), a.path_prefix, 1)
        base = sid.split("-")[0]          # item sid; any "-<variant>" suffix (-ex, -cf, -nt, -aL, ...) is a generation variant
        rows.append(dict(sid=sid, item_sid=base, arm=model, seed=seed, path=path, fps=fps_for(mp4, a.default_fps), canvas=True))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in rows:
        by[(r["arm"], r["seed"])] = by.get((r["arm"], r["seed"]), 0) + 1
    print(f"wrote {len(rows)} rows: " + ", ".join(f"{k[0]}/s{k[1]}={v}" for k, v in sorted(by.items())))


if __name__ == "__main__":
    main()
