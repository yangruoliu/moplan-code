#!/usr/bin/env python3
"""Step 2 of the MoProbe SSv2 build: stream the Something-Else annotation parts and keep only the
candidate validation clips. Writes boxes_val.json:
  {vid: {"n_frames": F, "placeholders": [...], "frames": {frame_idx(1-based): [[cat, x1,y1,x2,y2], ...]}}}
cat is the Something-Else standard_category: "hand" or "0000"/"0001"/... (placeholder index).
Streaming (ijson) because each part is ~1 GB of JSON.
"""
import argparse, csv, json, time
from pathlib import Path

import ijson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    want = {r["id"] for r in csv.DictReader(open(args.candidates))}
    out, seen = {}, 0
    t0 = time.time()
    for part in args.parts:
        with open(part, "rb") as f:
            for vid, frames in ijson.kvitems(f, ""):
                seen += 1
                if vid not in want:
                    continue
                fr = {}
                ph = None
                for fd in frames:
                    idx = int(Path(fd["name"]).stem)
                    ph = ph or fd.get("gt_placeholders")
                    fr[idx] = [[l.get("standard_category", l.get("category")),
                                round(float(l["box2d"]["x1"]), 1), round(float(l["box2d"]["y1"]), 1),
                                round(float(l["box2d"]["x2"]), 1), round(float(l["box2d"]["y2"]), 1)]
                               for l in fd.get("labels", [])]
                out[vid] = dict(n_frames=max(fr) if fr else 0, placeholders=ph or [], frames=fr)
        print(f"{part}: cumulative videos seen {seen}, kept {len(out)}, {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(args.out, "w"))
    print(f"wrote {len(out)} candidate clips with boxes to {args.out}")


if __name__ == "__main__":
    main()
