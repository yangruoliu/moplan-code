#!/usr/bin/env python
"""Filter cached SAM2 automatic masks (the "k16" instance-mask inputs of Stage-1).

The release setting is --max-kept 16 --union-cap 1.0 (the defaults): every SAM2 instance that passes the area and
border tests is kept, up to 16, largest first.  An earlier variant kept only the 8 largest instances under a union
cap of one half, which dropped small objects that are about to move (a held cube, a ball).

The tests remove SAM2 segments that have no analogue among the training instance masks (which cover the handful of
bodies that move): background plates and table / wall / floor spans.  Areas are measured inside the image's letterbox
content box on the 160 grid (the arithmetic of cache_sam2_masks.letterbox_square), except that inputs whose box is
within one grid row of the 16:9 band rows 35..125 (16:9, 1280x704 and 7:4 frames: every benchmark canvas but Morpheus)
use that band, as the published runs did.  The size comes from --manifest (width/height columns, else the
first_frame_path image); without it the 16:9 band is used for every input.  An instance is dropped when it covers more
than --area-max of the box or more than --border-max of the box's border ring.  When nothing survives, the two
smallest instances are kept.  (Before 2026-11-15 the 16:9 band was used for every input, so on clearly non-16:9
first frames - Morpheus 1280x1024 - objects outside the band were dropped; the paper's Morpheus rows were re-run
with this rule.)

Payload schema is preserved exactly (instance_masks [K,160,160] half, instance_channels [2,160,160] half), and
instance_channels is recomputed with motion_jepa.flow_dit_data.build_instance_inputs -- the same function
cache_sam2_masks.py uses -- rather than being copied forward, because dropping instances turns former
instance-instance boundaries into instance-background boundaries and shrinks the union.

usage: filter_sam2_masks_v2.py --in-dir SAM2_RAW --out-dir K16 [--max-kept 16 --union-cap 1.0]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_jepa.flow_dit_data import build_instance_inputs  # noqa: E402

SIDE = 160
ROW0, ROW1 = 35, 125          # letterbox content rows of a 1280x720 source
CONTENT_H = ROW1 - ROW0       # 90
CONTENT_W = SIDE              # 160
CONTENT_AREA = CONTENT_H * CONTENT_W          # 14400
RING = 2 * CONTENT_W + 2 * CONTENT_H - 4      # 496

AREA_FRAC_MAX = 0.20
BORDER_COVER_MAX = 0.35
MAX_KEPT = 8
UNION_COVER_CAP = 0.50
NUM_SLOTS = 16


def ring_mask() -> torch.Tensor:
    r = torch.zeros(CONTENT_H, CONTENT_W, dtype=torch.bool)
    r[0, :] = True
    r[-1, :] = True
    r[:, 0] = True
    r[:, -1] = True
    assert int(r.sum()) == RING, int(r.sum())
    return r


RING_M = ring_mask()
LEGACY_BOX = (ROW0, ROW1, 0, SIDE)   # the 16:9 band


def content_box(height: int, width: int, side: int = SIDE) -> tuple[int, int, int, int]:
    """letterbox content box (row0, row1, col0, col1) of a height x width image on the side x side grid"""
    scale = side / max(height, width)
    h, w = round(height * scale), round(width * scale)
    top, left = (side - h) // 2, (side - w) // 2
    return top, top + h, left, left + w


def filter_box(height: int, width: int, side: int = SIDE) -> tuple[int, int, int, int]:
    """the box the area / border tests use: the content box, or the 16:9 band when the box is within one row of it"""
    b = content_box(height, width, side)
    if b[2] == 0 and b[3] == side and abs(b[0] - ROW0) <= 1 and abs(b[1] - ROW1) <= 1:
        return LEGACY_BOX
    return b


def filter_sample(masks: torch.Tensor, box: tuple[int, int, int, int] = LEGACY_BOX):
    """masks [K,160,160] float binary -> (kept_indices, stats); area and border are measured inside box."""
    r0, r1, c0, c1 = box
    CONTENT_H, CONTENT_W = r1 - r0, c1 - c0
    CONTENT_AREA = CONTENT_H * CONTENT_W
    ring = torch.zeros(CONTENT_H, CONTENT_W, dtype=torch.bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    RING = int(ring.sum())
    content = masks[:, r0:r1, c0:c1] > 0.5            # [K,h,w] inside the box
    areas = content.flatten(1).sum(1)                  # px in the content region
    keep_any = areas > 0
    area_frac = areas / CONTENT_AREA
    border_cover = (content & ring[None]).flatten(1).sum(1) / RING

    idx_all = [i for i in range(masks.shape[0]) if keep_any[i]]
    survivors = [
        i for i in idx_all
        if area_frac[i] <= AREA_FRAC_MAX and border_cover[i] <= BORDER_COVER_MAX
    ]
    fallback = False
    if not survivors:
        # nothing passes: keep the 2 smallest original instances so the sample
        # is not empty (the model has never seen an all-background condition).
        fallback = True
        survivors = sorted(idx_all, key=lambda i: float(areas[i]))[:2]

    # largest first -- the same order build_instance_inputs uses.
    survivors.sort(key=lambda i: -float(areas[i]))

    kept: list[int] = []
    union = torch.zeros(CONTENT_H, CONTENT_W, dtype=torch.bool)
    for i in survivors:
        if len(kept) >= MAX_KEPT:
            break
        cand = union | content[i]
        cov = float(cand.sum()) / CONTENT_AREA
        if kept and cov > UNION_COVER_CAP:
            continue
        kept.append(i)
        union = cand

    stats = {
        "n_raw": len(idx_all),
        "n_kept": len(kept),
        "cov_raw": float((content[idx_all].any(0) if idx_all else
                          torch.zeros(CONTENT_H, CONTENT_W, dtype=torch.bool)).sum()) / CONTENT_AREA,
        "cov_kept": float(union.sum()) / CONTENT_AREA,
        "fallback": fallback,
        "n_drop_area": sum(1 for i in idx_all if area_frac[i] > AREA_FRAC_MAX),
        "n_drop_border": sum(1 for i in idx_all if area_frac[i] <= AREA_FRAC_MAX
                             and border_cover[i] > BORDER_COVER_MAX),
    }
    return kept, stats


def configure(max_kept: int = 16, union_cap: float = 1.0, area_max: float = 0.20, border_max: float = 0.35) -> None:
    """Set the filter caps (module globals read by filter_sample); the defaults are the CLI defaults (k16)."""
    global MAX_KEPT, UNION_COVER_CAP, AREA_FRAC_MAX, BORDER_COVER_MAX
    MAX_KEPT, UNION_COVER_CAP, AREA_FRAC_MAX, BORDER_COVER_MAX = max_kept, union_cap, area_max, border_max


def filter_payload(payload: dict, box: tuple[int, int, int, int] | None = None) -> tuple[dict, dict]:
    """One raw cache_sam2_masks.py payload -> (filtered payload, stats) under the current caps (see configure); box: see
    filter_box (None = the 16:9 band)."""
    box = tuple(box) if box is not None else LEGACY_BOX
    masks = payload["instance_masks"].float()
    assert masks.shape[-2:] == (SIDE, SIDE), masks.shape
    kept, st = filter_sample(masks, box)

    # fresh label map on the retained set only, then the exact training builder
    ids = torch.zeros(SIDE, SIDE, dtype=torch.long)
    for slot, i in enumerate(kept):
        ids[masks[i] > 0.5] = slot + 1
    new_masks, _, new_channels = build_instance_inputs(ids[None], NUM_SLOTS)

    # sanity: union channel must equal the retained union
    assert torch.equal((new_masks.sum(0) > 0).float(), new_channels[0])

    out = dict(payload)
    out["instance_masks"] = new_masks.half()
    out["instance_channels"] = new_channels.half()
    out["num_instances"] = int((new_masks.flatten(1).sum(1) > 0).sum())
    out["source"] = str(payload.get("source", "")) + "+filter_v2"
    out["filter"] = {
        "content_rows": [int(box[0]), int(box[1])], "content_box": [int(v) for v in box], "area_frac_max": AREA_FRAC_MAX,
        "border_cover_max": BORDER_COVER_MAX, "max_instances_kept": MAX_KEPT,
        "union_coverage_cap": UNION_COVER_CAP,
        "kept_source_slots": kept, "fallback": st["fallback"],
    }
    return out, st


def manifest_sizes(path: str) -> dict[str, tuple[int, int]]:
    """sample_id -> (height, width) from a manifest csv (width/height columns, else the first_frame_path image)"""
    import csv
    from PIL import Image
    out = {}
    for row in csv.DictReader(open(path, newline="")):
        if row.get("width") and row.get("height"):
            out[row["sample_id"]] = (int(row["height"]), int(row["width"]))
        else:
            with Image.open(row["first_frame_path"]) as im:
                out[row["sample_id"]] = (im.height, im.width)
    return out


def main(argv: list[str] | None = None) -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="raw SAM2 masks from cache_sam2_masks.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-kept", type=int, default=16)
    ap.add_argument("--union-cap", type=float, default=1.0)
    ap.add_argument("--area-max", type=float, default=0.20)
    ap.add_argument("--border-max", type=float, default=0.35)
    ap.add_argument("--tag", default="")
    ap.add_argument("--manifest", default="", help="csv with sample_id and width,height (or first_frame_path): each sample is "
                    "measured on its own content box (see filter_box); without it the 16:9 band is used")
    args = ap.parse_args(argv)
    configure(args.max_kept, args.union_cap, args.area_max, args.border_max)

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.suffix == ".pt")
    sizes = manifest_sizes(args.manifest) if args.manifest else {}
    print(f"{len(files)} samples", flush=True)

    cov_raw, cov_new, n_raw, n_new, n_fb = [], [], [], [], 0
    drop_area = drop_border = tot_inst = 0
    for path in files:
        payload = torch.load(path, map_location="cpu")
        out, st = filter_payload(payload, filter_box(*sizes[path.stem]) if args.manifest else None)
        tmp = out_dir / f"{path.stem}.tmp-{os.getpid()}"
        torch.save(out, tmp)
        tmp.replace(out_dir / path.name)

        cov_raw.append(st["cov_raw"]); cov_new.append(st["cov_kept"])
        n_raw.append(st["n_raw"]); n_new.append(out["num_instances"])
        n_fb += int(st["fallback"])
        drop_area += st["n_drop_area"]; drop_border += st["n_drop_border"]
        tot_inst += st["n_raw"]

    a = np.array
    cr, cn, nr, nn = a(cov_raw), a(cov_new), a(n_raw, float), a(n_new, float)

    def q(x):
        return dict(mean=float(x.mean()), median=float(np.median(x)),
                    p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                    min=float(x.min()), max=float(x.max()))

    print("\n=== union coverage of the content region ===")
    print("RAW  SAM2 :", {k: round(v, 4) for k, v in q(cr).items()})
    print("FILTERED  :", {k: round(v, 4) for k, v in q(cn).items()})
    print("\n=== instance count ===")
    print("RAW  SAM2 :", {k: round(v, 3) for k, v in q(nr).items()})
    print("FILTERED  :", {k: round(v, 3) for k, v in q(nn).items()})
    print(f"\ninstances dropped: area_frac>{AREA_FRAC_MAX}: {drop_area}/{tot_inst} "
          f"({100*drop_area/max(tot_inst, 1):.1f}%), border_cover>{BORDER_COVER_MAX} (extra): "
          f"{drop_border}/{tot_inst} ({100*drop_border/max(tot_inst, 1):.1f}%)")
    print(f"samples hitting the empty-survivor fallback: {n_fb}/{len(files)}")
    print(f"samples with coverage > 0.40: raw {int((cr>0.40).sum())}/{len(cr)} "
          f"-> filtered {int((cn>0.40).sum())}/{len(cn)}")
    print(f"samples with coverage > 0.50: raw {int((cr>0.50).sum())}/{len(cr)} "
          f"-> filtered {int((cn>0.50).sum())}/{len(cn)}")
    print(f"\nwrote {len(files)} -> {out_dir}")


if __name__ == "__main__":
    main()
