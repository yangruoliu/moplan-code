#!/usr/bin/env python3
"""Selection for the SSv2 TRAIN real pool: camera gate (relaxed tier), t0 / onset from the Something-Else
boxes, roles from the benchmark's SSv2 template table, then family quotas.  CPU only.

Imports the gate / box / onset logic of scripts/motionbench/ssv2_select.py (not modified; its main() is val-only) and
applies it to the train candidates of ssv2_train_candidates.py:
  --candidates   candidates_v0.csv (split == official_train asserted)
  --camera-stats camera_stats.csv from scripts/motionbench/ssv2_camera_prefilter.py --split official_train
  --boxes        boxes.json from scripts/motionbench/ssv2_boxes.py (filtered to the candidate ids)
writes <out-dir>/items_all.jsonl       every clip that passes the gates (same fields as the benchmark's items_draft.jsonl,
                                       plus family / pool / rank / label / noun / caption)
       <out-dir>/items_selected.jsonl  the quota-balanced pool (core first, then reserve, by rank) fed to ssv2_clips.py
       <out-dir>/select_summary.json   drop reasons, per-family gate pass rates (strict / relaxed), box coverage,
                                       hand-at-t0 share, selected counts

t0 is 3 source frames (0.25 s) before the onset (box-centre speed >= 2 % of width per frame for 2 frames) for event
templates, and 3 frames before the first hand-object contact for static controls (as in the benchmark).  The clip must keep
at least --min-sec-after seconds after t0.  Static controls must make up at least --min-static-share of the selection:
when they are short the other families are scaled down proportionally.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOTIONBENCH = HERE.parent / "motionbench"
sys.path.insert(0, str(MOTIONBENCH))
_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True          # import only: leave no __pycache__ in scripts/motionbench
import ssv2_select as S  # noqa: E402  (camera tier, box series, onset, contact; constants)
from ssv2_instructions import TABLE, roles_for  # noqa: E402
sys.dont_write_bytecode = _write_bytecode

sys.path.insert(0, str(HERE))
from ssv2_train_candidates import DEFAULT_QUOTAS, FAMILIES, STATIC_FAMILY, normalise_quotas  # noqa: E402

SPLIT = "official_train"
# Templates of the train whitelist that the benchmark's table does not cover (it was written for the benchmark subset).
ROLE_OVERRIDES = {
    101: dict(actor=0, patient=1, dir="any", static_caption=False, event="push_with_tool", rigid=True),   # Pushing A with B
}


def roles(template_id: int, n_placeholders: int) -> dict:
    if str(template_id) in TABLE:
        return roles_for(template_id, n_placeholders)
    if template_id in ROLE_OVERRIDES:
        return dict(ROLE_OVERRIDES[template_id])
    raise KeyError(f"template {template_id} has no roles entry")


def select_item(r: dict, st: dict | None, b: dict | None, min_sec_after: float) -> tuple[dict | None, str | None]:
    """One candidate row -> (item, None) or (None, drop_reason).  Mirrors the main loop of scripts/motionbench/ssv2_select.py."""
    tid = int(r["template_id"])
    if st is None or int(st["decode_ok"]) != 1:
        return None, "no_stats"
    tier = S.cam_tier(st)
    if tier is None:
        return None, "camera"
    if b is None:
        return None, "no_boxes"
    n, fps, w, h = int(st["n_frames"]), float(st["fps"]), int(st["w"]), int(st["h"])
    if abs(int(b["n_frames"]) - n) > 3:
        return None, "frame_count_mismatch"
    ph = json.loads(r["placeholders"])
    role = roles(tid, len(ph))
    cats = {c: S.box_series(b["frames"], c, n) for c in ("0000", "0001", "hand")}
    if not any(x is not None for x in cats["0000"]):
        return None, "no_object_box"
    actor, patient = role["actor"], role["patient"]
    actor_cat = "hand" if actor == "hand" else (f"{actor:04d}" if isinstance(actor, int) else None)
    patient_cat = f"{patient:04d}" if isinstance(patient, int) else None
    mover_cat = actor_cat if actor_cat and actor_cat != "hand" else (patient_cat or "0000")
    flags = []
    if role["static_caption"]:
        cf = S.contact_frame(cats["hand"], cats[patient_cat or "0000"])
        onset = cf
        t0 = max(0, (cf if cf is not None else S.PRE_ONSET_FRAMES) - S.PRE_ONSET_FRAMES)
        if cf is None:
            flags.append("no_hand_contact")
    else:
        onset = S.onset_from_boxes(cats[mover_cat], w)
        if onset is None:
            return None, "no_onset"
        t0 = max(0, onset - S.PRE_ONSET_FRAMES)
    sec_after = (n - 1 - t0) / fps
    if sec_after < min_sec_after:
        return None, "too_short_after_t0"
    bt0 = {c: cats[c][t0] for c in cats if cats[c][t0] is not None}
    if mover_cat not in bt0:
        return None, "mover_not_visible_at_t0"
    mb = bt0[mover_cat]
    if mb[0] < S.EDGE_MARGIN_PX or mb[1] < S.EDGE_MARGIN_PX or mb[2] > w - S.EDGE_MARGIN_PX or mb[3] > h - S.EDGE_MARGIN_PX:
        flags.append("mover_touches_edge_at_t0")
    after = cats[mover_cat][t0:t0 + int(1.5 * fps)]
    if sum(x is not None for x in after) < S.VIS_FRAC_AFTER * len(after):
        return None, "mover_not_visible_after_t0"
    dx = dy = None
    end = min(n - 1, t0 + int(2.0 * fps))
    s0 = cats[mover_cat][t0]
    later = [cats[mover_cat][i] for i in range(end, t0, -1) if cats[mover_cat][i] is not None]
    if s0 is not None and later:
        (xa, ya), (xb, yb) = S.centre(s0), S.centre(later[0])
        dx, dy = (xb - xa) / w * 100.0, (yb - ya) / h * 100.0
    d = role["dir"]
    if dx is not None:
        if d == "right" and dx < 2.0:
            return None, "lr_mismatch"
        if d == "left" and dx > -2.0:
            return None, "lr_mismatch"
        if d == "up" and dy > -2.0:
            flags.append("up_mismatch")
        if d == "down" and dy < 2.0 and not role["static_caption"]:
            flags.append("down_mismatch")
    if role["static_caption"] and dx is not None and math.hypot(dx, dy) > 5.0:
        flags.append("static_but_moved")
    item = dict(id=r["id"], template_id=tid, template=r["template"], family=r["family"], pool=r["pool"], rank=int(r["rank"]),
                groups=[r["family"]], placeholders=ph, noun=r.get("noun", ""), label=r["label"], caption=r["label"],
                n_frames=n, fps=fps, width=w, height=h, dur_s=round(n / fps, 3), cam_tier=tier, t0_frame=int(t0),
                onset_frame=(None if onset is None else int(onset)), sec_after_t0=round(sec_after, 3),
                actor_cat=actor_cat, patient_cat=patient_cat, mover_cat=mover_cat, hand_at_t0="hand" in bt0,
                boxes_t0={c: [round(v, 1) for v in bb] for c, bb in bt0.items()},
                dx_pct_w=(None if dx is None else round(dx, 1)), dy_pct_h=(None if dy is None else round(dy, 1)),
                flags=flags, static_caption=role["static_caption"], event=role["event"], rigid=role["rigid"],
                direction_expected=d)
    return item, None


def template_round_robin(items: list[dict]) -> list[dict]:
    """One family's items in pick order: round-robin over its templates (ascending id), each template's queue core
    before reserve, by rank -- so a capped family is not filled by its largest template."""
    queues = collections.defaultdict(list)
    for it in sorted(items, key=lambda it: (it["pool"] != "core", it["rank"])):
        queues[it.get("template_id", 0)].append(it)
    order, depth, keys = [], 0, sorted(queues)
    while len(order) < len(items):
        for k in keys:
            if depth < len(queues[k]):
                order.append(queues[k][depth])
        depth += 1
    return order


def apply_quotas(items: list[dict], quotas: dict, target: int, min_static_share: float) -> list[dict]:
    """Per family min(quota * target, passing), picked round-robin over the family's templates (core before reserve).
    If static controls end up below `min_static_share`, the other families are scaled down proportionally so that the
    share is met.  target 0: no cap."""
    by_family = collections.defaultdict(list)
    for it in items:
        by_family[it["family"]].append(it)
    for f in list(by_family):
        by_family[f] = template_round_robin(by_family[f])
    want = {f: (len(by_family[f]) if target <= 0 else min(int(round(quotas.get(f, 0.0) * target)), len(by_family[f])))
            for f in by_family}
    n_static = want.get(STATIC_FAMILY, 0)
    others = sum(v for f, v in want.items() if f != STATIC_FAMILY)
    if min_static_share > 0 and others and n_static < min_static_share * (n_static + others):
        allowed = int(n_static * (1.0 - min_static_share) / min_static_share)
        scale = allowed / others
        for f in want:
            if f != STATIC_FAMILY:
                want[f] = int(want[f] * scale)
    picked = []
    for f in FAMILIES:
        picked.extend(by_family.get(f, [])[: want.get(f, 0)])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--camera-stats", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-sec-after", type=float, default=2.0)
    ap.add_argument("--target", default="0", help="selection size the quotas are applied to: an integer, 0 (keep all "
                    "passing) or 'auto' (= the number of clips passing the gates: every family capped at its quota share)")
    ap.add_argument("--min-static-share", type=float, default=0.20)
    ap.add_argument("--quotas", default=",".join(f"{k}={v}" for k, v in DEFAULT_QUOTAS.items()))
    args = ap.parse_args()
    quotas = normalise_quotas({k: float(v) for k, v in (kv.split("=") for kv in args.quotas.split(","))}, args.min_static_share)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cand = list(csv.DictReader(open(args.candidates)))
    assert cand and all(r["split"] == SPLIT for r in cand), "candidates must be the official train split only"
    stats = {r["id"]: r for r in csv.DictReader(open(args.camera_stats))}
    boxes = json.load(open(args.boxes))

    kept, drop = [], collections.Counter()
    per_family = {f: collections.Counter() for f in FAMILIES}
    for r in cand:
        f = r["family"]
        st = stats.get(r["id"])
        pf = per_family[f]
        pf["candidates"] += 1
        pf[r["pool"]] += 1
        if st is not None and int(st["decode_ok"]) == 1:
            pf["decoded"] += 1
            tier = S.cam_tier(st)
            if tier == "strict":
                pf["cam_strict"] += 1
            if tier is not None:
                pf["cam_relaxed_or_strict"] += 1
        if r["id"] in boxes:
            pf["has_boxes"] += 1
        item, reason = select_item(r, st, boxes.get(r["id"]), args.min_sec_after)
        if item is None:
            drop[reason] += 1
            pf["drop:" + reason] += 1
            continue
        pf["passed"] += 1
        pf["passed_" + item["cam_tier"]] += 1
        if item["hand_at_t0"]:
            pf["hand_at_t0"] += 1
        if item["flags"]:
            pf["flagged"] += 1
        kept.append(item)
    target = len(kept) if args.target == "auto" else int(args.target)
    selected = apply_quotas(kept, quotas, target, args.min_static_share)
    for it in selected:
        per_family[it["family"]]["selected"] += 1
    with open(out / "items_all.jsonl", "w") as fh:
        for it in kept:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out / "items_selected.jsonl", "w") as fh:
        for it in selected:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_static = sum(1 for it in selected if it["family"] == STATIC_FAMILY)
    summary = dict(candidates=len(cand), passed=len(kept), selected=len(selected), dropped=dict(drop),
                   target=target, target_arg=args.target, min_sec_after=args.min_sec_after, min_static_share=args.min_static_share,
                   static_share_selected=round(n_static / max(1, len(selected)), 4),
                   by_cam_tier=dict(collections.Counter(it["cam_tier"] for it in kept)),
                   by_pool_selected=dict(collections.Counter(it["pool"] for it in selected)),
                   hand_at_t0=sum(1 for it in kept if it["hand_at_t0"]), flagged=sum(1 for it in kept if it["flags"]),
                   flags=dict(collections.Counter(fl for it in kept for fl in it["flags"])),
                   per_family={f: dict(c) for f, c in per_family.items()},
                   by_template_selected=dict(sorted(collections.Counter(it["template_id"] for it in selected).items())))
    json.dump(summary, open(out / "select_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_family", "by_template_selected")}, indent=1))
    print(f"  {'family':12s} {'cand':>6s} {'strict':>7s} {'relaxed+':>8s} {'boxes':>6s} {'passed':>6s} {'hand@t0':>7s} {'selected':>8s}")
    for f in FAMILIES:
        c = per_family[f]
        n = max(1, c["candidates"])
        print(f"  {f:12s} {c['candidates']:6d} {c['cam_strict'] / n:7.3f} {c['cam_relaxed_or_strict'] / n:8.3f} {c['has_boxes'] / n:6.3f} "
              f"{c['passed']:6d} {c['hand_at_t0'] / max(1, c['passed']):7.3f} {c['selected']:8d}")


if __name__ == "__main__":
    main()
