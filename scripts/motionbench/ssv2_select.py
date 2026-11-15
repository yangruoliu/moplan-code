#!/usr/bin/env python3
"""Steps 3-4 of the MoProbe SSv2 build: camera gate, onset / t0 detection from Something-Else boxes, role
assignment from the template table, label sanity checks. CPU only.

Writes items_draft.jsonl (one line per surviving clip) and select_summary.json.
  id, template_id, template, groups, placeholders, n_frames, fps, dur_s,
  cam_tier (strict|relaxed), t0_frame (0-based video frame used as the conditioning frame),
  onset_frame, sec_after_t0, actor_cat / patient_cat ("0000", "0001", "hand" or null),
  hand_at_t0, boxes_t0 {cat: [x1,y1,x2,y2]} in native px, dx_sign, dy_sign, flags [...],
  instruction_neutral, instruction_explicit, cf {kind, text}, static_caption, event, rigid, direction_expected
Nothing here is final: SAM2 masks, measured references and the GPU camera check come later.
"""
import argparse, csv, json, math, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from ssv2_instructions import make_instruction, roles_for, counterfactual_for  # noqa: E402

ONSET_SPEED_PCT_W = 2.0      # box-centre speed (% of width per frame) that starts an event
ONSET_CONSEC = 2
PRE_ONSET_FRAMES = 3         # t0 = onset - 3 frames (0.25 s at 12 fps)
MIN_SEC_AFTER_T0 = {"T1": 2.5, "T2": 2.5, "T3": 2.5, "T4": 2.0}
VIS_FRAC_AFTER = 0.5         # actor box present in >= this share of the 1.5 s after t0
EDGE_MARGIN_PX = 2


PRECONTACT_TEMPLATES = set() # templates whose hand may already hold the target object in the first frame:
                             # t0 = the last frame before the hand first touches either object box
CF_ROOM = 0.0                # an object at the frame edge cannot move further that way:
                             # left / right / up / down templates need this share of the frame free on the counterfactual side
ANY_CAMERA_TEMPLATES = set() # role-pair templates with no camera gate: these items serve track A only
                             # (camera-compensated displacement), never track C (which needs a still real camera)
LOOSE_TEMPLATES = set()      # templates admitted with a looser camera gate (role pairs are scarce); the built item
                             # still has to pass the reference-camera check (<= 16 px max, <= 8 px median)


def cam_tier(r, template_id=None):
    m, p, c = float(r["bcam_med_pct_w_per_s"]), float(r["bcam_peak_pct_w"]), float(r["bcam_cum_pct_w"])
    full = float(r["cam_cum_pct_w"])
    if m < 1.0 and p < 2.5 and c < 6.0 and full < 25.0:
        return "strict"
    if m < 2.0 and p < 4.0 and c < 10.0 and full < 40.0:
        return "relaxed"
    if template_id in ANY_CAMERA_TEMPLATES:
        return "any"
    if template_id in LOOSE_TEMPLATES and m < 3.0 and p < 6.0 and c < 16.0 and full < 60.0:
        return "loose"
    return None


def box_series(frames: dict, cat: str, n: int):
    """[n] list of boxes (or None) for the category, 1-based annotation index → 0-based video frame."""
    out = [None] * n
    for idx, labels in frames.items():
        i = int(idx) - 1
        if 0 <= i < n:
            for l in labels:
                if l[0] == cat:
                    out[i] = l[1:]
                    break
    # linear interpolation of short gaps (<= 3 frames)
    last = None
    for i in range(n):
        if out[i] is not None:
            if last is not None and 1 < i - last <= 4:
                for j in range(last + 1, i):
                    w = (j - last) / (i - last)
                    out[j] = [(1 - w) * a + w * b for a, b in zip(out[last], out[i])]
            last = i
    return out


def centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def onset_from_boxes(series, width):
    """first frame index where the centre speed exceeds the threshold for ONSET_CONSEC consecutive frames."""
    run = 0
    for i in range(1, len(series)):
        if series[i] is None or series[i - 1] is None:
            run = 0
            continue
        (x0, y0), (x1, y1) = centre(series[i - 1]), centre(series[i])
        speed = math.hypot(x1 - x0, y1 - y0) / width * 100.0
        run = run + 1 if speed >= ONSET_SPEED_PCT_W else 0
        if run >= ONSET_CONSEC:
            return i - ONSET_CONSEC + 1
    return None


def contact_frame(hand, obj, pad=2.0):
    for i, (h, o) in enumerate(zip(hand, obj)):
        if h is None or o is None:
            continue
        if not (h[2] + pad < o[0] or o[2] + pad < h[0] or h[3] + pad < o[1] or o[3] + pad < h[1]):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--camera-stats", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--loose-templates", default="", help="comma-separated template ids admitted with the looser camera gate")
    ap.add_argument("--templates", default="", help="only these template ids (comma-separated)")
    ap.add_argument("--t0-precontact", default="", help="template ids whose t0 is the last frame before the hand touches an object")
    ap.add_argument("--camera-any", default="", help="template ids with no camera gate (track A only)")
    ap.add_argument("--cf-room", type=float, default=0.0, help="min free share of the frame on the counterfactual side (direction templates)")
    args = ap.parse_args()
    global CF_ROOM
    ANY_CAMERA_TEMPLATES.update(int(x) for x in args.camera_any.split(",") if x.strip())
    PRECONTACT_TEMPLATES.update(int(x) for x in args.t0_precontact.split(",") if x.strip()); CF_ROOM = args.cf_room
    LOOSE_TEMPLATES.update(int(x) for x in args.loose_templates.split(",") if x.strip())
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    only = {int(x) for x in args.templates.split(",") if x.strip()}
    cand = {r["id"]: r for r in csv.DictReader(open(args.candidates)) if r["split"] == "official_val" and (not only or int(r["template_id"]) in only)}
    stats = {r["id"]: r for r in csv.DictReader(open(args.camera_stats))}
    boxes = json.load(open(args.boxes))
    kept, drop = [], {}

    def dropped(reason):
        drop[reason] = drop.get(reason, 0) + 1

    for vid, r in cand.items():
        tid = r["template_id"]
        groups = r["groups"].split("|")
        track = "T4" if any(g.startswith("t4") for g in groups) else "T3" if "t3_relational" in groups else "T1"
        st = stats.get(vid)
        if st is None or int(st["decode_ok"]) != 1:
            dropped("no_stats"); continue
        tier = cam_tier(st, int(tid))
        if tier is None:
            dropped("camera"); continue
        b = boxes.get(vid)
        if b is None:
            dropped("no_boxes"); continue
        n, fps, w, h = int(st["n_frames"]), float(st["fps"]), int(st["w"]), int(st["h"])
        if abs(b["n_frames"] - n) > 3:
            dropped("frame_count_mismatch"); continue
        roles = roles_for(tid, len(json.loads(r["placeholders"])))
        ph = json.loads(r["placeholders"])
        cats = {"0000": box_series(b["frames"], "0000", n), "0001": box_series(b["frames"], "0001", n),
                "hand": box_series(b["frames"], "hand", n)}
        if not any(x is not None for x in cats["0000"]):
            dropped("no_object_box"); continue
        actor = roles["actor"]
        actor_cat = "hand" if actor == "hand" else (f"{actor:04d}" if isinstance(actor, int) else None)
        patient = roles["patient"]
        patient_cat = f"{patient:04d}" if isinstance(patient, int) else None
        mover_cat = actor_cat if actor_cat and actor_cat != "hand" else (patient_cat or "0000")
        flags = []
        if int(tid) in PRECONTACT_TEMPLATES:
            objs = [c for c in ("0000", "0001") if any(x is not None for x in cats[c])]
            first_touch = None
            for c in objs:
                f = contact_frame(cats["hand"], cats[c], pad=4.0)
                if f is not None and (first_touch is None or f < first_touch):
                    first_touch = f
            if first_touch is None:
                dropped("precontact_no_touch"); continue
            free = [i for i in range(first_touch - 1, -1, -1) if all(cats[c][i] is not None for c in objs)]
            if not free:
                dropped("precontact_touch_from_start"); continue
            t0 = free[0]; onset = first_touch
        elif roles["static_caption"]:
            cf = contact_frame(cats["hand"], cats[patient_cat or "0000"])
            onset = cf
            t0 = max(0, (cf if cf is not None else PRE_ONSET_FRAMES) - PRE_ONSET_FRAMES)
            if cf is None:
                flags.append("no_hand_contact")
        else:
            onset = onset_from_boxes(cats[mover_cat], w)
            if onset is None:
                dropped("no_onset"); continue
            t0 = max(0, onset - PRE_ONSET_FRAMES)
        sec_after = (n - 1 - t0) / fps
        if sec_after < MIN_SEC_AFTER_T0[track]:
            dropped("too_short_after_t0"); continue
        bt0 = {c: cats[c][t0] for c in cats if cats[c][t0] is not None}
        if mover_cat not in bt0:
            dropped("mover_not_visible_at_t0"); continue
        mb = bt0[mover_cat]
        if CF_ROOM > 0 and roles["dir"] in ("left", "right", "up", "down"):
            room = {"right": mb[0] / w, "left": (w - mb[2]) / w, "up": (h - mb[3]) / h, "down": mb[1] / h}[roles["dir"]]
            if room < CF_ROOM:
                dropped("cf_room"); continue
        if mb[0] < EDGE_MARGIN_PX or mb[1] < EDGE_MARGIN_PX or mb[2] > w - EDGE_MARGIN_PX or mb[3] > h - EDGE_MARGIN_PX:
            flags.append("mover_touches_edge_at_t0")
        after = cats[mover_cat][t0:t0 + int(1.5 * fps)]
        if sum(x is not None for x in after) < VIS_FRAC_AFTER * len(after):
            dropped("mover_not_visible_after_t0"); continue
        # displacement sign over the event window (for L/R / up / down sanity)
        dx = dy = None
        end = min(n - 1, t0 + int(2.0 * fps))
        s0 = cats[mover_cat][t0]
        later = [cats[mover_cat][i] for i in range(end, t0, -1) if cats[mover_cat][i] is not None]
        if s0 is not None and later:
            (xa, ya), (xb, yb) = centre(s0), centre(later[0])
            dx, dy = (xb - xa) / w * 100.0, (yb - ya) / h * 100.0
        d = roles["dir"]
        if dx is not None:
            if d == "right" and dx < 2.0:
                dropped("lr_mismatch"); continue
            if d == "left" and dx > -2.0:
                dropped("lr_mismatch"); continue
            if d == "up" and dy > -2.0:
                flags.append("up_mismatch")
            if d == "down" and dy < 2.0 and not roles["static_caption"]:
                flags.append("down_mismatch")
        if roles["static_caption"] and dx is not None and math.hypot(dx, dy) > 5.0:
            flags.append("static_but_moved")
        hand_at_t0 = "hand" in bt0
        item = dict(id=vid, template_id=int(tid), template=r["template"], groups=groups, track=track,
                    placeholders=ph, n_frames=n, fps=fps, width=w, height=h, dur_s=round(n / fps, 3),
                    cam_tier=tier, t0_frame=int(t0), onset_frame=(None if onset is None else int(onset)),
                    sec_after_t0=round(sec_after, 3), actor_cat=actor_cat, patient_cat=patient_cat, mover_cat=mover_cat,
                    hand_at_t0=hand_at_t0, boxes_t0={c: [round(v, 1) for v in bb] for c, bb in bt0.items()},
                    dx_pct_w=(None if dx is None else round(dx, 1)), dy_pct_h=(None if dy is None else round(dy, 1)),
                    flags=flags, static_caption=roles["static_caption"], event=roles["event"], rigid=roles["rigid"],
                    direction_expected=d, instruction_neutral=make_instruction(tid, ph, "neutral"),
                    instruction_explicit=make_instruction(tid, ph, "explicit"), cf=counterfactual_for(tid, ph))
        kept.append(item)
    with open(out / "items_draft.jsonl", "w") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    by_track = {}
    for it in kept:
        by_track[it["track"]] = by_track.get(it["track"], 0) + 1
    by_tier = {}
    for it in kept:
        by_tier[it["cam_tier"]] = by_tier.get(it["cam_tier"], 0) + 1
    by_tmpl = {}
    for it in kept:
        by_tmpl[it["template_id"]] = by_tmpl.get(it["template_id"], 0) + 1
    summary = dict(kept=len(kept), dropped=drop, by_track=by_track, by_cam_tier=by_tier,
                   by_template=dict(sorted(by_tmpl.items())), flagged=sum(1 for it in kept if it["flags"]),
                   hand_at_t0=sum(1 for it in kept if it["hand_at_t0"]))
    json.dump(summary, open(out / "select_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_template"}, indent=1))


if __name__ == "__main__":
    main()
