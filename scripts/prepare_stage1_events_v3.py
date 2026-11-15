#!/usr/bin/env python
"""Event curriculum v3: relational events that do not move objects downward.

Why: the v2 event curriculum is 45% downward primary motion and only 7% relational
("A towards B") motion, and its only event-style captions (no screen-direction words)
describe releases. On Physics-IQ the fine-tuned models then move rolling balls,
colliding objects and dominoes downward. v3 adds, on the SAME FOT scenes, first frames,
SAM2 conditions and train/val split as v2:

  collide      A rolls/slides into B and pushes it along the A->B direction
               (ball captions include the Physics-IQ "rolls out of a pipe towards B" form,
               so a visible ball named that way moves instead of abstaining)
  knock_over   P slides/rolls into a tall T; T tips over away from P (rotation about its
               bottom corner, i.e. a domino-like topple)
  stop_short   counterfactual of knock_over: P stops before reaching T, T stays upright
  topple       T tips over toward / away from a named landmark object O

Captions never contain screen-direction words: the direction must be read off the
relation between the named objects in the first frame. Every row gets invariant
paraphrases and, where available, counterfactual partners from the same scene
(role swap, knock vs stop-short, toward vs away) for the paired text loss.

Phases
  targets   (CPU, sharded) analytic targets + rows + pairs + task-object ids per shard
  merge     combined training root = v2 dynamic rows (plain falls/lowers subsampled)
            + v3 train rows; symlinked v2 targets/caches, copied same-scene caches for v3
  prompts   (GPU) umT5 embeddings for every new caption, merged into the v2 prompt table
No Physics-IQ image, caption, flow or score is consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.piq_gt_v2 import flyobj_compose as fot  # noqa: E402

TIMES = [.1, .2, .4, .8, 1.2, 2.4, 4.]
SOURCE_TAG = "stage1_event_curriculum_v3"

# ----------------------------------------------------------------------------- kinematics

_base_displacement_field = fot.displacement_field


def displacement_field(points, centre, motion, t):
    """FOT displacement plus a 'topple' profile: rigid rotation about a bottom corner.

    In image coordinates (y down) a positive angle with R = [[c, -s], [s, c]] turns the
    object clockwise on screen, i.e. its top moves right: sign +1 = tips right about the
    bottom-right corner, sign -1 = tips left about the bottom-left corner. The angle grows
    like a fall (0.5 * alpha * tau^2) and stops at theta_max (object lying on its side).
    """
    if motion["profile"] != "topple":
        return _base_displacement_field(points, centre, motion, t)
    tau = max(0.0, t - motion.get("start", 0.0))
    theta = min(motion["theta_max"], 0.5 * motion["alpha"] * tau * tau) * motion["sign"]
    pivot = np.asarray(motion["pivot"], np.float32)
    rel = points - pivot
    c, s = math.cos(theta), math.sin(theta)
    rotated = np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], 1)
    return (rotated - rel).astype(np.float32), np.ones(len(points), bool)


fot.displacement_field = displacement_field   # build_target resolves the module global


def unit(vector):
    vector = np.asarray(vector, np.float32)
    return vector / (float(np.linalg.norm(vector)) + 1e-6)


def is_tall(obj):
    x0, y0, x1, y1 = obj["bbox"]
    return (y1 - y0) >= 30 and (y1 - y0) >= 1.15 * (x1 - x0)


def topple_room(obj, tip_right, objects=()):
    """The lying object must stay in frame and must not land on another object
    (the analytic topple has no object-object contact after the tip starts)."""
    x0, y0, x1, y1 = obj["bbox"]
    height, width = y1 - y0, x1 - x0
    if not ((x1 + height <= fot.W - 4) if tip_right else (x0 - height >= 4)):
        return False
    lx0, lx1 = (x1, x1 + height) if tip_right else (x0 - height, x0)
    ly0, ly1 = y1 - width, y1
    for other in objects:
        if other["id"] == obj["id"]:
            continue
        ox0, oy0, ox1, oy1 = other["bbox"]
        if min(lx1, ox1) - max(lx0, ox0) > 4 and min(ly1, oy1) - max(ly0, oy0) > 4:
            return False
    return True


def topple_motion(obj, rng, tip_right, start):
    x0, _, x1, y1 = obj["bbox"]
    theta_max = math.radians(rng.uniform(70.0, 88.0))
    duration = rng.uniform(0.35, 0.8)
    return {"id": obj["id"], "profile": "topple", "pivot": [float(x1 if tip_right else x0), float(y1)],
            "sign": 1 if tip_right else -1, "theta_max": theta_max,
            "alpha": 2.0 * theta_max / (duration * duration), "start": float(start), "spin": 0.0}


def approach_motion(obj_id, rng, u, gap, stop_short=False):
    """Decelerating approach along u: reaches the contact distance `gap` (returns hit time)
    or, for stop_short, ends well before it."""
    stop = rng.uniform(0.8, 1.6)
    if stop_short:
        cap = gap * rng.uniform(0.35, 0.7)
        plan = cap * rng.uniform(1.0, 1.25)
        return {"id": obj_id, "profile": "decel", "u": u.tolist(), "plan": plan, "stop_time": stop,
                "cap": cap, "start": 0.0, "spin": 0.0}, None
    plan = gap * rng.uniform(1.3, 2.0)
    hit_time = (1.0 - math.sqrt(max(0.0, 1.0 - gap / plan))) * stop
    return {"id": obj_id, "profile": "decel", "u": u.tolist(), "plan": plan, "stop_time": stop,
            "cap": gap, "start": 0.0, "spin": 0.0}, hit_time


# ----------------------------------------------------------------------------- captions

COLLIDE_BALL = ["The {A} rolls into the {B}.", "The {A} rolls toward the {B} and hits it.",
                "The {A} is rolled at the {B}, pushing it away.", "The {A} rolls across and bumps into the {B}."]
COLLIDE_PIPE = ["The {A} rolls out of a pipe towards the {B}.", "{a_A} rolls out of the tube and into the {B}.",
                "The {A} rolls out of the pipe and knocks the {B} away."]
COLLIDE_SLIDE = ["The {A} slides into the {B}.", "The {A} is pushed into the {B}.",
                 "The {A} is shoved toward the {B} and bumps it away.", "The {A} slides across and hits the {B}."]
KNOCK = ["The {P} {moves} into the {T} and knocks it over.", "The {P} hits the {T}, which tips over.",
         "The {T} is knocked over by the {P}.", "The {P} {moves} into the {T}, and the {T} falls over."]
STOP_SHORT = ["The {P} {moves} toward the {T} but stops before touching it.",
              "The {P} {moves} toward the {T} and stops short of it.",
              "The {P} is nudged toward the {T}, but the {T} is not touched and stays upright."]
TOPPLE_TOWARD = ["The {T} tips over toward the {O}.", "The {T} falls over sideways toward the {O}.",
                 "The {T} topples toward the {O}."]
TOPPLE_AWAY = ["The {T} tips over away from the {O}.", "The {T} falls over sideways, away from the {O}.",
               "The {T} topples away from the {O}."]


def fill(template, names, roles, moves):
    values = {k: names[v] for k, v in roles.items()}
    values["moves"] = moves
    if "a_A" in template:
        values["a_A"] = f"{fot.article(names[roles['A']]).capitalize()} {names[roles['A']]}"
    return template.format(**values)


def caption_family(objects, names, rng, templates, roles, moves="slides", count=3):
    """Main caption + paraphrases: different action templates, re-drawn scene sentence."""
    chosen = rng.sample(templates, min(count, len(templates)))
    return [f"{fot.scene_sentence(objects, names, rng)} {fill(t, names, roles, moves)} Static shot with no camera movement."
            for t in chosen]


# ----------------------------------------------------------------------------- planning

def plan_collide(ctx, rng, a, b, names):
    A = ctx["by_id"][a]
    B = ctx["by_id"][b]
    u = unit(np.asarray(B["center"], np.float32) - np.asarray(A["center"], np.float32))
    gap, first = fot.first_obstacle(ctx, a, u, set())
    if first != b or not math.isfinite(gap) or gap < 8:
        return None
    ball = A["shape"] == "ball"
    motions = fot.translate_motion(ctx, rng, a, u, gap + rng.uniform(40, 160), "roll" if ball else "push")
    if len(motions) < 2 or motions[1]["id"] != b:
        return None          # B must actually be pushed
    templates = (COLLIDE_BALL + COLLIDE_PIPE) if ball else COLLIDE_SLIDE
    captions = caption_family(ctx["objects"], names, rng, templates, {"A": a, "B": b})
    return {"kind": "collide", "event": "collide", "mover": a, "partner": b, "motions": motions,
            "direction_vector": u.tolist(), "caption": captions[0], "paraphrases": captions[1:]}


def plan_knock(ctx, rng, p, t, names, stop_short=False):
    P = ctx["by_id"][p]
    T = ctx["by_id"][t]
    if not is_tall(T):
        return None
    u = unit(np.asarray(T["center"], np.float32) - np.asarray(P["center"], np.float32))
    if abs(float(u[0])) < 0.6:
        return None          # side impacts only: the target tips over sideways
    gap, first = fot.first_obstacle(ctx, p, u, set())
    if first != t or not math.isfinite(gap) or gap < 10:
        return None
    tip_right = float(u[0]) > 0
    if not stop_short and not topple_room(T, tip_right, ctx["objects"]):
        return None
    approach, hit_time = approach_motion(p, rng, u, gap, stop_short=stop_short)
    moves = "rolls" if P["shape"] == "ball" else "slides"
    if stop_short:
        captions = caption_family(ctx["objects"], names, rng, STOP_SHORT, {"P": p, "T": t}, moves)
        return {"kind": "stop_short", "event": "stop_short", "mover": p, "partner": t, "motions": [approach],
                "direction_vector": u.tolist(), "caption": captions[0], "paraphrases": captions[1:]}
    captions = caption_family(ctx["objects"], names, rng, KNOCK, {"P": p, "T": t}, moves)
    return {"kind": "knock_over", "event": "knock_over", "mover": p, "partner": t,
            "motions": [approach, topple_motion(T, rng, tip_right, hit_time)], "direction_vector": u.tolist(),
            "caption": captions[0], "paraphrases": captions[1:]}


def plan_topple(ctx, rng, t, o, names, toward=True):
    T = ctx["by_id"][t]
    O = ctx["by_id"][o]
    if not is_tall(T):
        return None
    dx = float(O["center"][0]) - float(T["center"][0])
    if abs(dx) < 40:
        return None          # the landmark must be clearly to one side
    tip_right = (dx > 0) == toward
    if not topple_room(T, tip_right, ctx["objects"]):
        return None
    captions = caption_family(ctx["objects"], names, rng, TOPPLE_TOWARD if toward else TOPPLE_AWAY, {"T": t, "O": o})
    return {"kind": "topple", "event": "topple_toward" if toward else "topple_away", "mover": t, "landmark": o,
            "motions": [topple_motion(T, rng, tip_right, rng.uniform(0.0, 0.3))], "tip_right": tip_right,
            "caption": captions[0], "paraphrases": captions[1:]}


def plan_scene_v3(scene, labels):
    objects = scene["objects"]
    ctx = {"objects": objects, "by_id": {o["id"]: o for o in objects},
           "contours": {o["id"]: fot.contour_points(labels, o["id"]) for o in objects}}
    rng = random.Random("event-v3-" + scene["scene_id"])
    names = {o["id"]: fot.name_of(o, rng) for o in objects}
    ids = [o["id"] for o in objects]
    pairs = [(a, b) for a in ids for b in ids if a != b]
    rng.shuffle(pairs)
    variants = []
    # 1) collisions: A into B and, when feasible, the role swap B into A
    for a, b in pairs:
        first = plan_collide(ctx, rng, a, b, names)
        if first is None:
            continue
        variants.append(first)
        swap = plan_collide(ctx, rng, b, a, names)
        if swap is not None:
            variants.append(swap)
        break
    # 2) knock over vs stop short (same pusher and target)
    for p, t in pairs:
        knock = plan_knock(ctx, rng, p, t, names)
        if knock is None:
            continue
        short = plan_knock(ctx, rng, p, t, names, stop_short=True)
        variants.append(knock)
        if short is not None:
            variants.append(short)
        break
    # 3) topple toward vs away from a landmark
    for t, o in pairs:
        toward = plan_topple(ctx, rng, t, o, names, toward=True)
        away = plan_topple(ctx, rng, t, o, names, toward=False)
        if toward is None or away is None:
            continue
        variants += [toward, away]
        break
    for index, variant in enumerate(variants):
        variant["sample_id"] = f"event3_{scene['scene_id']}_v{index:02d}"
    return variants


def distinct_semantics(a, b):
    """Counterfactual = a different activity set, actor, direction or topple side."""
    am, bm = a["motions"], b["motions"]
    if {m["id"] for m in am} != {m["id"] for m in bm}:
        return True
    other = {m["id"]: m for m in bm}
    for motion in am:
        paired = other[motion["id"]]
        if motion["profile"] != paired["profile"]:
            return True
        if motion["profile"] == "topple" and motion["sign"] != paired["sign"]:
            return True
        if "u" in motion and "u" in paired and sum(x * y for x, y in zip(motion["u"], paired["u"])) < .8:
            return True
    return False


def task_ids(variant):
    """Same policy as build_event_task_object_ids.py: actor plus simulated participants."""
    return sorted({variant["mover"]} | {m["id"] for m in variant["motions"]})


# ----------------------------------------------------------------------------- phases

def scene_list(v2_root):
    scenes = []
    for path in sorted((v2_root / "scenes").glob("*.json")):
        scene = json.loads(path.read_text())
        scenes.append((scene["scene_id"], scene["split"], scene.get("source_frame")))
    return scenes


def run_targets(args):
    root = args.output
    for folder in ("targets", "scenes", "parts"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    fot.HORIZONS = TIMES
    rows, pairs, ids, stats = [], {}, {}, Counter()
    for scene_id, split, _ in scene_list(args.v2_root)[args.shard_index::args.num_shards]:
        folder = args.source / "scenes" / scene_id
        scene = json.loads((folder / "scene.json").read_text())
        if scene["split"] != split:
            raise ValueError(f"split mismatch for {scene_id}")
        labels = cv2.imread(str(folder / "labels.png"), cv2.IMREAD_UNCHANGED)
        frame = folder / "frame.jpg"
        digest = hashlib.sha256((folder / "scene.json").read_bytes()).hexdigest()
        kept = []
        for variant in plan_scene_v3(scene, labels):
            target, movers = fot.build_target(scene, variant, labels, frame)
            if not movers:
                stats["skipped_no_observable_motion"] += 1
                continue
            target["event_recipe"] = SOURCE_TAG
            target["source_scene_sha256"] = digest
            torch.save(target, root / "targets" / (variant["sample_id"] + ".pt"))
            row = {"sample_id": variant["sample_id"], "scenario": fot.SCENARIO, "split": split,
                   "video_path": str(frame), "caption": variant["caption"], "source": SOURCE_TAG,
                   "motion_kind": variant["kind"], "scene_id": scene_id, "event": variant["event"]}
            rows.append(row)
            kept.append((variant, target))
            ids[variant["sample_id"]] = {"sample_id": variant["sample_id"], "scenario": fot.SCENARIO,
                                         "mover_instance_ids": task_ids(variant),
                                         "source_scene_sha256": digest, "event": variant["event"]}
            stats[f"{split}/{variant['event']}"] += 1
        for variant, target in kept:
            options = [{"relation": "invariant", "caption": c} for c in variant["paraphrases"]]
            partner_key = (variant.get("partner"), variant.get("landmark"))
            ranked = sorted(kept, key=lambda item: (
                {item[0]["mover"], item[0].get("partner"), item[0].get("landmark")} != {variant["mover"], *partner_key},
                item[0]["sample_id"]))
            added = []
            for other, other_target in ranked:
                if other["sample_id"] == variant["sample_id"] or not distinct_semantics(variant, other):
                    continue
                common = target["visibility"].bool() & other_target["visibility"].bool()
                if not bool((torch.where(common, target["flow"] - other_target["flow"], 0).abs() > .01).any()):
                    continue
                added.append({"relation": "counterfactual", "sample_id": other["sample_id"], "caption": other["caption"]})
                if len(added) >= max(1, len(options)):
                    break
            pairs[variant["sample_id"]] = options + added
            stats["pairs_counterfactual"] += len(added)
        (root / "scenes" / f"{scene_id}.json").write_text(json.dumps(
            {"scene_id": scene_id, "split": split, "objects": scene["objects"], "source_frame": str(frame),
             "variants": [v for v, _ in kept], "horizon_seconds": TIMES},
            default=lambda value: value.item() if hasattr(value, "item") else float(value)))
    (root / "parts" / f"{args.shard_index:03d}.json").write_text(
        json.dumps({"rows": rows, "pairs": pairs, "ids": ids, "stats": stats}))
    print(json.dumps({"shard": args.shard_index, **stats}), flush=True)


def direction_of(motion):
    if motion["profile"] == "spin":
        return "spin"
    if motion["profile"] == "topple":
        return "right" if motion["sign"] > 0 else "left"     # centroid moves sideways first
    u = motion["u"]
    angle = math.degrees(math.atan2(u[1], u[0]))
    return "right" if -45 <= angle < 45 else "down" if 45 <= angle < 135 else "up" if -135 <= angle < -45 else "left"


def keep_v2_row(row, args):
    """Deterministically subsample plain downward v2 rows; releases keep their pairs."""
    fraction = {"fall": args.keep_fall, "lower": args.keep_lower}.get(row.get("event"), 1.0)
    if fraction >= 1.0:
        return True
    value = int(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < fraction


def link(src: Path, dst: Path):
    if not dst.exists():
        dst.symlink_to(src)


def token_lengths(captions, wan_model):
    """umT5 token counts (with EOS) exactly as cache_stage1_flowdit_prompts counts them."""
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained(Path(wan_model) / "tokenizer", local_files_only=True)
    captions = sorted(set(captions))
    lengths = {}
    for start in range(0, len(captions), 512):
        chunk = captions[start:start + 512]
        encoded = tokenizer(chunk, add_special_tokens=True, truncation=False)["input_ids"]
        lengths.update({c: len(ids) for c, ids in zip(chunk, encoded)})
    return lengths


def run_merge(args):
    v3 = args.output
    parts = sorted((v3 / "parts").glob("*.json"))
    if len(parts) != args.num_shards:
        raise ValueError(f"incomplete v3 targets: {len(parts)}/{args.num_shards} shards")
    rows, pairs, ids, stats = [], {}, {}, Counter()
    for path in parts:
        part = json.loads(path.read_text())
        rows += part["rows"]; pairs.update(part["pairs"]); ids.update(part["ids"]); stats.update(part["stats"])
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate v3 sample ids")
    # captions must fit the prompt pad length (the encoder rejects >= pad tokens):
    # drop over-long rows and pair specs before anything references them
    lengths = token_lengths([r["caption"] for r in rows] +
                            [s["caption"] for specs in pairs.values() for s in specs], args.wan_model)
    fits = lambda caption: lengths[caption] < args.pad_length
    long_rows = {r["sample_id"] for r in rows if not fits(r["caption"])}
    rows = [r for r in rows if r["sample_id"] not in long_rows]
    long_specs = 0
    for sid in list(pairs):
        if sid in long_rows:
            del pairs[sid]
            continue
        kept_specs = [s for s in pairs[sid] if fits(s["caption"]) and s.get("sample_id", sid) not in long_rows]
        long_specs += len(pairs[sid]) - len(kept_specs)
        pairs[sid] = kept_specs
    ids = {sid: v for sid, v in ids.items() if sid not in long_rows}
    stats["dropped_long_caption_rows"] = len(long_rows)
    stats["dropped_long_caption_specs"] = long_specs
    stats["max_caption_tokens"] = max(lengths.values())
    combined = args.combined
    for folder in ("targets", "flowdit", "sam2_conditions", "manifests", "manifests_relational"):
        (combined / folder).mkdir(parents=True, exist_ok=True)
    v2_dynamic = {split: [json.loads(l) for l in (args.v2_manifests / f"{split}.jsonl").read_text().splitlines()]
                  for split in ("train", "val")}
    train = [r for r in v2_dynamic["train"] if keep_v2_row(r, args)]
    v3_train = [r for r in rows if r["split"] == "train"]
    v3_val = [r for r in rows if r["split"] == "val"]
    manifest_train = sorted(train + v3_train, key=lambda r: r["sample_id"])
    manifest_val = v2_dynamic["val"]          # unchanged fixed validation for comparability
    (combined / "manifests" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest_train))
    (combined / "manifests" / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest_val))
    (combined / "manifests_relational" / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in v3_val))
    (combined / "manifests_relational" / "train.jsonl").write_text("")
    # caches: every v2 sample stays resolvable through the combined (primary) paths --
    # besides the manifests, the trainer also loads the scene-disjoint gate-calibration
    # rows through them. v3 rows get same-scene copies with their own sample id.
    v2_data = args.v2_data
    for folder in ("targets", "flowdit", "sam2_conditions"):
        for source in (v2_data / folder).glob("*.pt"):
            link(source, combined / folder / source.name)
    for row in rows:
        sid = row["sample_id"]
        link(v3 / "targets" / f"{sid}.pt", combined / "targets" / f"{sid}.pt")
        sibling = f"event_{row['scene_id']}_v00"
        for folder in ("flowdit", "sam2_conditions"):
            destination = combined / folder / f"{sid}.pt"
            if destination.exists():
                continue
            payload = torch.load(v2_data / folder / f"{sibling}.pt", map_location="cpu", weights_only=False)
            if folder == "flowdit" and payload.get("first_frame_source") != row["video_path"]:
                raise ValueError(f"{sid}: sibling cache uses a different first frame")
            payload["sample_id"] = sid
            torch.save(payload, destination)
    # pairs: the trainer resolves partners over the whole joint pool, so a v2 pair stays
    # valid while its base and partner are in the combined dynamic manifest or in the
    # (unchanged) event_static source -- e.g. release (dynamic) <-> hold (static).
    members = {r["sample_id"] for r in manifest_train}
    static_train = {json.loads(l)["sample_id"] for l in
                    (args.v2_static_manifests / "train.jsonl").read_text().splitlines()}
    pool = members | static_train
    v2_pairs = json.loads((v2_data / "event_pairs_train.json").read_text())["pairs"]
    merged_pairs, dropped = {}, Counter()
    for sid, specs in v2_pairs.items():
        if sid not in pool:
            dropped["base_subsampled"] += len(specs)
            continue
        kept_specs = [s for s in specs if s.get("sample_id", sid) in pool]
        dropped["partner_subsampled"] += len(specs) - len(kept_specs)
        if kept_specs:
            merged_pairs[sid] = kept_specs
    merged_pairs.update({sid: specs for sid, specs in pairs.items() if sid in members and specs})
    (combined / "event_pairs_train.json").write_text(json.dumps({"schema": "stage1_event_pairs_v1", "pairs": merged_pairs}))
    (combined / "event_pairs_val.json").write_text((v2_data / "event_pairs_val.json").read_text())
    (combined / "event_pairs_relational_val.json").write_text(json.dumps(
        {"schema": "stage1_event_pairs_v1", "pairs": {sid: specs for sid, specs in pairs.items()
                                                      if specs and sid in {r["sample_id"] for r in v3_val}}}))
    # task-object identity labels
    mapping = dict(json.loads(args.v2_task_ids.read_text())["mapping"])   # v2 policy, every v2 row
    mapping.update(ids)
    missing = (members | {r["sample_id"] for r in manifest_val}) - set(mapping)
    if missing:
        raise ValueError(f"{len(missing)} rows without task-object ids, e.g. {sorted(missing)[:3]}")
    (combined / "task_object_ids.json").write_text(json.dumps(
        {"schema": "stage1_supervision_mover_ids_v1", "rows": len(mapping), "mapping": mapping,
         "policy": "recipe actor plus simulated participants (v2 repair policy, v3 recipes)"}, indent=1))
    # direction census of the primary training source
    scenes = {}
    for folder in (v2_data / "scenes", v3 / "scenes"):
        for path in folder.glob("*.json"):
            for variant in json.loads(path.read_text())["variants"]:
                scenes[variant["sample_id"]] = variant
    census = Counter()
    for row in manifest_train:
        motions = scenes[row["sample_id"]]["motions"]
        census[direction_of(motions[0]) if motions else "none"] += 1
    total = sum(census.values())
    receipt = {"schema": SOURCE_TAG, "v3_rows": len(rows), "v3_counts": stats, "train_rows": len(manifest_train),
               "v2_train_rows_kept": len(train), "v2_train_rows_total": len(v2_dynamic["train"]),
               "keep_fall": args.keep_fall, "keep_lower": args.keep_lower,
               "pairs_train": len(merged_pairs), "v2_pair_specs_dropped": dict(dropped),
               "pair_relations": dict(Counter(s["relation"] for v in merged_pairs.values() for s in v)),
               "primary_direction_census": dict(census),
               "primary_direction_share": {k: round(v / total, 4) for k, v in census.items()},
               "validation": "v2 fixed dynamic val unchanged; v3 relational val in manifests_relational/val.jsonl",
               "benchmark_inputs_used": False}
    (combined / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


@torch.inference_mode()
def run_prompts(args):
    from scripts.cache_stage1_flowdit_prompts import encode_captions
    from motion_jepa.flow_dit_data import caption_key
    combined = args.combined
    base = torch.load(args.v2_data / "prompt_table.pt", map_location="cpu", weights_only=False)
    embeddings = dict(base["embeddings"])
    wanted = {}
    for name in ("manifests/train.jsonl", "manifests/val.jsonl", "manifests_relational/val.jsonl"):
        for line in (combined / name).read_text().splitlines():
            row = json.loads(line)
            wanted[caption_key(row["caption"])] = row["caption"]
    for name in ("event_pairs_train.json", "event_pairs_val.json", "event_pairs_relational_val.json"):
        for specs in json.loads((combined / name).read_text())["pairs"].values():
            for spec in specs:
                wanted[caption_key(spec["caption"])] = spec["caption"]
    todo = {key: caption for key, caption in wanted.items() if key not in embeddings}
    print(json.dumps({"captions_needed": len(wanted), "new_to_encode": len(todo)}), flush=True)
    pad = int(base["prompt_pad_length"])
    too_long = {c: n for c, n in token_lengths(list(todo.values()), args.wan_model).items() if n >= pad}
    if too_long:   # fail before the (slow) encoder pass, not after it
        raise SystemExit(f"{len(too_long)} captions reach the {pad}-token pad length; rerun merge: "
                         + json.dumps(dict(list(too_long.items())[:5])))
    new, provenance = encode_captions(todo, Path(args.wan_model), int(base["prompt_pad_length"])) if todo else ({}, {})
    embeddings.update(new)
    destination = combined / "prompt_table.pt"
    temporary = destination.with_suffix(".tmp")
    torch.save({"embeddings": embeddings, "negative_key": base["negative_key"],
                "prompt_pad_length": base["prompt_pad_length"],
                "provenance": {"base": str(args.v2_data / "prompt_table.pt"), "v3_new": len(new),
                               "v3_encoder": {k: v for k, v in provenance.items() if k != "token_counts"}}}, temporary)
    temporary.replace(destination)
    print(json.dumps({"prompt_table": str(destination), "entries": len(embeddings), "new": len(new)}), flush=True)


# Inputs each phase reads (v3, v4 and v4.4 share this command line).
PHASE_INPUTS = {
    "targets": ("source", "v2_root"),
    "merge": ("combined", "v2_data", "v2_manifests", "v2_static_manifests", "v2_task_ids", "wan_model"),
    "prompts": ("combined", "v2_data", "wan_model"),
}


def add_input_arguments(p):
    p.add_argument("--source", type=Path,
                   help="FOT v1 scene root (flyobj_compose.py output, with scenes/<scene_id>/scene.json)")
    p.add_argument("--v2-root", type=Path,
                   help="v2 event curriculum root (prepare_stage1_events.py output, with scenes/*.json)")
    p.add_argument("--v2-data", type=Path,
                   help="v2 data root with targets/, flowdit/, sam2_conditions/, scenes/, prompt_table.pt "
                        "and event_pairs_{train,val}.json (normally the same directory as --v2-root)")
    p.add_argument("--v2-manifests", type=Path, help="v2 dynamic-event manifests (train.jsonl, val.jsonl)")
    p.add_argument("--v2-static-manifests", type=Path, help="v2 static (hold) manifests (train.jsonl)")
    p.add_argument("--v2-task-ids", type=Path,
                   help="v2 task-object id manifest (build_event_task_object_ids.py output)")


def check_phase_inputs(p, args):
    missing = ["--" + name.replace("_", "-") for name in PHASE_INPUTS[args.phase] if not getattr(args, name)]
    if missing:
        p.error(f"phase {args.phase} needs " + ", ".join(missing))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["targets", "merge", "prompts"])
    add_input_arguments(p)
    p.add_argument("--output", type=Path, required=True, help="v3-only root (targets, scenes, parts)")
    p.add_argument("--combined", type=Path, default=None, help="combined training root for merge/prompts")
    p.add_argument("--pad-length", type=int, default=64, help="prompt pad length of the v2 prompt table")
    p.add_argument("--keep-fall", type=float, default=0.3)
    p.add_argument("--keep-lower", type=float, default=0.5)
    p.add_argument("--wan-model", default=os.environ.get("WAN_MODEL_DIR", ""),
                   help="Wan2.2-I2V-A14B-Diffusers directory (tokenizer, text encoder); default $WAN_MODEL_DIR")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()
    check_phase_inputs(p, args)
    torch.set_num_threads(1)
    if args.phase == "targets":
        run_targets(args)
    elif args.phase == "merge":
        run_merge(args)
    else:
        run_prompts(args)


if __name__ == "__main__":
    main()
