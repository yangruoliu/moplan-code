#!/usr/bin/env python
"""Text-conditioned FlyingObjects (FOT), step 2: counterfactual composites and targets.

Why: Stage-1 ignores its caption because on PhyCo the first frame already determines the
motion (objects always move toward the frame centre).  Here every first frame is shared by
K=4 captions that change WHICH object moves and HOW/WHERE it moves, so the image alone
cannot predict the flow and the caption has to be read.

The motion vocabulary follows the Physics-IQ captions: "rolls towards the right",
"rolls towards the rubber ducky" (and hits it), "the grabber releases the mug", "is slowly
lowered onto ...", "rotates clockwise" (turntables), "roll towards each other", static.

Phase `scenes` (needs ImageNet + cutouts): background + 2-4 cutouts of distinct classes at
640x360 -> scenes/<sid>/frame.jpg, labels.png (instance ids 1..n), scene.json (objects and
the K motion recipes with captions).
Phase `targets` (CPU, needs torch + motion_jepa): recipes -> Stage-1 target .pt files with
the exact PhyCo layout (flow [7,2,160,160] grid px, letterboxed rows 35..124, visibility,
object/future masks, instance_id, affine_2d), manifests, mover_ids.json, static rows apart.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 360                 # composite frame
SIDE = 160                      # Stage-1 grid
SCALE = SIDE / W                # frame px -> grid px
TOP = (SIDE - round(H * SCALE)) // 2   # 35: first content row of the letterbox
HORIZONS = [0.5714, 1.1429, 1.7143, 2.2857, 2.8571, 3.4286, 4.0]
LABEL_SOURCE = "flyobj_text_composite_analytic_v1"
SCENARIO = "flyobj_text_v1"
K_VARIANTS = 4

_S = math.sqrt(0.5)
DIRS_BY_ANGLE = ["right", "upper right", "up", "upper left", "left", "lower left", "down", "lower right"]
DIRS = {"right": (1.0, 0.0), "upper right": (_S, -_S), "up": (0.0, -1.0), "upper left": (-_S, -_S),
        "left": (-1.0, 0.0), "lower left": (-_S, _S), "down": (0.0, 1.0), "lower right": (_S, _S)}
OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up", "upper right": "lower left",
            "lower left": "upper right", "upper left": "lower right", "lower right": "upper left"}
DIR_WEIGHTS = {"right": .26, "left": .26, "upper right": .06, "upper left": .06,
               "lower right": .07, "lower left": .07, "up": .08, "down": .14}
DIR_PHRASES = {
    "right": ["to the right", "towards the right", "toward the right", "rightward", "towards the right side of the frame"],
    "left": ["to the left", "towards the left", "toward the left", "leftward", "towards the left side of the frame"],
    "up": ["upward", "up", "towards the top of the frame"],
    "down": ["downward", "down", "towards the bottom of the frame"],
    "upper right": ["up and to the right", "towards the upper right", "diagonally up and to the right"],
    "upper left": ["up and to the left", "towards the upper left", "diagonally up and to the left"],
    "lower right": ["down and to the right", "towards the lower right", "diagonally down and to the right"],
    "lower left": ["down and to the left", "towards the lower left", "diagonally down and to the left"],
}
# Indoor scenes / surfaces for backgrounds (Physics-IQ is shot indoors on tables).
BACKGROUND_NAMES = [
    "dining table", "desk", "pool table", "bookcase", "wardrobe", "chiffonier", "china cabinet",
    "entertainment center", "file", "home theater", "library", "restaurant", "studio couch", "quilt",
    "doormat", "prayer rug", "tray", "window shade", "sliding door", "four-poster", "medicine chest",
    "refrigerator", "microwave", "dishwasher", "stove", "washer", "shoji", "bakery", "grocery store",
    "toyshop", "barbershop", "bookshop", "confectionery", "shoe shop", "butcher shop", "tobacco shop",
    "rocking chair", "folding chair", "plate rack", "wall clock", "radiator", "window screen",
    "bannister", "altar", "dutch oven", "crate", "chest",
]
SURFACES = ["table", "wooden table", "surface", "tabletop", "floor", "counter"]
# Marginal mix of motion kinds for a freely drawn variant.
KIND_WEIGHTS = {"push": .25, "roll": .15, "fall": .14, "lower": .08, "lift": .04, "spin": .10,
                "towards": .14, "pair": .04, "static": .06}


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def colour_name(rgba: np.ndarray) -> str | None:
    """Dominant colour word of a cutout (None when unsaturated / mixed)."""
    alpha = rgba[..., 3] > 127
    pixels = rgba[..., :3][alpha]
    if len(pixels) < 50:
        return None
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV_FULL).reshape(-1, 3) / 255.0
    hue, sat, val = hsv[:, 0] * 360, hsv[:, 1], hsv[:, 2]
    if np.mean(val < 0.2) > 0.6:
        return "black"
    if np.mean((sat < 0.15) & (val > 0.75)) > 0.6:
        return "white"
    coloured = (sat > 0.35) & (val > 0.25)
    if coloured.mean() < 0.4:
        return None
    angle = np.deg2rad(hue[coloured])
    mean = float(np.rad2deg(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())) % 360)
    spread = float(np.hypot(np.sin(angle).mean(), np.cos(angle).mean()))
    if spread < 0.75:
        return None
    for limit, name in ((15, "red"), (40, "orange"), (70, "yellow"), (170, "green"), (260, "blue"),
                        (320, "purple"), (345, "pink"), (360, "red")):
        if mean < limit:
            return name
    return None


# ----------------------------------------------------------------------------- scenes

def load_backgrounds(imagenet: Path, split: str, rng: random.Random, count: int) -> list[Path]:
    names = {}
    for line in (imagenet / "LOC_synset_mapping.txt").read_text().splitlines():
        synset, desc = line.split(" ", 1)
        names[desc.split(",")[0].strip()] = synset
    synsets = [names[n] for n in BACKGROUND_NAMES if n in names]
    paths: list[Path] = []
    if split == "train":
        for synset in synsets:
            folder = imagenet / "ILSVRC/Data/CLS-LOC/train" / synset
            files = sorted(folder.glob("*.JPEG"))
            paths += rng.sample(files, min(len(files), max(1, count // len(synsets) + 20)))
    else:
        import csv
        wanted = set(synsets)
        with (imagenet / "LOC_val_solution.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row["PredictionString"].split()[0] in wanted:
                    paths.append(imagenet / "ILSVRC/Data/CLS-LOC/val" / f"{row['ImageId']}.JPEG")
        paths.sort()
    rng.shuffle(paths)
    return paths


def background(path: Path, rng: random.Random) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    crop_w = int(w * rng.uniform(0.6, 1.0))
    crop_h = min(h, int(crop_w * H / W))
    crop_w = int(crop_h * W / H)
    x0, y0 = rng.randint(0, w - crop_w), rng.randint(0, h - crop_h)
    image = cv2.resize(image[y0:y0 + crop_h, x0:x0 + crop_w], (W, H), interpolation=cv2.INTER_AREA)
    sigma = rng.choice([0, 0, 0.8, 1.5, 2.5])
    if sigma:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    return np.clip(image.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255)


def compose(rng: random.Random, cutouts: list[dict], bg_path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    frame = background(bg_path, rng)
    labels = np.zeros((H, W), np.uint8)
    count = rng.choices([2, 3, 4], weights=[0.35, 0.4, 0.25])[0]
    chosen, used = [], set()
    for item in rng.sample(cutouts, len(cutouts)):
        if item["name"] not in used:
            chosen.append(item)
            used.add(item["name"])
        if len(chosen) == count:
            break
    boxes, objects = [], []
    for item in chosen:
        rgba = cv2.imread(item["path"], cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            continue
        for _ in range(60):
            height = rng.uniform(66, 150)
            scale = height / rgba.shape[0]
            width = rgba.shape[1] * scale
            if width > 210:
                scale = 210 / rgba.shape[1]
                width, height = 210, rgba.shape[0] * scale
            if height < 44:
                break
            w, h = int(round(width)), int(round(height))
            x0, y0 = rng.randint(10, W - w - 10), rng.randint(10, H - h - 10)
            box = (x0 - 18, y0 - 18, x0 + w + 18, y0 + h + 18)
            if all(box[2] <= b[0] or b[2] <= box[0] or box[3] <= b[1] or b[3] <= box[1] for b in boxes):
                break
        else:
            continue
        if height < 44:
            continue
        patch = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        alpha = cv2.GaussianBlur(patch[..., 3] / 255.0, (3, 3), 0.8)
        if rng.random() < 0.5:  # soft contact shadow
            dx, dy = rng.randint(-6, 6), rng.randint(4, 12)
            shadow = np.zeros((H, W), np.float32)
            sx0, sy0 = max(0, x0 + dx), max(0, y0 + dy)
            sx1, sy1 = min(W, x0 + dx + w), min(H, y0 + dy + h)
            shadow[sy0:sy1, sx0:sx1] = alpha[sy0 - (y0 + dy):sy1 - (y0 + dy), sx0 - (x0 + dx):sx1 - (x0 + dx)]
            shadow = cv2.GaussianBlur(shadow, (0, 0), 5)
            frame *= (1 - 0.35 * shadow)[..., None]
        gain = rng.uniform(0.85, 1.15)
        region = frame[y0:y0 + h, x0:x0 + w]
        region[:] = region * (1 - alpha[..., None]) + np.clip(patch[..., :3] * gain, 0, 255) * alpha[..., None]
        identifier = len(objects) + 1
        labels[y0:y0 + h, x0:x0 + w][alpha >= 0.5] = identifier
        boxes.append(box)
        colour = colour_name(rgba)
        objects.append({"id": identifier, "cutout": item["id"], "name": item["name"], "shape": item["motion"],
                        "colour": colour})
    if len(objects) < 2:
        return None
    for obj in objects:
        ys, xs = np.nonzero(labels == obj["id"])
        if len(xs) < 300:
            return None
        obj.update({"bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "center": [float(xs.mean()), float(ys.mean())],
                    "radius": float(0.25 * (xs.max() - xs.min() + ys.max() - ys.min()))})
    return np.clip(frame, 0, 255).astype(np.uint8), labels, objects


# ----------------------------------------------------------------------------- motions

def contour_points(labels: np.ndarray, identifier: int) -> np.ndarray:
    mask = (labels == identifier).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return np.concatenate([c[:, 0, :] for c in contours]).astype(np.float32)


def contact_distance(points_a: np.ndarray, points_b: np.ndarray, u: np.ndarray, tol: float = 2.0) -> float:
    """Travel of A along unit u until its outline touches B's outline (inf if never)."""
    rel = points_b[None, :, :] - points_a[:, None, :]
    along = rel @ u
    perp = np.abs(rel @ np.array([-u[1], u[0]], np.float32))
    hit = (perp < tol) & (along > 0)
    return float(along[hit].min()) - 3.0 if hit.any() else math.inf


def frame_distance(bbox: list[int], u: np.ndarray, margin: float = 4.0) -> float:
    x0, y0, x1, y1 = bbox
    limits = []
    if u[0] > 1e-6:
        limits.append((W - 1 - margin - x1) / u[0])
    if u[0] < -1e-6:
        limits.append((x0 - margin) / -u[0])
    if u[1] > 1e-6:
        limits.append((H - 1 - margin - y1) / u[1])
    if u[1] < -1e-6:
        limits.append((y0 - margin) / -u[1])
    return max(0.0, min(limits)) if limits else math.inf


def first_obstacle(ctx: dict, mover: int, u: np.ndarray, exclude: set[int]) -> tuple[float, int | None]:
    best, hit = math.inf, None
    for obj in ctx["objects"]:
        other = obj["id"]
        if other == mover or other in exclude:
            continue
        distance = contact_distance(ctx["contours"][mover], ctx["contours"][other], u)
        if distance < best:
            best, hit = distance, other
    return best, hit


def decel_stop_time(rng: random.Random, kind: str) -> float:
    return rng.uniform(3.0, 8.0) if kind == "roll" else rng.uniform(0.8, 3.0)


def magnitude_for(rng: random.Random, adverb: str) -> float:
    if adverb in ("gently", "slowly"):
        return rng.uniform(40, 100)
    if adverb in ("hard", "quickly"):
        return rng.uniform(150, 240)
    return rng.uniform(70, 220)


def translate_motion(ctx: dict, rng: random.Random, mover: int, u: np.ndarray, distance: float,
                     kind: str, exclude: set[int] | None = None) -> list[dict]:
    """A moves along u (decelerating), stops at the frame edge; if it reaches another
    object first it stops there and pushes that object along u (no further chains)."""
    obj = ctx["by_id"][mover]
    stop = decel_stop_time(rng, kind)
    edge = frame_distance(obj["bbox"], u)
    contact, other = first_obstacle(ctx, mover, u, exclude or set())
    spin = 0.0
    if kind == "roll" and abs(u[0]) > 0.7:
        spin = math.copysign(1.0 / max(obj["radius"], 8.0), u[0])
    motions = []
    if other is not None and contact < min(distance, edge):
        motions.append({"id": mover, "profile": "decel", "u": u.tolist(), "plan": distance, "stop_time": stop,
                        "cap": max(contact, 0.0), "start": 0.0, "spin": spin})
        x = 1 - math.sqrt(max(0.0, 1 - max(contact, 0.0) / distance))
        hit_time = x * stop
        target = ctx["by_id"][other]
        pushed = rng.uniform(0.6, 1.1) * (distance - max(contact, 0.0))
        pushed = min(pushed, frame_distance(target["bbox"], u),
                     first_obstacle(ctx, other, u, {mover})[0])
        if pushed >= 6:
            motions.append({"id": other, "profile": "decel", "u": u.tolist(), "plan": pushed,
                            "stop_time": rng.uniform(0.6, 2.5), "cap": pushed, "start": hit_time,
                            "spin": 0.0})
    else:
        motions.append({"id": mover, "profile": "decel", "u": u.tolist(), "plan": distance, "stop_time": stop,
                        "cap": min(distance, edge), "start": 0.0, "spin": spin})
    return motions


def vertical_motion(ctx: dict, rng: random.Random, mover: int, kind: str) -> list[dict]:
    obj = ctx["by_id"][mover]
    u = np.array([0.0, -1.0 if kind == "lift" else 1.0], np.float32)
    room = frame_distance(obj["bbox"], u)
    contact, _ = first_obstacle(ctx, mover, u, set())
    cap = min(room, contact)
    if kind == "fall":
        return [{"id": mover, "profile": "fall", "u": u.tolist(), "plan": cap, "g": rng.uniform(3000, 7000),
                 "cap": cap, "start": 0.0, "spin": 0.0}]
    speed = rng.uniform(15, 60)
    return [{"id": mover, "profile": "linear", "u": u.tolist(), "plan": cap, "speed": speed,
             "cap": min(cap, rng.uniform(40, 200)), "start": rng.uniform(0.0, 0.4), "spin": 0.0}]


def name_of(obj: dict, rng: random.Random, colour_p: float = 0.5) -> str:
    if obj["colour"] and rng.random() < colour_p and obj["colour"] not in obj["name"]:
        return f"{obj['colour']} {obj['name']}"
    return obj["name"]


def scene_sentence(objects: list[dict], names: dict[int, str], rng: random.Random) -> str:
    parts = [f"{article(names[o['id']])} {names[o['id']]}" for o in objects]
    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    surface = rng.choice(SURFACES)
    template = rng.choice(["{L} are on {a} {s}.", "There is {L} on {a} {s}.", "{A} {s} with {L} on it.",
                           "{L} are placed on {a} {s}."])
    sentence = template.format(L=listed, a=article(surface), s=surface, A=article(surface).capitalize())
    return sentence[0].upper() + sentence[1:]


def plan_variant(ctx: dict, rng: random.Random, kind: str, mover: int | None, avoid: dict | None,
                 force: dict | None = None) -> dict | None:
    """One caption + motion recipe.  `avoid` describes a sibling variant to contrast with."""
    objects = ctx["objects"]
    by_id = ctx["by_id"]
    if mover is None:
        mover = rng.choice(objects)["id"]
    obj = by_id[mover]
    names = {o["id"]: name_of(o, rng) for o in objects}
    the = names[mover]
    if kind == "roll" and obj["shape"] != "ball":
        kind = "push"
    adverb = rng.choice(["", "", "", "gently", "hard"]) if kind in ("push", "roll") else ""
    info: dict = {"kind": kind, "mover": mover}
    motions: list[dict] = []
    if kind in ("push", "roll"):
        options = [d for d in DIRS if not avoid or d != avoid.get("direction")]
        if avoid and avoid.get("direction") in OPPOSITE and rng.random() < 0.6:
            options = [OPPOSITE[avoid["direction"]]]
        if force and force.get("direction"):
            options = [force["direction"]]
        weights = [DIR_WEIGHTS[d] for d in options]
        for _ in range(8):
            direction = rng.choices(options, weights=weights)[0]
            u = np.array(DIRS[direction], np.float32)
            if frame_distance(obj["bbox"], u) >= 30:
                break
        else:
            return None
        distance = magnitude_for(rng, adverb)
        motions = translate_motion(ctx, rng, mover, u, distance, kind)
        info["direction"] = direction
        phrase = rng.choice(DIR_PHRASES[direction])
        adv = f" {adverb}" if adverb else ""
        if kind == "roll":
            action = rng.choice([f"The {the} rolls{adv} {phrase}.", f"The {the} is rolled{adv} {phrase}.",
                                 f"The {the} is kicked{adv} {phrase}.", f"A hand rolls the {the}{adv} {phrase}."])
        else:
            action = rng.choice([f"The {the} slides{adv} {phrase}.", f"The {the} is pushed{adv} {phrase}.",
                                 f"A hand pushes the {the}{adv} {phrase}.", f"Someone shoves the {the}{adv} {phrase}.",
                                 f"The {the} is flicked{adv} {phrase}."])
    elif kind == "towards":
        others = [o for o in objects if o["id"] != mover and (not avoid or o["id"] != avoid.get("toward"))]
        if force and force.get("toward") is not None:
            others = [o for o in objects if o["id"] == force["toward"] and o["id"] != mover]
        if not others:
            return None
        target = rng.choice(others)
        away = bool(force["away"]) if force and "away" in force else rng.random() < 0.2
        vector = np.array(target["center"], np.float32) - np.array(obj["center"], np.float32)
        u = vector / (np.linalg.norm(vector) + 1e-6)
        if away:
            u = -u
            if frame_distance(obj["bbox"], u) < 30:
                return None
            distance = magnitude_for(rng, "")
            motions = translate_motion(ctx, rng, mover, u, distance, "roll" if obj["shape"] == "ball" else "push")
        else:
            gap, first = first_obstacle(ctx, mover, u, set())
            if first != target["id"] or not math.isfinite(gap):
                return None  # another object is in the way
            distance = gap + rng.uniform(20, 140)
            motions = translate_motion(ctx, rng, mover, u, distance, "roll" if obj["shape"] == "ball" else "push")
        info.update({"toward": target["id"], "away": away, "direction_vector": u.tolist()})
        verb = "rolls" if obj["shape"] == "ball" else rng.choice(["slides", "is pushed", "moves"])
        prep = "away from" if away else rng.choice(["towards", "toward"])
        action = f"The {the} {verb} {prep} the {names[target['id']]}."
        if force and force.get("hint"):
            # Physics-IQ style: the target object and the screen direction in one sentence
            # ("rolls out of the pipe towards the right and onto the table ... towards the duck").
            screen = DIRS_BY_ANGLE[int(((math.degrees(math.atan2(-u[1], u[0])) % 360) + 22.5) % 360 // 45)]
            phrase = rng.choice(DIR_PHRASES[screen])
            info["direction"] = screen
            action = (f"The {the} {verb} {prep} the {names[target['id']]}, {phrase}." if rng.random() < 0.5
                      else f"The {the} {verb} {phrase}, {prep} the {names[target['id']]}.")
    elif kind in ("fall", "lower", "lift"):
        motions = vertical_motion(ctx, rng, mover, kind)
        if motions[0]["cap"] < 12:
            return None
        below = first_obstacle(ctx, mover, np.array([0.0, 1.0], np.float32), set())[1]
        info["direction"] = "up" if kind == "lift" else "down"
        if force and force.get("into") and below is None:
            return None
        if kind == "fall":
            action = rng.choice([f"The {the} falls.", f"The {the} is released and falls.", f"The {the} drops.",
                                 f"A grabber tool lets go of the {the}.", f"The {the} is dropped."])
            if below is not None and rng.random() < 0.5:
                action = f"The {the} is released and falls onto the {names[below]}."
            if force and force.get("into"):
                # Physics-IQ phrasing: the container after "into" stays put, the named object drops
                action = rng.choice([f"A grabber tool drops the {the} into the {names[below]}.",
                                     f"The {the} is dropped into the {names[below]}.",
                                     f"The {the} falls into the {names[below]}."])
        elif kind == "lower":
            action = rng.choice([f"The {the} is slowly lowered.", f"The {the} is lowered.",
                                 f"The {the} slowly descends."])
            if below is not None and motions[0]["cap"] >= motions[0]["plan"] - 1 and rng.random() < 0.6:
                action = f"The {the} is slowly lowered onto the {names[below]}."
            if force and force.get("into"):
                action = rng.choice([f"The {the} is slowly lowered into the {names[below]}.",
                                     f"A grabber tool lowers the {the} into the {names[below]}.",
                                     f"The {the} is placed into the {names[below]}."])
        else:
            action = rng.choice([f"The {the} is slowly lifted.", f"The {the} is raised.", f"The {the} rises slowly."])
    elif kind == "spin":
        clockwise = not (avoid and avoid.get("clockwise")) if avoid and "clockwise" in avoid else rng.random() < 0.5
        if force and "clockwise" in force:
            clockwise = bool(force["clockwise"])
        omega = rng.uniform(0.5, 1.6)
        motions = [{"id": mover, "profile": "spin", "omega": omega * (1 if clockwise else -1), "start": 0.0,
                    "radius": max(0.5 * (obj["bbox"][2] - obj["bbox"][0]), 6.0), "spin": 0.0}]
        info["clockwise"] = clockwise
        sense = "clockwise" if clockwise else rng.choice(["counterclockwise", "anticlockwise"])
        action = rng.choice([f"The {the} rotates {sense}.", f"The {the} is on a turntable that rotates {sense}.",
                             f"The {the} spins {sense}."])
    elif kind == "pair":
        others = [o for o in objects if o["id"] != mover]
        partner = rng.choice(others)
        vector = np.array(partner["center"], np.float32) - np.array(obj["center"], np.float32)
        u = vector / (np.linalg.norm(vector) + 1e-6)
        gap, first = first_obstacle(ctx, mover, u, set())
        if first != partner["id"] or first_obstacle(ctx, partner["id"], -u, set())[1] != mover \
                or not math.isfinite(gap) or gap < 20:
            return None  # the two must have a clear path to each other
        share = gap / 2
        stop = rng.uniform(1.0, 3.0)
        motions = [{"id": mover, "profile": "decel", "u": u.tolist(), "plan": share * 1.4, "stop_time": stop,
                    "cap": share, "start": 0.0, "spin": 0.0},
                   {"id": partner["id"], "profile": "decel", "u": (-u).tolist(), "plan": share * 1.4,
                    "stop_time": stop, "cap": share, "start": 0.0, "spin": 0.0}]
        verb = "roll" if obj["shape"] == partner["shape"] == "ball" else "move"
        action = f"The {the} and the {names[partner['id']]} {verb} towards each other."
        info["partner"] = partner["id"]
    elif kind == "static":
        action = rng.choice(["Nothing moves.", "All objects stay still.", "The objects rest in place and nothing moves."])
    else:
        raise ValueError(kind)
    caption = f"{scene_sentence(objects, names, rng)} {action} Static shot with no camera movement."
    info.update({"caption": caption, "motions": motions})
    return info


def plan_scene(ctx: dict, rng: random.Random) -> list[dict]:
    """K counterfactual variants that share one first frame."""
    kinds = list(KIND_WEIGHTS)
    weights = list(KIND_WEIGHTS.values())
    variants: list[dict] = []
    first = None
    for _ in range(12):
        first = plan_variant(ctx, rng, rng.choices(kinds, weights=weights)[0], None, None)
        if first and first["kind"] != "static":
            break
    if not first:
        return []
    variants.append(first)
    a = first["mover"]
    # 1) same mover, contrasting motion
    contrast = {"push": "push", "roll": "roll", "towards": "towards", "spin": "spin",
                "fall": "push", "lower": "lift", "lift": "lower", "pair": "push"}[first["kind"]]
    second = plan_variant(ctx, rng, contrast, a, first) or plan_variant(ctx, rng, "push", a, first)
    if second:
        variants.append(second)
    # 2) a different mover
    b_choices = [o["id"] for o in ctx["objects"] if o["id"] != a]
    b = rng.choice(b_choices)
    for _ in range(6):
        third = plan_variant(ctx, rng, rng.choices(kinds[:-2], weights=weights[:-2])[0], b, None)
        if third:
            variants.append(third)
            break
    # 3) static, a pair, or another contrast for B
    roll = rng.random()
    extra_kind = "static" if roll < 0.25 else ("pair" if roll < 0.45 else "push")
    for _ in range(6):
        extra = plan_variant(ctx, rng, extra_kind, b if extra_kind == "push" else None,
                             variants[-1] if extra_kind == "push" and variants[-1]["mover"] == b else None)
        if extra:
            variants.append(extra)
            break
    return variants[:K_VARIANTS]


SINGLE_MOVER = ("push", "roll", "fall", "lower", "lift", "spin", "towards")


def force_of(variant: dict) -> dict:
    kind = variant["kind"]
    if kind in ("push", "roll"):
        return {"direction": variant["direction"]}
    if kind == "spin":
        return {"clockwise": variant["clockwise"]}
    if kind == "towards":
        return {"toward": variant["toward"], "away": variant.get("away", False)}
    return {}


def extend_scene(ctx: dict, rng: random.Random, variants: list[dict], count: int = 4) -> list[dict]:
    """Round-2 variants for an existing first frame.  Noun contrasts repeat a sibling's verb and
    direction (or turning sense / target) with ANOTHER object as the subject, so only the noun
    tells which object moves; "A moves towards B" is contrasted with "B moves towards A"."""
    new: list[dict] = []
    ids = [o["id"] for o in ctx["objects"]]
    sources = [v for v in variants if v["kind"] in SINGLE_MOVER]
    rng.shuffle(sources)
    for source in sources:
        if len(new) >= 2:
            break
        others = [i for i in ids if i != source["mover"]]
        rng.shuffle(others)
        for other in others:
            force = force_of(source)
            if source["kind"] == "towards" and force["toward"] == other:
                force = {**force, "toward": source["mover"]}
            planned = plan_variant(ctx, rng, source["kind"], other, None, force)
            if planned:
                planned.update({"contrast": "noun", "contrast_of": source["sample_id"]})
                new.append(planned)
                break
    for _ in range(8):
        planned = plan_variant(ctx, rng, "towards", None, None)
        if planned:
            planned["contrast"] = "towards_extra"
            new.append(planned)
            break
    pool = [v for v in variants + new if v["kind"] in SINGLE_MOVER]
    rng.shuffle(pool)
    for source in pool:
        if len(new) >= count:
            break
        others = [i for i in ids if i != source["mover"]]
        rng.shuffle(others)
        for other in others:
            force = force_of(source)
            if source["kind"] == "towards" and force["toward"] == other:
                force = {**force, "toward": source["mover"]}
            planned = plan_variant(ctx, rng, source["kind"], other, None, force)
            if planned:
                planned.update({"contrast": "noun", "contrast_of": source.get("sample_id", "")})
                new.append(planned)
                break
    return new[:count]


def extend_towards(ctx: dict, rng: random.Random, count: int = 2) -> list[dict]:
    """Round 3: 'towards <object>' variants whose caption also names the screen direction."""
    new: list[dict] = []
    ids = [o["id"] for o in ctx["objects"]]
    pairs = [(a, b) for a in ids for b in ids if a != b]
    rng.shuffle(pairs)
    for mover, target in pairs:
        if len(new) >= count:
            break
        planned = plan_variant(ctx, rng, "towards", mover, None, {"toward": target, "away": False, "hint": True})
        if planned:
            planned["contrast"] = "towards_hint"
            new.append(planned)
    return new


NONRIGID_TEMPLATES = [
    # liquids: the named rigid object stays put while something is poured on / into it
    "Water is poured into the {a}.", "A dispenser pours red liquid into the {a}.", "Juice is being poured over the {a}.",
    "A stream of water flows onto the {a} from above.", "The {a} slowly fills with liquid from a jug held above it.",
    # light and shadow
    "A spotlight shines on the {a}, casting a long shadow on the wall behind it.",
    "A lamp illuminates the {a} and its shadow slowly stretches across the wall.",
    "The {a} is reflected in a mirror behind it.",
    # heat and smoke
    "Smoke rises from the {a}.", "The {a} is burning with a small flame.", "Steam rises from the {a}.",
    # explicit stationary phrasing (Physics-IQ style)
    "A stationary {a} on the table.", "The {a} rests on the surface and nothing happens.",
    "The {a} and the {b} stay exactly where they are.",
]


def extend_negatives(ctx: dict, rng: random.Random, count: int = 2) -> list[dict]:
    """Round 4: captions that describe a non-rigid process (pouring, light, smoke) or an explicitly
    stationary scene; every rigid object stays still.  Teaches the model to keep the listed
    objects static when the caption's subject is not rigid-object motion."""
    new: list[dict] = []
    objects = ctx["objects"]
    for template in rng.sample(NONRIGID_TEMPLATES, count):
        names = {o["id"]: name_of(o, rng) for o in objects}
        a, b = rng.sample(list(names), 2) if len(names) >= 2 else (next(iter(names)), next(iter(names)))
        action = template.format(a=names[a], b=names[b])
        new.append({"kind": "static", "mover": a, "contrast": "nonrigid", "motions": [],
                    "caption": f"{scene_sentence(objects, names, rng)} {action} Static shot with no camera movement."})
    return new


def extend_into(ctx: dict, rng: random.Random, count: int = 2) -> list[dict]:
    """Round 5: 'dropped / lowered into the <container>' variants.  The object above falls or is
    lowered until it reaches the object below; the container is named but must stay still."""
    new: list[dict] = []
    ids = [o["id"] for o in ctx["objects"]]
    rng.shuffle(ids)
    for mover in ids:
        if len(new) >= count:
            break
        kind = rng.choice(["fall", "lower"])
        planned = plan_variant(ctx, rng, kind, mover, None, {"into": True})
        if planned:
            planned["contrast"] = "into"
            new.append(planned)
    return new


def run_extend_into(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    stats = Counter()
    for path in sorted((out / "scenes").glob("*/scene.json"))[args.shard_index::args.num_shards]:
        scene = json.loads(path.read_text())
        if any(v.get("round") == 5 for v in scene["variants"]):
            stats["already_extended"] += 1
            continue
        labels = cv2.imread(str(path.parent / "labels.png"), cv2.IMREAD_UNCHANGED)
        objects = scene["objects"]
        ctx = {"objects": objects, "by_id": {o["id"]: o for o in objects},
               "contours": {o["id"]: contour_points(labels, o["id"]) for o in objects}}
        rng = random.Random(f"{args.seed}-{scene['scene_id']}-round5")
        next_index = 1 + max(int(v["sample_id"].rsplit("_v", 1)[1]) for v in scene["variants"])
        for i, variant in enumerate(extend_into(ctx, rng)):
            variant["sample_id"] = f"{scene['scene_id']}_v{next_index + i}"
            variant["round"] = 5
            scene["variants"].append(variant)
            stats[f"into_{variant['kind']}"] += 1
        path.write_text(json.dumps(scene, default=lambda v: v.item() if hasattr(v, "item") else float(v)))
        stats["scenes"] += 1
    print(json.dumps({"shard": args.shard_index, **dict(sorted(stats.items()))}))


ABSENT_NOUNS = ["tennis ball", "golf ball", "orange", "lemon", "green apple", "coffee mug", "wine bottle", "teapot",
                "teddy bear", "clock", "soccer ball", "baseball", "banana", "candle", "vase", "pitcher", "basketball",
                "water bottle", "remote control", "piggy bank", "computer mouse", "wallet"]
ABSENT_TEMPLATES = [
    "A {x} rolls out of a pipe towards the {a}.", "A {x} rolls towards the {a}.", "A {x} rolls out of the tube towards the right.",
    "A {x} rolls out of the pipe towards the left and onto the table.", "The {x} is pushed towards the {a}.",
    "A {x} falls onto the {a}.", "A grabber tool releases a {x} above the {a}.", "The {x} slides to the right.",
]


def extend_absent(ctx: dict, rng: random.Random, count: int = 2) -> list[dict]:
    """Round 6: the caption's mover is an object that is NOT in the frame (a ball still inside a
    pipe, an object released from off-screen).  Nothing visible moves, so the target is static.
    Teaches: a named mover without a visible instance -> keep everything still."""
    new: list[dict] = []
    objects = ctx["objects"]
    present = {o["name"] for o in objects}
    choices = [n for n in ABSENT_NOUNS if n not in present]
    for template in rng.sample(ABSENT_TEMPLATES, count):
        names = {o["id"]: name_of(o, rng) for o in objects}
        a = rng.choice(list(names))
        x = rng.choice(choices)
        action = template.format(x=x, a=names[a])
        new.append({"kind": "static", "mover": a, "contrast": "absent", "absent_noun": x, "motions": [],
                    "caption": f"{scene_sentence(objects, names, rng)} {action} Static shot with no camera movement."})
    return new


def run_extend_absent(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    stats = Counter()
    for path in sorted((out / "scenes").glob("*/scene.json"))[args.shard_index::args.num_shards]:
        scene = json.loads(path.read_text())
        if any(v.get("round") == 6 for v in scene["variants"]):
            stats["already_extended"] += 1
            continue
        objects = scene["objects"]
        ctx = {"objects": objects, "by_id": {o["id"]: o for o in objects}}
        rng = random.Random(f"{args.seed}-{scene['scene_id']}-round6")
        next_index = 1 + max(int(v["sample_id"].rsplit("_v", 1)[1]) for v in scene["variants"])
        for i, variant in enumerate(extend_absent(ctx, rng)):
            variant["sample_id"] = f"{scene['scene_id']}_v{next_index + i}"
            variant["round"] = 6
            scene["variants"].append(variant)
            stats["absent"] += 1
        path.write_text(json.dumps(scene, default=lambda v: v.item() if hasattr(v, "item") else float(v)))
        stats["scenes"] += 1
    print(json.dumps({"shard": args.shard_index, **dict(sorted(stats.items()))}))


def run_extend_negatives(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    stats = Counter()
    for path in sorted((out / "scenes").glob("*/scene.json"))[args.shard_index::args.num_shards]:
        scene = json.loads(path.read_text())
        if any(v.get("round") == 4 for v in scene["variants"]):
            stats["already_extended"] += 1
            continue
        objects = scene["objects"]
        ctx = {"objects": objects, "by_id": {o["id"]: o for o in objects}}
        rng = random.Random(f"{args.seed}-{scene['scene_id']}-round4")
        next_index = 1 + max(int(v["sample_id"].rsplit("_v", 1)[1]) for v in scene["variants"])
        for i, variant in enumerate(extend_negatives(ctx, rng)):
            variant["sample_id"] = f"{scene['scene_id']}_v{next_index + i}"
            variant["round"] = 4
            scene["variants"].append(variant)
            stats["nonrigid"] += 1
        path.write_text(json.dumps(scene, default=lambda v: v.item() if hasattr(v, "item") else float(v)))
        stats["scenes"] += 1
    print(json.dumps({"shard": args.shard_index, **dict(sorted(stats.items()))}))


def run_extend_towards(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    stats = Counter()
    for path in sorted((out / "scenes").glob("*/scene.json"))[args.shard_index::args.num_shards]:
        scene = json.loads(path.read_text())
        if any(v.get("round") == 3 for v in scene["variants"]):
            stats["already_extended"] += 1
            continue
        labels = cv2.imread(str(path.parent / "labels.png"), cv2.IMREAD_UNCHANGED)
        objects = scene["objects"]
        ctx = {"objects": objects, "by_id": {o["id"]: o for o in objects},
               "contours": {o["id"]: contour_points(labels, o["id"]) for o in objects}}
        rng = random.Random(f"{args.seed}-{scene['scene_id']}-round3")
        next_index = 1 + max(int(v["sample_id"].rsplit("_v", 1)[1]) for v in scene["variants"])
        for i, variant in enumerate(extend_towards(ctx, rng)):
            variant["sample_id"] = f"{scene['scene_id']}_v{next_index + i}"
            variant["round"] = 3
            scene["variants"].append(variant)
            stats["towards_hint"] += 1
        path.write_text(json.dumps(scene, default=lambda v: v.item() if hasattr(v, "item") else float(v)))
        stats["scenes"] += 1
    print(json.dumps({"shard": args.shard_index, **dict(sorted(stats.items()))}))


def run_extend(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    stats = Counter()
    scenes = sorted((out / "scenes").glob("*/scene.json"))[args.shard_index::args.num_shards]
    for path in scenes:
        scene = json.loads(path.read_text())
        if any(v.get("round") == 2 for v in scene["variants"]):
            stats["already_extended"] += 1
            continue
        labels = cv2.imread(str(path.parent / "labels.png"), cv2.IMREAD_UNCHANGED)
        objects = scene["objects"]
        ctx = {"objects": objects, "by_id": {o["id"]: o for o in objects},
               "contours": {o["id"]: contour_points(labels, o["id"]) for o in objects}}
        rng = random.Random(f"{args.seed}-{scene['scene_id']}-round2")
        next_index = 1 + max(int(v["sample_id"].rsplit("_v", 1)[1]) for v in scene["variants"])
        for i, variant in enumerate(extend_scene(ctx, rng, scene["variants"])):
            variant["sample_id"] = f"{scene['scene_id']}_v{next_index + i}"
            variant["round"] = 2
            scene["variants"].append(variant)
            stats[f"{variant.get('contrast')}_{variant['kind']}"] += 1
        path.write_text(json.dumps(scene, default=lambda v: v.item() if hasattr(v, "item") else float(v)))
        stats["scenes"] += 1
    print(json.dumps({"shard": args.shard_index, **dict(sorted(stats.items()))}))


def run_recalibrate(args: argparse.Namespace) -> None:
    """Spin speed calibrated to Physics-IQ turntables: the visible surface moves 4-35 px/s at
    640x360 (GT v2 median mover displacement at 1.2 s is 9-19 px; teapot 40 px), whereas the
    first FOT draw used omega 0.5-1.6 rad/s, i.e. 2-4x too fast.  Captions are unchanged, so
    only the targets and flow latents of spin variants change (their latents must be removed
    so the cache rebuilds them)."""
    out = Path(args.output_dir)
    changed = []
    for path in sorted((out / "scenes").glob("*/scene.json"))[args.shard_index::args.num_shards]:
        scene = json.loads(path.read_text())
        dirty = False
        for variant in scene["variants"]:
            if variant["kind"] != "spin" or variant.get("spin_calibration") == "piq_surface_4_35px_s":
                continue
            rng = random.Random(f"{args.seed}-{variant['sample_id']}-spin")
            for motion in variant["motions"]:
                if motion.get("profile") == "spin":
                    surface_speed = rng.uniform(4.0, 35.0)
                    motion["omega"] = math.copysign(surface_speed / motion["radius"], motion["omega"])
            variant["spin_calibration"] = "piq_surface_4_35px_s"
            changed.append(variant["sample_id"])
            dirty = True
        if dirty:
            path.write_text(json.dumps(scene, default=lambda v: v.item() if hasattr(v, "item") else float(v)))
    (out / "parts").mkdir(exist_ok=True)
    (out / "parts" / f"recalibrated_{args.shard_index}.txt").write_text("".join(f"{sid}\n" for sid in changed))
    print(json.dumps({"shard": args.shard_index, "spin_recalibrated": len(changed)}))


def run_scenes(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed + 1000 * args.shard_index)
    out = Path(args.output_dir)
    cutouts = [json.loads(l) for l in Path(args.cutouts).read_text().splitlines() if l.strip()]
    for item in cutouts:
        item["path"] = str(Path(args.cutout_dir) / Path(item["path"]).name)
    # held-out cutouts for val: stable hash split, ~12%
    def is_val(item):
        return (int(item["id"].split("_")[-1]) * 2654435761 % 1000) < 120
    pool = [c for c in cutouts if is_val(c) == (args.split == "val")]
    backgrounds = load_backgrounds(Path(args.imagenet), args.split, random.Random(args.seed + 7), args.num_scenes)
    stats = Counter()
    made = 0
    index = args.shard_index
    while made < args.num_scenes // args.num_shards + (1 if args.shard_index < args.num_scenes % args.num_shards else 0):
        scene_id = f"fot_{args.split}_{index:05d}"
        index += args.num_shards
        composed = compose(rng, pool, backgrounds[(index * 7919) % len(backgrounds)])
        if composed is None:
            stats["compose_failed"] += 1
            continue
        frame, labels, objects = composed
        ctx = {"objects": objects, "by_id": {o["id"]: o for o in objects},
               "contours": {o["id"]: contour_points(labels, o["id"]) for o in objects}}
        variants = plan_scene(ctx, rng)
        if len(variants) < 3:
            stats["too_few_variants"] += 1
            continue
        folder = out / "scenes" / scene_id
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(folder / "labels.png"), labels)
        for i, variant in enumerate(variants):
            variant["sample_id"] = f"{scene_id}_v{i}"
            stats[f"kind_{variant['kind']}"] += 1
        (folder / "scene.json").write_text(json.dumps({"scene_id": scene_id, "split": args.split,
                                                       "objects": objects, "variants": variants},
                                                      default=lambda v: v.item() if hasattr(v, "item") else float(v)))
        made += 1
    print(json.dumps({"shard": args.shard_index, "scenes": made, **stats}))


# ----------------------------------------------------------------------------- targets

def displacement_field(points: np.ndarray, centre: np.ndarray, motion: dict, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Frame-px displacement of first-frame points [N,2] at time t, plus self-visibility."""
    tau = max(0.0, t - motion.get("start", 0.0))
    visible = np.ones(len(points), bool)
    if motion["profile"] == "spin":
        # rotation about the object's vertical axis, seen from the side: the visible front
        # surface moves left for clockwise (viewed from above), right for counterclockwise
        radius = motion["radius"]
        offset = np.clip((points[:, 0] - centre[0]) / radius, -1, 1)
        phi = np.arcsin(offset)
        theta = motion["omega"] * tau
        new = radius * np.sin(phi - theta)
        visible = np.cos(phi - theta) > 0.05
        disp = np.zeros_like(points)
        disp[:, 0] = new - radius * offset
        return disp, visible
    u = np.asarray(motion["u"], np.float32)
    if motion["profile"] == "fall":
        s = 0.5 * motion["g"] * tau * tau
    elif motion["profile"] == "linear":
        s = motion["speed"] * tau
    else:
        x = min(tau, motion["stop_time"]) / motion["stop_time"]
        s = motion["plan"] * (1 - (1 - x) ** 2)
    s = min(s, motion["cap"])
    disp = np.repeat((s * u)[None], len(points), 0)
    if motion.get("spin"):
        theta = motion["spin"] * s
        rel = points - centre
        c, sn = math.cos(theta), math.sin(theta)
        rotated = np.stack([c * rel[:, 0] - sn * rel[:, 1], sn * rel[:, 0] + c * rel[:, 1]], 1)
        disp = disp + rotated - rel
    return disp.astype(np.float32), visible


def build_target(scene: dict, variant: dict, labels_frame: np.ndarray, frame_path: Path):
    import torch
    from motion_jepa.flow_dit_slots import fit_instance_affine
    from motion_jepa.video import forward_splat_mask
    grid_labels = np.zeros((SIDE, SIDE), np.int16)
    content_h = round(H * SCALE)
    grid_labels[TOP:TOP + content_h] = cv2.resize(labels_frame, (SIDE, content_h), interpolation=cv2.INTER_NEAREST)
    gy, gx = np.mgrid[0:SIDE, 0:SIDE]
    fx = (gx + 0.5) / SCALE - 0.5
    fy = (gy - TOP + 0.5) / SCALE - 0.5
    T = len(HORIZONS)
    flow = np.zeros((T, 2, SIDE, SIDE), np.float32)
    visible = np.zeros((T, SIDE, SIDE), bool)
    content = np.zeros((SIDE, SIDE), bool)
    content[TOP:TOP + content_h] = True
    visible[:, content] = True
    centres = {o["id"]: np.asarray(o["center"], np.float32) for o in scene["objects"]}
    footprint = np.zeros((T, SIDE, SIDE), bool)
    movers = set()
    for motion in variant["motions"]:
        sel = grid_labels == motion["id"]
        if not sel.any():
            continue
        points = np.stack([fx[sel], fy[sel]], 1).astype(np.float32)
        for k, t in enumerate(HORIZONS):
            disp, self_visible = displacement_field(points, centres[motion["id"]], motion, t)
            flow[k, 0][sel] = disp[:, 0] * SCALE
            flow[k, 1][sel] = disp[:, 1] * SCALE
            dest = points + disp
            inside = (dest[:, 0] >= 0) & (dest[:, 0] <= W - 1) & (dest[:, 1] >= 0) & (dest[:, 1] <= H - 1)
            visible[k][sel] = inside & self_visible
            gxd = np.clip(np.round((dest[:, 0] + 0.5) * SCALE - 0.5), 0, SIDE - 1).astype(int)
            gyd = np.clip(np.round((dest[:, 1] + 0.5) * SCALE - 0.5 + TOP), 0, SIDE - 1).astype(int)
            footprint[k][gyd[inside], gxd[inside]] = True
        movers.add(motion["id"])
    kernel = np.ones((3, 3), np.uint8)
    mover_mask = np.isin(grid_labels, list(movers)) if movers else np.zeros_like(content)
    for k in range(T):
        closed = cv2.morphologyEx(footprint[k].astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        visible[k] &= ~(closed & ~mover_mask)      # static pixels covered by a mover
    flow_t = torch.from_numpy(flow)
    instance = torch.from_numpy(grid_labels)[None]
    magnitude = flow_t.square().sum(1).sqrt()
    moving = torch.zeros(SIDE, SIDE, dtype=torch.bool)
    moving_ids = []
    for identifier in sorted(movers):
        belongs = instance[0] == identifier
        if belongs.sum() >= 4 and float(magnitude[:, belongs].amax(0).mean()) >= 1.0:
            moving |= belongs
            moving_ids.append(int(identifier))
    visible_t = torch.from_numpy(visible)[:, None]
    object_mask = moving[None]
    valid = object_mask[None].float() * visible_t.float()
    future_mask = forward_splat_mask(object_mask[None].float(), flow_t[None])[0] > 0.25
    affine_2d, ids = fit_instance_affine(flow_t, instance * moving[None].long(), valid, eval_side=float(SIDE))
    payload = {
        "cache_format_version": 1, "kind": "targets", "sample_id": variant["sample_id"], "scenario": SCENARIO,
        "flow": flow_t.half(), "visibility": visible_t, "object_mask": object_mask, "future_mask": future_mask,
        "instance_id": instance.short(), "affine_2d": affine_2d.half(), "affine_2d_instance_ids": ids,
        "horizon_seconds": HORIZONS, "source_fps": 30.0, "flow_source": "analytic_2d_kinematics_v1",
        "instance_source": "composite_alpha_exact", "label_source": LABEL_SOURCE, "caption": variant["caption"],
        "caption_source": "flyobj_template_v1", "video_path": str(frame_path),
        "motion_kind": variant["kind"],
    }
    return payload, moving_ids


def run_targets(args: argparse.Namespace) -> None:
    import torch
    out = Path(args.output_dir)
    (out / "targets").mkdir(parents=True, exist_ok=True)
    scenes = sorted((out / "scenes").glob("*/scene.json"))[args.shard_index::args.num_shards]
    rows, movers, stats = [], {}, Counter()
    for path in scenes:
        scene = json.loads(path.read_text())
        labels = cv2.imread(str(path.parent / "labels.png"), cv2.IMREAD_UNCHANGED)
        frame_path = (Path(args.frame_root) if args.frame_root else path.parent.parent) / path.parent.name / "frame.jpg"
        for variant in scene["variants"]:
            sid = variant["sample_id"]
            destination = out / "targets" / f"{sid}.pt"
            payload, moving_ids = build_target(scene, variant, labels, frame_path)
            static = variant["kind"] == "static"
            if not static and not moving_ids:
                stats["skip_no_visible_mover"] += 1
                continue
            torch.save(payload, destination)
            rows.append({"sample_id": sid, "scenario": SCENARIO, "split": scene["split"],
                         "video_path": str(frame_path), "caption": variant["caption"],
                         "source": "flyobj_text_v1", "motion_kind": variant["kind"], "static": static})
            if not static:
                movers[sid] = {"sample_id": sid, "scenario": SCENARIO, "mover_instance_ids": sorted(moving_ids),
                               "named_mover_id": int(variant["mover"]), "motion_kind": variant["kind"]}
            stats[f"{scene['split']}_{variant['kind']}"] += 1
    (out / "parts").mkdir(exist_ok=True)
    (out / "parts" / f"rows_{args.shard_index}.json").write_text(json.dumps(rows))
    (out / "parts" / f"movers_{args.shard_index}.json").write_text(json.dumps(movers))
    print(json.dumps({"shard": args.shard_index, "rows": len(rows), **stats}))


def run_finalize(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    rows, movers = [], {}
    for part in sorted((out / "parts").glob("rows_*.json")):
        rows += json.loads(part.read_text())
    for part in sorted((out / "parts").glob("movers_*.json")):
        movers.update(json.loads(part.read_text()))
    for name, keep in (("manifests", lambda r: not r["static"]), ("manifests_static", lambda r: r["static"])):
        (out / name).mkdir(exist_ok=True)
        for split in ("train", "val"):
            selected = sorted((r for r in rows if r["split"] == split and keep(r)), key=lambda r: r["sample_id"])
            with (out / name / f"{split}.jsonl").open("w") as handle:
                for row in selected:
                    handle.write(json.dumps({k: row[k] for k in ("sample_id", "scenario", "split", "video_path",
                                                                 "caption", "source")}) + "\n")
            print(name, split, len(selected))
    (out / "mover_ids.json").write_text(json.dumps({"schema": "stage1_supervision_mover_ids_v1",
                                                    "rows": len(movers), "mapping": movers,
                                                    "source": "flyobj_composite_recipe_plus_analytic_flow_ge_1px"}))
    print("kinds", Counter(r["motion_kind"] for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["scenes", "extend", "extend_towards", "extend_negatives", "extend_into", "extend_absent", "recalibrate", "targets", "finalize"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutouts", default="")
    parser.add_argument("--cutout-dir", default="")
    parser.add_argument("--imagenet", default="",
                        help="ImageNet ILSVRC-2012 root with LOC_synset_mapping.txt, LOC_val_solution.csv and "
                             "ILSVRC/Data/CLS-LOC/{train,val} (scenes phase)")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--num-scenes", type=int, default=100)
    parser.add_argument("--frame-root", default="", help="where scenes/<sid>/frame.jpg lives at training time")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.phase == "scenes" and not args.imagenet:
        parser.error("the scenes phase needs --imagenet")
    if args.phase == "scenes":
        run_scenes(args)
    elif args.phase == "extend":
        run_extend(args)
    elif args.phase == "recalibrate":
        run_recalibrate(args)
    elif args.phase == "extend_towards":
        run_extend_towards(args)
    elif args.phase == "extend_negatives":
        run_extend_negatives(args)
    elif args.phase == "extend_into":
        run_extend_into(args)
    elif args.phase == "extend_absent":
        run_extend_absent(args)
    elif args.phase == "targets":
        run_targets(args)
    else:
        run_finalize(args)


if __name__ == "__main__":
    main()
