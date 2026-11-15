#!/usr/bin/env python
"""Physics reward of a generated clip: does every object go where it really went, and does the rest stay put?

Object positions come from SAM2 mask propagation (scripts/control/masktrack.py: centroid of the object's mask), the
background and the camera from CoTracker points on the free dense grid.  Point tracking alone is not enough for the
objects: at the generator's 7 fps it reads many fast real falls as "did not move", while the mask reading of the
7-fps clip agrees closely with the source-rate one.  A scene built without mask readings falls back to the median of
the object's own points (mode "points", unit tests and old scenes).

Two steps, both pure numpy (the trackers are run by the caller, scripts/control/measure.py):

  build_scene(...)   once per real clip: the measured displacement of every object of the measured plan
                     (scripts/control/real_scene.py: Stage-1's instances + the movers it was not shown) and of the
                     free dense-grid cells, at up to 6 evaluation times, relative to the camera.
  score(scene, ...)  per generated clip: the SAME first-frame points tracked in the generated video.

Lengths are in work-frame pixels; D is the frame diagonal.  An object "really moves" at a time when its displacement
is >= 0.03 D, "is still" when <= 0.01 D; in between it is not scored.

  moving cell   rho = |g| / |r|, cos = angle between g and r
                s_dir = cos * q - (1 - q),  q = min(1, rho / 0.25)     (not moving = -1 like reversed ...)
                s_mag = clip(2 - |log2 rho|, 0, 1)
                a     = s_dir * (0.5 + 0.5 * s_mag)                     (... but frozen = -0.5, reversed at full length = -1,
                                                                         right direction at 3x the length = 0.71, correct = 1)
                the same on the displacement of each segment between evaluation times (weight 0.3): arriving in one
                jump is not the same as travelling there.
                An object the tracker loses in the generated clip while it is visible in the real one scores -1.
  R_actor       mean over really moving objects of their mean cell score (every object weighs the same).
                A clip in which nothing moves: R_hold = 1 - 2 * P_quiet.
  P_quiet       every object that does not really move (still, or below the "moves" threshold) must be where it really
                is: a cell is violated when the generated object is >= 0.03 D away from its real position.  The term
                is the violated share of the WORST such object, not the mean over objects: one hand that carries the
                cube down instead of letting it go is a wrong video, however many other things stay in place
                (a mean over objects would hide it).
  P_static      0.6 * P_quiet + 0.2 * free background cells that moved + 0.2 * camera drift
  P_identity    share of object cells lost by the tracker in the generated clip only, plus the drop of dense-grid visibility

  R = clip(R_actor - P_static - 0.5 * P_identity, -1, 1)
      (right mover, and a second object wrongly travelling with it: about 0.4-0.6; nothing moves: -0.5; reversed: -1)

That is version "v4" (score(..., version="v4"), kept bit-identical).  Version "v5" (the default, VERSION) judges the same
readings without its plateaus and ties:

  times         up to 12 evaluation times chosen from the STORED frames of the scene (select_times; a scene of
                rescene_v5.py stores every frame), capped by the generated clip; the onset uses every stored frame.
  moving cell   cell_score_v5: rho = |g| / |r|, L = log2 rho, c = cos; w = min(1, rho / 0.25);
                m = clip(1 - |L| / 2, 0, 1), p = clip(|L| / 2 - 1, 0, 1); inner = c (0.5 + 0.5 m) - 0.5 p (c >= 0), c - 0.5 p (c < 0);
                s = clip(w inner + (1 - w) (-0.5), -1, 1).  Anchors: frozen -0.5, correct 1, x0.5 / x2 0.75, x3 0.60,
                x4 0.50, x8 0.25, x16 0, 30 deg 0.87, 60 deg 0.50, 90 deg 0, reversed -1 (half a point per octave,
                far above the centroid noise).  The same on the segments between evaluation times.
  rotation cell a cell that moves by its points but not by its centroid (relabel_v2 contract) with a trusted real
                rotation |theta_r| >= 10 deg: rot_score_v5 on the rigid-fit angle of the object's own tracked points in
                the generated clip (>= 6 points seen at both frames, residual <= 3 px); a failed fit is rot_unscored,
                not penalised, so a correct pure rotation is not scored as a translation error.
  onset         first stored frame with |r| >= 4 px against the same in the generated clip (never: H + 1 / fps);
                s_on = 1 - 2 min(1, |dt| / 0.75 s).
  object value  0.55 mean(cells) + 0.30 mean(segments) + 0.15 s_on, renormalised over the terms present; a cell visible
                in the real clip and lost in the generated one stays -1.
  leak          soft: off = clip((|g - r| - STILL) / (2 MOVE - STILL), 0, 1) on the still cells (0 up to 5.2 px, 0.4 at
                15.5 px, 1 at 31 px), leak = max over objects of the mean over times; P_bg on the same ramp; P_cam as v4.
  R_raw         with a mover: mean object value - (0.6 leak + 0.2 P_bg + 0.2 P_cam) - 0.5 P_identity
                without:      (1 - 2 leak) - (0.2 P_bg + 0.2 P_cam) - 0.5 P_identity        (leak is not counted twice)
  R             clip(R_raw, -2, 1); both are returned (a group of all-wrong candidates no longer ties at -1).

Physics-IQ futures never reach this file: a scene needs a real continuation, and only training / dev clips have one.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import motion_plan as mp  # noqa: E402
from scripts.control import real_scene as rs  # noqa: E402

MOVE_FRAC, STILL_FRAC = 0.03, 0.01
VERSION_V4, VERSION = "physics_reward_v4", "physics_reward_v5"
ONSET_PX, ONSET_S, ROT_MIN_DEG, ROT_MIN_POINTS, ROT_MAX_RESIDUAL = 4.0, 0.75, 10.0, 6, 3.0
SCENE_KT = ("real", "real_ok", "moving", "static", "moving_centroid", "static_centroid", "max_point_px", "fill", "real_rot", "rot_known")   # [K, T]
SCENE_T = ("camera_real", "pool_real", "pool_ok", "visible_rate_real")                                                                      # [T, ...]


def eval_times(clip_seconds: float, fps: float = 7.0, max_times: Optional[int] = 6) -> np.ndarray:
    """Frame times k / fps inside the clip (k >= 1), thinned evenly to at most `max_times` (None: every frame), the last
    frame always kept."""
    last = int(np.floor(clip_seconds * fps + 1e-6))
    if last < 1:
        return np.zeros(0)
    frames = np.arange(1, last + 1)
    if max_times is not None and len(frames) > max_times:
        frames = frames[np.unique(np.rint(np.linspace(0, len(frames) - 1, max_times + 1)[1:]).astype(int))]
    return frames / fps


def select_times(times: np.ndarray, fps: float, horizon: Optional[float] = None, max_times: Optional[int] = 12) -> np.ndarray:
    """Indices of the stored evaluation `times` of a scene that eval_times(min(horizon, times[-1]), fps, max_times) lands
    on (nearest stored frame within half a frame; a v4 scene that stores 6 of 36 frames keeps the ones that match)."""
    times = np.asarray(times, np.float64)
    if not len(times):
        return np.zeros(0, int)
    end = float(times[-1]) if horizon is None else min(float(times[-1]), float(horizon))
    chosen = []
    for when in eval_times(end, fps, max_times):
        index = int(np.abs(times - when).argmin())
        if abs(float(times[index]) - when) <= 0.5 / fps + 1e-6 and index not in chosen:
            chosen.append(index)
    return np.array(sorted(chosen), int)


def _at(tracks: np.ndarray, visible: np.ndarray, track_times: np.ndarray, when: float, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """(displacement [N, 2] from frame 0, ok [N]) at `when` seconds (nearest tracked frame; nothing is ok when it is too far)."""
    frame = int(np.abs(track_times - when).argmin())
    ok = visible[frame] & visible[0]
    if abs(float(track_times[frame]) - when) > tolerance:
        ok = np.zeros_like(ok)
    return tracks[frame] - tracks[0], ok


def _rigid_angle(tracks: np.ndarray, visible: np.ndarray, chosen: np.ndarray, frame: int) -> tuple[float, bool]:
    """(degrees, ok) of the rigid fit of an object's own points between frame 0 and `frame`: at least ROT_MIN_POINTS
    seen at both, mean residual at most ROT_MAX_RESIDUAL px (float16 tracks: reliable for objects of >= 24 px)."""
    both = chosen & visible[0] & visible[frame]
    if int(both.sum()) < ROT_MIN_POINTS:
        return 0.0, False
    affine, residual = mp.fit_affine(tracks[0][both], tracks[frame][both], min_points=ROT_MIN_POINTS, model="rigid")
    if residual > ROT_MAX_RESIDUAL:
        return 0.0, False
    return float(np.degrees(mp._polar(affine[:, :2])[0])), True


def _read(tracks: np.ndarray, visible: np.ndarray, track_times: np.ndarray, times: np.ndarray, groups: list[np.ndarray], pool: np.ndarray,
          object_disp: Optional[np.ndarray] = None, object_ok: Optional[np.ndarray] = None, object_times: Optional[np.ndarray] = None,
          rotation: bool = False) -> dict:
    """Object and pool displacements relative to the camera at the evaluation times.  With object_disp [F', K, 2] /
    object_ok [F', K] at object_times [F'] (mask centroids) the objects are read from those, otherwise from their points.
    rotation=True adds rot [K, T] (degrees) / rot_ok [K, T]: the rigid-fit angle of every object's own points."""
    track_times = np.asarray(track_times, np.float64)
    step = float(np.median(np.diff(track_times))) if len(track_times) > 1 else 0.0
    count = len(times)
    out = {"object": np.zeros((len(groups), count, 2)), "object_ok": np.zeros((len(groups), count), bool), "camera": np.zeros((count, 2)),
           "pool": np.zeros((count, int(pool.sum()), 2)), "pool_ok": np.zeros((count, int(pool.sum())), bool), "visible_rate": np.zeros(count)}
    if rotation:
        out["rot"], out["rot_ok"] = np.zeros((len(groups), count)), np.zeros((len(groups), count), bool)
    for e, when in enumerate(times):
        disp, ok = _at(tracks, visible, track_times, float(when), 0.5 * step + 1e-6)
        seen = pool & ok
        camera = np.median(disp[seen], 0) if int(seen.sum()) >= 8 else np.zeros(2)
        out["camera"][e], out["pool"][e], out["pool_ok"][e] = camera, disp[pool] - camera, ok[pool]
        out["visible_rate"][e] = float(ok[pool].mean()) if pool.any() else 1.0
        if rotation and ok.any():
            frame = int(np.abs(track_times - float(when)).argmin())
            for k, chosen in enumerate(groups):
                out["rot"][k, e], out["rot_ok"][k, e] = _rigid_angle(tracks, visible, chosen, frame)
        if object_disp is not None:
            object_times = np.asarray(object_times, np.float64)
            frame = int(np.abs(object_times - when).argmin())
            gap = float(np.median(np.diff(object_times))) if len(object_times) > 1 else 0.0
            if abs(float(object_times[frame]) - when) <= 0.5 * gap + 1e-6:
                out["object"][:, e] = np.asarray(object_disp[frame], np.float64) - camera
                out["object_ok"][:, e] = np.asarray(object_ok[frame]).astype(bool)
                out["object"][~out["object_ok"][:, e], e] = 0.0
            continue
        for k, chosen in enumerate(groups):
            need = min(3, int(chosen.sum()))
            both = chosen & ok
            if need and int(both.sum()) >= need:
                out["object"][k, e], out["object_ok"][k, e] = np.median(disp[both], 0) - camera, True
    return out


def build_scene(points: np.ndarray, owner: np.ndarray, slots: Sequence[int], masks: np.ndarray, tracks: np.ndarray, visible: np.ndarray,
                track_times: np.ndarray, clip_seconds: float, *, fps: float = 7.0, max_times: Optional[int] = 6, object_disp: Optional[np.ndarray] = None,
                object_ok: Optional[np.ndarray] = None, object_times: Optional[np.ndarray] = None) -> dict:
    """The real side of the reward.  `owner` is the re-labelled owner of real_scene.measured_plan (cluster points carry
    their cluster's slot), `slots` / `masks` [K, H, W] the objects of the measured plan, tracks on the clip timeline.
    object_disp / object_ok / object_times: mask-centroid readings of the same objects on the clip (mode "mask"; a
    generated clip must then be scored with mask readings too).  max_times=None stores every frame (scene v5)."""
    points, owner = np.asarray(points, np.float64), np.asarray(owner)
    height, width = masks.shape[1:] if len(masks) else (256, 448)
    diagonal = float(np.hypot(height, width))
    times = eval_times(clip_seconds, fps, max_times)
    groups = [owner == slot for slot in slots]
    union = np.asarray(masks).astype(bool).any(0) if len(masks) else np.zeros((height, width), bool)
    rows, cols = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1), np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    pool = (owner == rs.DENSE_OWNER) & ~union[rows, cols]                       # free cells: on no listed object
    real = _read(np.asarray(tracks, np.float64), np.asarray(visible).astype(bool), track_times, times, groups, pool, object_disp, object_ok, object_times)
    size = np.linalg.norm(real["object"], axis=-1)
    return {"version": VERSION, "mode": "mask" if object_disp is not None else "points", "mask_bits": np.packbits(np.asarray(masks).astype(bool).reshape(-1)),
            "mask_shape": np.array([len(masks), height, width], np.int32),
            "points": points.astype(np.float32), "owner": owner.astype(np.int16), "slots": np.array(list(slots), np.int16),
            "pool": pool, "times": times.astype(np.float32), "diagonal": np.float32(diagonal), "fps": np.float32(fps),
            "real": real["object"].astype(np.float32), "real_ok": real["object_ok"], "moving": real["object_ok"] & (size >= MOVE_FRAC * diagonal),
            "static": real["object_ok"] & (size <= STILL_FRAC * diagonal), "camera_real": real["camera"].astype(np.float32),
            "pool_real": real["pool"].astype(np.float32), "pool_ok": real["pool_ok"], "visible_rate_real": real["visible_rate"].astype(np.float32)}


def cell_score(guess: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """a in [-1, 1] for displacement vectors [..., 2] (truth must be non-zero)."""
    g, r = np.linalg.norm(guess, axis=-1), np.linalg.norm(truth, axis=-1)
    rho = g / np.maximum(r, 1e-9)
    cosine = (guess * truth).sum(-1) / np.maximum(g * r, 1e-9)
    q = np.minimum(1.0, rho / 0.25)
    direction = cosine * q - (1.0 - q)
    magnitude = np.clip(2.0 - np.abs(np.log2(np.maximum(rho, 1e-9))), 0.0, 1.0)
    return direction * (0.5 + 0.5 * magnitude)


def _ramp_score(rho: np.ndarray, cosine: np.ndarray) -> np.ndarray:
    """v5 cell value from the length ratio and the direction cosine (see the module text)."""
    rho, cosine = np.asarray(rho, np.float64), np.asarray(cosine, np.float64)
    log = np.abs(np.log2(np.maximum(rho, 2.0 ** -6)))
    w = np.minimum(1.0, rho / 0.25)
    magnitude, over = np.clip(1.0 - 0.5 * log, 0.0, 1.0), np.clip(0.5 * log - 1.0, 0.0, 1.0)
    inner = np.where(cosine >= 0.0, cosine * (0.5 + 0.5 * magnitude) - 0.5 * over, cosine - 0.5 * over)
    return np.clip(w * inner + (1.0 - w) * -0.5, -1.0, 1.0)


def cell_score_v5(guess: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """s in [-1, 1] for displacement vectors [..., 2] (truth must be non-zero): no plateau, half a point per octave."""
    g, r = np.linalg.norm(guess, axis=-1), np.linalg.norm(truth, axis=-1)
    cosine = (guess * truth).sum(-1) / np.maximum(g * r, 1e-9)
    return _ramp_score(g / np.maximum(r, 1e-9), cosine)


def rot_score_v5(guess_deg: np.ndarray, truth_deg: np.ndarray) -> np.ndarray:
    """The same scale on rotation angles (degrees; truth non-zero): the ratio of the magnitudes, the sign as the direction."""
    g, r = np.abs(np.asarray(guess_deg, np.float64)), np.abs(np.asarray(truth_deg, np.float64))
    same = np.where(np.asarray(guess_deg) * np.asarray(truth_deg) >= 0.0, 1.0, -1.0)
    return _ramp_score(g / np.maximum(r, 1e-9), same)


def soft_off(distance: np.ndarray, diagonal: float) -> np.ndarray:
    """Still-cell violation in [0, 1]: 0 up to STILL (1 % D), 1 from 2 MOVE - STILL (5 % D) on."""
    still, move = STILL_FRAC * diagonal, MOVE_FRAC * diagonal
    return np.clip((np.asarray(distance, np.float64) - still) / (2.0 * move - still), 0.0, 1.0)


def scene_masks(scene: dict) -> np.ndarray:
    """[K, H, W] bool first-frame masks of the scene's objects (what the mask tracker is prompted with)."""
    count, height, width = (int(v) for v in scene["mask_shape"])
    return np.unpackbits(scene["mask_bits"])[: count * height * width].reshape(count, height, width).astype(bool)


def score(scene: dict, tracks: np.ndarray, visible: np.ndarray, track_times: np.ndarray, *, object_disp: Optional[np.ndarray] = None,
          object_ok: Optional[np.ndarray] = None, object_times: Optional[np.ndarray] = None, w_static: float = 1.0, w_identity: float = 0.5,
          w_segment: float = 0.3, version: str = "v5", horizon: Optional[float] = None, max_times: Optional[int] = 12, w_onset: float = 0.15) -> dict:
    """Reward of a generated clip whose frames were tracked from scene["points"] (and, for a scene in mode "mask",
    whose objects were propagated from scene_masks(scene): object_disp [F', K, 2], object_ok [F', K], object_times [F']).
    version "v5" (module text) or "v4" (bit-identical to the old reward); `horizon` truncates the scene first (seconds,
    snapped to the nearest frame: 1.714 means the 12th frame at 7 fps), max_times / w_onset are v5 only."""
    if horizon is not None and len(scene["times"]):
        cut = np.floor(horizon * float(scene["fps"]) + 0.5) / float(scene["fps"])
        if float(np.asarray(scene["times"])[-1]) > cut + 1e-6:
            scene = truncate_scene(scene, cut)
    if version == "v4":
        return _score_v4(scene, tracks, visible, track_times, object_disp=object_disp, object_ok=object_ok, object_times=object_times,
                         w_static=w_static, w_identity=w_identity, w_segment=w_segment)
    if version != "v5":
        raise ValueError(f"unknown reward version {version!r}")
    return _score_v5(scene, tracks, visible, track_times, object_disp=object_disp, object_ok=object_ok, object_times=object_times,
                     w_static=w_static, w_identity=w_identity, w_segment=w_segment, max_times=max_times, w_onset=w_onset)


def _score_v4(scene: dict, tracks: np.ndarray, visible: np.ndarray, track_times: np.ndarray, *, object_disp: Optional[np.ndarray] = None,
              object_ok: Optional[np.ndarray] = None, object_times: Optional[np.ndarray] = None, w_static: float = 1.0, w_identity: float = 0.5,
              w_segment: float = 0.3) -> dict:
    owner, times, diagonal = scene["owner"], np.asarray(scene["times"], np.float64), float(scene["diagonal"])
    move, still = MOVE_FRAC * diagonal, STILL_FRAC * diagonal
    groups = [owner == slot for slot in scene["slots"]]
    if str(scene.get("mode", "points")) == "mask" and object_disp is None:
        raise ValueError("this scene was measured with mask readings; score the generated clip with them too")
    gen = _read(np.asarray(tracks, np.float64), np.asarray(visible).astype(bool), track_times, times, groups, np.asarray(scene["pool"]).astype(bool),
                object_disp, object_ok, object_times)
    real, real_ok, moving, static = scene["real"].astype(np.float64), scene["real_ok"], scene["moving"], scene["static"]
    lost = real_ok & ~gen["object_ok"]                                           # visible in the real clip, lost in the generated one
    measured = moving & ~lost                                                    # plain readings of the same cells, for tables
    g_size, r_size = np.linalg.norm(gen["object"], axis=-1), np.linalg.norm(real, axis=-1)
    cosine = (gen["object"] * real).sum(-1) / np.maximum(g_size * r_size, 1e-9)
    follows = g_size >= 0.25 * r_size
    stats = {"moving_cells": int(moving.sum()), "lost_cells": int((moving & lost).sum()), "dir30": int((measured & follows & (cosine >= np.cos(np.radians(30)))).sum()),
             "dir60": int((measured & follows & (cosine >= 0.5)).sum()), "nearzero": int((measured & ~follows).sum()),
             "ratios": [round(float(v), 3) for v in (g_size / np.maximum(r_size, 1e-9))[measured]],
             "static_cells": int((real_ok & ~moving & ~lost).sum()),
             "static_moved": int(((real_ok & ~moving & ~lost) & (np.linalg.norm(gen["object"] - real, axis=-1) >= move)).sum())}
    per_object, actors = {}, []
    for k, slot in enumerate(scene["slots"]):
        cells = np.nonzero(moving[k])[0]
        if not len(cells):
            continue
        cumulative = np.where(lost[k, cells], -1.0, cell_score(gen["object"][k, cells], real[k, cells]))
        value = float(cumulative.mean())
        scored = np.nonzero(real_ok[k])[0]                                       # segments between consecutive measured times (from the first frame)
        previous_r, previous_g, previous_ok, segments = np.zeros(2), np.zeros(2), True, []
        for e in scored:
            r_seg = real[k, e] - previous_r
            if np.linalg.norm(r_seg) >= 0.5 * move:
                usable = gen["object_ok"][k, e] and previous_ok
                segments.append(float(cell_score(gen["object"][k, e] - previous_g, r_seg)) if usable else -1.0)
            previous_r, previous_g, previous_ok = real[k, e], gen["object"][k, e], bool(gen["object_ok"][k, e])
        if segments:
            value = (1.0 - w_segment) * value + w_segment * float(np.mean(segments))
        per_object[int(slot)] = round(value, 4)
        actors.append(value)
    quiet = real_ok & ~moving & ~lost                                            # objects that do not really move, where both sides are measured
    off = np.linalg.norm(gen["object"] - real, axis=-1) >= move
    shares = [float(off[k][quiet[k]].mean()) for k in range(len(groups)) if quiet[k].any()]
    leak, leak_mean = (max(shares), float(np.mean(shares))) if shares else (0.0, 0.0)
    actor = float(np.mean(actors)) if actors else 1.0 - 2.0 * leak
    quiet = scene["pool_ok"] & gen["pool_ok"] & (np.linalg.norm(scene["pool_real"], axis=-1) <= still)
    p_bg = float((np.linalg.norm(gen["pool"], axis=-1)[quiet] >= move).mean()) if quiet.any() else 0.0
    drift = np.linalg.norm(gen["camera"] - scene["camera_real"], axis=-1)
    p_cam = float(np.clip(drift.max() / (2.0 * move), 0.0, 1.0)) if len(drift) else 0.0
    p_static = 0.6 * leak + 0.2 * p_bg + 0.2 * p_cam
    lost_share = float(lost.sum() / max(1, int(real_ok.sum())))
    fade = float(np.clip(scene["visible_rate_real"] - gen["visible_rate"], 0.0, 1.0).max()) if len(times) else 0.0
    p_identity = float(np.clip(lost_share + fade, 0.0, 1.0))
    reward = float(np.clip(actor - w_static * p_static - w_identity * p_identity, -1.0, 1.0)) if len(times) else 0.0
    return {"reward": reward, "actor": actor, "has_mover": bool(actors), "movers": len(actors), "p_static": p_static, "static_leak": leak,
            "static_leak_mean": leak_mean,
            "p_background": p_bg, "p_camera": p_cam, "p_identity": p_identity, "lost": lost_share, "fade": fade, "per_object": per_object,
            "times": int(len(times)), "stats": stats, "version": VERSION_V4}


def _median(values: list) -> Optional[float]:
    return float(np.median(values)) if len(values) else None


def _score_v5(scene: dict, tracks: np.ndarray, visible: np.ndarray, track_times: np.ndarray, *, object_disp: Optional[np.ndarray] = None,
              object_ok: Optional[np.ndarray] = None, object_times: Optional[np.ndarray] = None, w_static: float = 1.0, w_identity: float = 0.5,
              w_segment: float = 0.3, max_times: Optional[int] = 12, w_onset: float = 0.15) -> dict:
    track_times = np.asarray(track_times, np.float64)
    fps, diagonal = float(scene["fps"]), float(scene["diagonal"])
    if len(scene["times"]) and len(track_times) and float(np.asarray(scene["times"])[-1]) > float(track_times[-1]) + 0.5 / fps:
        scene = truncate_scene(scene, float(track_times[-1]) + 0.5 / fps)                  # judged on the frames the clip has
    owner, times = scene["owner"], np.asarray(scene["times"], np.float64)
    move, still = MOVE_FRAC * diagonal, STILL_FRAC * diagonal
    groups = [owner == slot for slot in scene["slots"]]
    if str(scene.get("mode", "points")) == "mask" and object_disp is None:
        raise ValueError("this scene was measured with mask readings; score the generated clip with them too")
    use = select_times(times, fps, None, max_times)
    gen = _read(np.asarray(tracks, np.float64), np.asarray(visible).astype(bool), track_times, times, groups, np.asarray(scene["pool"]).astype(bool),
                object_disp, object_ok, object_times, rotation=True)
    real, real_ok, moving, static = scene["real"].astype(np.float64), scene["real_ok"].astype(bool), scene["moving"].astype(bool), scene["static"].astype(bool)
    moving_c = scene["moving_centroid"].astype(bool) if "moving_centroid" in scene else moving
    rot_known = scene["rot_known"].astype(bool) if "rot_known" in scene else np.zeros_like(moving)
    real_rot = scene["real_rot"].astype(np.float64) if "real_rot" in scene else np.zeros(moving.shape)
    lost = real_ok & ~gen["object_ok"]
    g_size, r_size = np.linalg.norm(gen["object"], axis=-1), np.linalg.norm(real, axis=-1)
    cosine = (gen["object"] * real).sum(-1) / np.maximum(g_size * r_size, 1e-9)
    sel = np.zeros(len(times), bool)
    sel[use] = True
    measured = moving_c & ~lost & sel                                            # translation cells read on both sides, for tables
    follows = g_size >= 0.25 * r_size
    quiet_obj = real_ok & ~moving & ~lost
    off = soft_off(np.linalg.norm(gen["object"] - real, axis=-1), diagonal)
    stats = {"moving_cells": int((moving_c & sel).sum()), "lost_cells": int((moving & lost & sel).sum()),
             "dir30": int((measured & follows & (cosine >= np.cos(np.radians(30)))).sum()), "dir60": int((measured & follows & (cosine >= 0.5)).sum()),
             "nearzero": int((measured & ~follows).sum()), "ratios": [round(float(v), 3) for v in (g_size / np.maximum(r_size, 1e-9))[measured]],
             "static_cells": int((quiet_obj & sel).sum()), "static_moved": int((quiet_obj & sel & (np.linalg.norm(gen["object"] - real, axis=-1) >= move)).sum()),
             "objects": [], "times_used": int(len(use)), "rot_cells": 0, "rot_unscored": 0, "movers_unscored": 0}
    never = (float(times[-1]) if len(times) else 0.0) + 1.0 / fps
    per_object, actors, log_ratios, angles, onsets = {}, [], [], [], []
    w_cells = 1.0 - w_segment - w_onset
    for k, slot in enumerate(scene["slots"]):
        cells = [e for e in use if moving[k, e]]
        info = {"slot": int(slot), "kind": "still", "cells": 0, "rot_cells": 0, "rot_unscored": 0, "lost": int((lost[k] & sel).sum()),
                "leak_off": float(off[k][quiet_obj[k] & sel].mean()) if (quiet_obj[k] & sel).any() else None,
                "abs_log2_ratio": None, "angle_deg": None, "onset_real": None, "onset_gen": None, "onset_err": None, "value": None}
        if not cells:
            stats["objects"].append(info)
            continue
        scores, kinds = [], set()
        for e in cells:
            if lost[k, e]:
                scores.append(-1.0)
                kinds.add("translation" if moving_c[k, e] else "rotation")
            elif moving_c[k, e]:
                scores.append(float(cell_score_v5(gen["object"][k, e], real[k, e])))
                kinds.add("translation")
                log_ratios.append(abs(float(np.log2(max(g_size[k, e] / max(r_size[k, e], 1e-9), 2.0 ** -6)))))
                angles.append(float(np.degrees(np.arccos(np.clip(cosine[k, e], -1.0, 1.0)))))
            elif rot_known[k, e] and abs(real_rot[k, e]) >= ROT_MIN_DEG and gen["rot_ok"][k, e]:
                scores.append(float(rot_score_v5(gen["rot"][k, e], real_rot[k, e])))
                kinds.add("rotation")
                info["rot_cells"] += 1
            else:
                info["rot_unscored"] += 1                                        # a point-mover without a scorable rotation
        stats["rot_cells"] += info["rot_cells"]
        stats["rot_unscored"] += info["rot_unscored"]
        info["kind"] = "mixed" if len(kinds) > 1 else (kinds.pop() if kinds else "unscored")
        info["cells"] = len(scores)
        own = [(e, c) for e, c in zip(cells, scores) if moving_c[k, e] and not lost[k, e]]
        if own:
            ratios = [abs(float(np.log2(max(g_size[k, e] / max(r_size[k, e], 1e-9), 2.0 ** -6)))) for e, _ in own]
            info["abs_log2_ratio"] = _median(ratios)
            info["angle_deg"] = _median([float(np.degrees(np.arccos(np.clip(cosine[k, e], -1.0, 1.0)))) for e, _ in own])
        previous_r, previous_g, previous_ok, segments = np.zeros(2), np.zeros(2), True, []
        for e in use:                                                            # segments between consecutive evaluation times (from the first frame)
            if not real_ok[k, e]:
                continue
            r_seg = real[k, e] - previous_r
            if np.linalg.norm(r_seg) >= 0.5 * move:
                usable = gen["object_ok"][k, e] and previous_ok
                segments.append(float(cell_score_v5(gen["object"][k, e] - previous_g, r_seg)) if usable else -1.0)
            previous_r, previous_g, previous_ok = real[k, e], gen["object"][k, e], bool(gen["object_ok"][k, e])
        t_real = next((float(times[e]) for e in range(len(times)) if real_ok[k, e] and r_size[k, e] >= ONSET_PX), None)
        t_gen = next((float(times[e]) for e in range(len(times)) if gen["object_ok"][k, e] and g_size[k, e] >= ONSET_PX), never)
        terms = []
        if scores:
            terms.append((w_cells, float(np.mean(scores))))
        if segments:
            terms.append((w_segment, float(np.mean(segments))))
        if t_real is not None:
            info.update(onset_real=t_real, onset_gen=t_gen, onset_err=abs(t_gen - t_real))
            terms.append((w_onset, 1.0 - 2.0 * min(1.0, abs(t_gen - t_real) / ONSET_S)))
            onsets.append(abs(t_gen - t_real))
        if not terms:
            stats["movers_unscored"] += 1
            stats["objects"].append(info)
            continue
        value = sum(w * v for w, v in terms) / sum(w for w, _ in terms)
        info["value"] = per_object[int(slot)] = round(value, 4)
        actors.append(value)
        stats["objects"].append(info)
    shares = [float(off[k][quiet_obj[k] & sel].mean()) for k in range(len(groups)) if (quiet_obj[k] & sel).any()]
    leak, leak_mean = (max(shares), float(np.mean(shares))) if shares else (0.0, 0.0)
    pool_quiet = scene["pool_ok"] & gen["pool_ok"] & (np.linalg.norm(scene["pool_real"], axis=-1) <= still) & sel[:, None]
    p_bg = float(soft_off(np.linalg.norm(gen["pool"], axis=-1), diagonal)[pool_quiet].mean()) if pool_quiet.any() else 0.0
    drift = np.linalg.norm(gen["camera"] - scene["camera_real"], axis=-1)[use]
    p_cam = float(np.clip(drift.max() / (2.0 * move), 0.0, 1.0)) if len(drift) else 0.0
    if actors:
        actor, p_static = float(np.mean(actors)), 0.6 * leak + 0.2 * p_bg + 0.2 * p_cam
    else:
        actor, p_static = 1.0 - 2.0 * leak, 0.2 * p_bg + 0.2 * p_cam
    lost_share = float((lost & sel).sum() / max(1, int((real_ok & sel).sum())))
    fade = float(np.clip(scene["visible_rate_real"] - gen["visible_rate"], 0.0, 1.0)[use].max()) if len(use) else 0.0
    p_identity = float(np.clip(lost_share + fade, 0.0, 1.0))
    reward_raw = float(actor - w_static * p_static - w_identity * p_identity) if len(use) else 0.0
    reward = float(np.clip(reward_raw, -2.0, 1.0)) if len(use) else 0.0
    return {"reward": reward, "reward_raw": reward_raw, "actor": actor, "has_mover": bool(actors), "movers": len(actors), "p_static": p_static,
            "static_leak": leak, "static_leak_mean": leak_mean, "p_background": p_bg, "p_camera": p_cam, "p_identity": p_identity, "lost": lost_share,
            "fade": fade, "per_object": per_object, "times": int(len(use)), "times_used": int(len(use)), "rot_cells": stats["rot_cells"],
            "rot_unscored": stats["rot_unscored"], "onset_err": float(np.mean(onsets)) if onsets else None, "abs_log2_ratio": _median(log_ratios),
            "angle_deg": _median(angles), "stats": stats, "version": VERSION}


def truncate_scene(scene: dict, seconds: float) -> dict:
    """The scene of a shorter clip: only the evaluation times up to `seconds` (a rollout that generates the first
    13 frames of a 37-frame clip is compared with those frames only).  Every per-time key of the scene is cut
    (SCENE_KT [K, T] incl. the relabel_v2 / scene-v5 keys, SCENE_T [T, ...])."""
    keep = np.asarray(scene["times"], np.float64) <= seconds + 1e-6
    out = dict(scene)
    out["times"] = scene["times"][keep]
    for key in SCENE_KT:
        if key in scene and np.ndim(scene[key]) >= 2 and scene[key].shape[1] == len(keep):
            out[key] = scene[key][:, keep]
    for key in SCENE_T:
        if key in scene and np.ndim(scene[key]) >= 1 and scene[key].shape[0] == len(keep):
            out[key] = scene[key][keep]
    return out


def save_scene(path, scene: dict) -> None:
    np.savez_compressed(path, **{k: (np.array(v) if not isinstance(v, np.ndarray) else v) for k, v in scene.items()})


def load_scene(path) -> dict:
    data = np.load(path, allow_pickle=False)
    scene = {k: data[k] for k in data.files}
    scene["version"] = str(scene["version"])
    scene["mode"] = str(scene["mode"]) if "mode" in scene else "points"
    return scene
