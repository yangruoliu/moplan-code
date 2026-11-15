#!/usr/bin/env python
"""Physics reward and plain motion readings of generated clips (infer_stage2_cond_v2.py output).

For every <arm>__<sid>[__s<k>].npy in --generated whose clip has a scene (<scenes>/<sid>.npz, written by
cache_stage2_cond_v2.py, or scenes_v5/<sid>.npz written by rescene_v5.py for reward v5): the generated frames are measured
with the instrument the real clip was measured with (scripts/control/measure.py: SAM 2.1 mask centroids for the objects,
CoTracker3 points for background and camera), then reward.score.

  track   (GPU) writes <name><suffix> and <name>.measure.npz (the raw readings) next to the clip, skips finished ones
  rescore (CPU) the rewards again from <name>.measure.npz against the current scenes / reward version
  table   (CPU) per arm: reward, R_actor, direction within 60 deg / near-zero / length ratio on really moving objects,
          still objects that moved, background, camera, identity (v5: reward_raw, |log2 ratio|, angle, onset error, leak);
          paired differences between arms over the same (clip, seed) with a bootstrap over clips; optional split by a
          manifest column

Reward versions (--reward-version, default v5): v4 writes <name>.reward.json, v5 <name>.reward_v5.json (--suffix overrides);
both may sit in one folder.  --scene-of strip-at scores a clip named <sid>@<anything> (RL rollouts pred__<sid>@a<j>, response
sweeps <sid>@<condition>) against the cache scene <sid>.npz; --horizon cuts the scene to the first seconds, --max-times is
the number of evaluation times of v5.

usage: score_generated.py track --generated DIR --scenes DIR [--cotracker-repo R] [--checkpoint C] [--sam2-checkpoint S]
                                [--shard-index i --num-shards n] [--reward-version v5] [--suffix S] [--horizon H] [--max-times 12]
                                [--scene-of sid|strip-at]
       (--cotracker-repo / --checkpoint / --sam2-checkpoint default to $COTRACKER_REPO / $COTRACKER_CKPT / $SAM2_CHECKPOINT)
       score_generated.py rescore --generated DIR --scenes DIR [the same reward flags]
       score_generated.py table --generated DIR [--pairs pred-null,pred-zero] [--manifest M.jsonl --category scenario] [--out JSON]
                                [--reward-version v5 | --suffix S]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.control import reward as rw  # noqa: E402


SUFFIX = {"v4": ".reward.json", "v5": ".reward_v5.json"}


def parse_name(stem: str) -> tuple[str, str, int]:
    arm, rest = stem.split("__", 1)
    sid, seed = rest, 0
    if "__s" in rest and rest.rsplit("__s", 1)[1].isdigit():
        sid, number = rest.rsplit("__s", 1)
        seed = int(number)
    return arm, sid, seed


def scene_id(sid: str, scene_of: str = "sid") -> str:
    """The cache scene a clip is scored against: its own id, or ("strip-at") the part before "@" (RL rollouts are
    named <sid>@a<j>, response sweeps <sid>@<condition>; their cache scene is <sid>.npz)."""
    return sid.split("@", 1)[0] if scene_of == "strip-at" else sid


def reward_suffix(a) -> str:
    return a.suffix or SUFFIX[a.reward_version]


def score_measure(scene: dict, seen, *, version: str = "v5", horizon: float | None = None, max_times: int | None = 12) -> dict:
    """reward.score from a stored measurement (the arrays of <name>.measure.npz)."""
    times = np.asarray(seen["times"], np.float64)
    return rw.score(scene, np.asarray(seen["tracks"], np.float64), np.asarray(seen["visible"]).astype(bool), times, object_disp=seen["object_disp"],
                    object_ok=seen["object_ok"], object_times=times, version=version, horizon=horizon, max_times=max_times)


def track(a) -> None:
    from scripts.control.measure import Instrument

    generated, scenes, suffix = Path(a.generated), Path(a.scenes), reward_suffix(a)
    files = sorted(generated.glob("*__*.npy"))[a.shard_index:: a.num_shards]
    masks = {"sam2_checkpoint": a.sam2_checkpoint} if a.sam2_checkpoint else {}
    instrument = Instrument(a.cotracker_repo, a.checkpoint, **masks, scale=a.track_scale, vis_threshold=a.vis_threshold)
    done = missing = 0
    for count, path in enumerate(files):
        target = path.with_suffix(suffix)
        if target.is_file() and not a.overwrite:
            continue
        arm, sid, seed = parse_name(path.stem)
        scene_path = scenes / f"{scene_id(sid, a.scene_of)}.npz"
        if not scene_path.is_file():
            missing += 1
            continue
        scene = rw.load_scene(scene_path)
        frames = np.load(path)
        times = np.arange(len(frames)) / float(scene["fps"])
        seen = instrument.measure(frames, scene["points"], rw.scene_masks(scene))
        result = rw.score(scene, seen["tracks"], seen["visible"], times, object_disp=seen["object_disp"], object_ok=seen["object_ok"], object_times=times,
                          version=a.reward_version, horizon=a.horizon, max_times=a.max_times)
        result.update(arm=arm, sample_id=sid, seed_index=seed, frames=int(len(frames)))
        np.savez_compressed(path.with_suffix(".measure.npz"), object_disp=seen["object_disp"], object_ok=seen["object_ok"], times=times.astype(np.float32),
                            slots=scene["slots"], tracks=seen["tracks"].astype(np.float16), visible=seen["visible"])
        target.write_text(json.dumps(result) + "\n")
        done += 1
        if (count + 1) % 50 == 0:
            print(f"[score] {count + 1}/{len(files)}", flush=True)
    print(f"SCORE_DONE shard {a.shard_index}/{a.num_shards}: scored {done} ({a.reward_version} -> {suffix}), no scene {missing}, of {len(files)}")


def rescore(a) -> None:
    """Rewards again from the stored measurements (<name>.measure.npz) against the CURRENT scene labels / reward
    version: no tracking, no GPU.  Used after a label-contract change of the scenes (relabel_v2) or a reward change
    (v5 on scenes_v5) so that every panel is on one contract."""
    generated, scenes, suffix = Path(a.generated), Path(a.scenes), reward_suffix(a)
    done = missing = 0
    for path in sorted(generated.glob("*__*.measure.npz")):
        name = path.name[: -len(".measure.npz")]
        arm, sid, seed = parse_name(name)
        scene_path = scenes / f"{scene_id(sid, a.scene_of)}.npz"
        if not scene_path.is_file():
            missing += 1
            continue
        scene, seen = rw.load_scene(scene_path), np.load(path)
        result = score_measure(scene, seen, version=a.reward_version, horizon=a.horizon, max_times=a.max_times)
        result.update(arm=arm, sample_id=sid, seed_index=seed, frames=int(len(seen["times"])), labels_version=int(scene.get("labels_version", 1)))
        (generated / f"{name}{suffix}").write_text(json.dumps(result) + "\n")
        done += 1
    print(f"RESCORE_DONE: {done} clips ({a.reward_version} -> {suffix}), no scene {missing}")


def summarise(rows: list[dict]) -> dict:
    """Per-arm summary: clip means for the reward terms, cell-pooled per clip then averaged for the motion readings;
    the v5 readings (reward_raw, |log2 ratio|, angle, onset, leak, rot_cells, times_used) are None on v4 rows."""
    mean = lambda values: float(np.mean(values)) if len(values) else None      # noqa: E731
    median = lambda values: float(np.median(values)) if len(values) else None  # noqa: E731
    movers = [r for r in rows if r["stats"]["moving_cells"] > 0]
    per = lambda key: [r["stats"][key] / r["stats"]["moving_cells"] for r in movers]      # noqa: E731
    ratios = [float(np.median(r["stats"]["ratios"])) for r in movers if r["stats"]["ratios"]]
    still = [r["stats"]["static_moved"] / r["stats"]["static_cells"] for r in rows if r["stats"]["static_cells"] > 0]
    present = lambda key: [r[key] for r in rows if r.get(key) is not None]     # noqa: E731
    return {"n": len(rows), "n_mover": len(movers), "reward": mean([r["reward"] for r in rows]), "reward_mover": mean([r["reward"] for r in movers]),
            "actor": mean([r["actor"] for r in movers]), "dir30": mean(per("dir30")), "dir60": mean(per("dir60")), "nearzero": mean(per("nearzero")),
            "lost": mean(per("lost_cells")), "ratio": float(np.median(ratios)) if ratios else None, "static_moved": mean(still),
            "background": mean([r["p_background"] for r in rows]), "camera": mean([r["p_camera"] for r in rows]), "identity": mean([r["p_identity"] for r in rows]),
            "reward_raw": mean(present("reward_raw")), "log2ratio": median(present("abs_log2_ratio")), "angle": median(present("angle_deg")),
            "onset": mean(present("onset_err")), "leak": mean([r["static_leak"] for r in rows]),
            "rot_cells": int(sum(r.get("rot_cells", 0) for r in rows)), "times_used": mean(present("times_used"))}


def paired(rows_a: dict, rows_b: dict, key, groups: dict, draws: int = 2000, seed: int = 0) -> dict | None:
    """Mean of key(a) - key(b) over shared (clip, seed), bootstrap over clips (a clip's seeds stay together)."""
    shared = sorted(set(rows_a) & set(rows_b))
    by_clip = defaultdict(list)
    for item in shared:
        value_a, value_b = key(rows_a[item]), key(rows_b[item])
        if value_a is not None and value_b is not None:
            by_clip[groups.get(item[0], item[0])].append(value_a - value_b)
    if not by_clip:
        return None
    clips = [float(np.mean(v)) for v in by_clip.values()]
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(clips, len(clips))) for _ in range(draws)])
    return {"n": len(clips), "diff": float(np.mean(clips)), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
            "wins": float(np.mean(np.array(clips) > 0))}


def table(a) -> None:
    rows = defaultdict(dict)                                   # arm -> (sid, seed) -> result
    for path in sorted(Path(a.generated).glob(f"*{reward_suffix(a)}")):
        result = json.loads(path.read_text())
        rows[result["arm"]][(result["sample_id"], result["seed_index"])] = result
    category, group = {}, {}
    if a.manifest:
        for line in Path(a.manifest).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                category[row["sample_id"]] = str(row.get(a.category, ""))
                group[row["sample_id"]] = str(row.get(a.group, row["sample_id"]))
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"                       # noqa: E731
    report = {"arms": {}, "pairs": {}, "by_category": {}}
    print(f"{'arm':12s} {'n':>4s} {'mov':>4s} {'reward':>6s} {'R|mov':>6s} {'actor':>6s} {'dir30':>6s} {'dir60':>6s} {'near0':>6s} {'lost':>6s} {'ratio':>6s} "
          f"{'still+':>6s} {'bg':>6s} {'cam':>6s} {'ident':>6s} {'raw':>6s} {'l2r':>6s} {'angle':>6s} {'onset':>6s} {'leak':>6s} {'times':>6s}")
    for arm in sorted(rows):
        report["arms"][arm] = s = summarise(list(rows[arm].values()))
        print(f"{arm:12s} {s['n']:4d} {s['n_mover']:4d} " + " ".join(fmt(s[k]) for k in ("reward", "reward_mover", "actor", "dir30", "dir60", "nearzero", "lost",
                                                                                         "ratio", "static_moved", "background", "camera", "identity", "reward_raw",
                                                                                         "log2ratio", "angle", "onset", "leak", "times_used")))
    if category:
        for name in sorted(set(category.values())):
            for arm in sorted(rows):
                chosen = [r for (sid, _), r in rows[arm].items() if category.get(sid) == name]
                if chosen:
                    report["by_category"][f"{name}/{arm}"] = s = summarise(chosen)
                    print(f"  {name[:18]:18s} {arm:10s} n {s['n']:4d} mov {s['n_mover']:4d} reward {fmt(s['reward'])} dir60 {fmt(s['dir60'])} near0 {fmt(s['nearzero'])} "
                          f"ratio {fmt(s['ratio'])} still+ {fmt(s['static_moved'])}")
    readings = {"reward": lambda r: r["reward"], "actor": lambda r: r["actor"] if r["has_mover"] else None,
                "dir60": lambda r: r["stats"]["dir60"] / r["stats"]["moving_cells"] if r["stats"]["moving_cells"] else None,
                "nearzero": lambda r: r["stats"]["nearzero"] / r["stats"]["moving_cells"] if r["stats"]["moving_cells"] else None,
                "static_moved": lambda r: r["stats"]["static_moved"] / r["stats"]["static_cells"] if r["stats"]["static_cells"] else None,
                "identity": lambda r: r["p_identity"], "background": lambda r: r["p_background"],        # penalties: lower is better
                "log2ratio": lambda r: r.get("abs_log2_ratio"), "angle": lambda r: r.get("angle_deg"), "onset": lambda r: r.get("onset_err"),
                "leak": lambda r: r["static_leak"]}
    for pair in [v for v in a.pairs.split(",") if v]:
        first, second = pair.split("-")
        if first not in rows or second not in rows:
            continue
        report["pairs"][pair] = {}
        for name, key in readings.items():
            result = paired(rows[first], rows[second], key, group)
            report["pairs"][pair][name] = result
            if result:
                print(f"  {pair:14s} {name:13s} {result['diff']:+.3f} [{result['lo']:+.3f}, {result['hi']:+.3f}] n {result['n']} wins {result['wins']:.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("track")
    t.add_argument("--generated", required=True)
    t.add_argument("--scenes", required=True)
    t.add_argument("--cotracker-repo", default=os.environ.get("COTRACKER_REPO", ""), help="co-tracker git checkout (default $COTRACKER_REPO)")
    t.add_argument("--checkpoint", default=os.environ.get("COTRACKER_CKPT", ""), help="CoTracker3 scaled_offline.pth (default $COTRACKER_CKPT)")
    t.add_argument("--sam2-checkpoint", default=os.environ.get("SAM2_CHECKPOINT", ""), help="SAM 2.1 hiera-large .pt (default $SAM2_CHECKPOINT)")
    t.add_argument("--track-scale", type=float, default=2.0)
    t.add_argument("--vis-threshold", type=float, default=0.5)
    t.add_argument("--shard-index", type=int, default=0)
    t.add_argument("--num-shards", type=int, default=1)
    t.add_argument("--overwrite", action="store_true")
    t.set_defaults(run=track)
    r = sub.add_parser("rescore")
    r.add_argument("--generated", required=True)
    r.add_argument("--scenes", required=True)
    r.set_defaults(run=rescore)
    for q in (t, r):
        q.add_argument("--reward-version", choices=("v4", "v5"), default="v5")
        q.add_argument("--suffix", default="", help="reward file suffix (default .reward.json for v4, .reward_v5.json for v5)")
        q.add_argument("--horizon", type=float, default=None, help="score only the first seconds of the scene")
        q.add_argument("--max-times", type=int, default=12, help="v5: evaluation times chosen from the stored frames")
        q.add_argument("--scene-of", choices=("sid", "strip-at"), default="sid", help="strip-at: clip <sid>@<x> is scored against scene <sid>")
    s = sub.add_parser("table")
    s.add_argument("--generated", required=True)
    s.add_argument("--reward-version", choices=("v4", "v5"), default="v5")
    s.add_argument("--suffix", default="", help="reward files to read (default by --reward-version)")
    s.add_argument("--pairs", default="pred-null,pred-zero,real-null,real-pred")
    s.add_argument("--manifest", default="")
    s.add_argument("--category", default="scenario")
    s.add_argument("--group", default="trial_id", help="manifest column that identifies clips of the same scene (bootstrap unit)")
    s.add_argument("--out", default="")
    s.set_defaults(run=table)
    a = p.parse_args()
    a.run(a)


if __name__ == "__main__":
    main()
