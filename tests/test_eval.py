"""CPU tests of the evaluation and aggregation scripts on tiny synthetic inputs (no GPU, no model weights, no benchmark data).

Run: PYTHONPATH=. python -m pytest tests/test_eval.py
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "motionbench"))

# sha256 of the Morpheus generator-prompt table built from the shipped inputs (the table of the reference runs)
MORPHEUS_PROMPT_TABLE_SHA256 = "8b56ae45bd27c2b743ddbeedcca280b86bc4d77fdb4cbc4199dfd1d02d98eeee"


def run(*args: str, cwd: Path | None = None) -> str:
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    res = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=cwd, env=env)
    assert res.returncode == 0, res.stderr[-2000:]
    return res.stdout


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# ------------------------------------------------------------------------------------------------ Physics-IQ aggregation
def test_agg_runs_v_mean_std(tmp_path):
    vals = [40.0, 38.5, 39.25]
    names = []
    for k, v in enumerate(vals, 1):
        name = f"m-bpp-run_0{k}"
        names.append(name)
        off = {"verified_score": v, "score_spatiotemporal_view": 0.1 * k, "score_spatial_view": 0.2, "score_weighted_spatial_view": 0.3,
               "score_mse_view": 0.4}
        (tmp_path / f"{name}__verified.json").write_text(json.dumps({"status": "ok", "official": off, "files": {"csv": "x.csv"}}))
    run(str(ROOT / "scripts/s2eval_verified/agg_runs_v.py"), "--results-dir", str(tmp_path), "--mode", "verified", "--label", "m-bpp", *names)
    out = json.loads((tmp_path / "agg__m-bpp__verified.json").read_text())
    assert out["n"] == 3 and out["per_run"] == vals
    assert out["mean"] == pytest.approx(statistics.mean(vals))
    assert out["std_ddof1"] == pytest.approx(statistics.stdev(vals))
    assert out["score_spatiotemporal_view_x100"]["mean"] == pytest.approx(20.0)


def test_piq_summary_has_no_machine_paths_and_agg_finds_the_csvs(tmp_path):
    results, work = tmp_path / "results", tmp_path / "work"
    (work / "code").mkdir(parents=True)
    names = []
    for k, v in enumerate([40.0, 39.0], 1):
        name = f"m-bpp-run_0{k}"
        names.append(name)
        res = results / f"{name}__verified"
        res.mkdir(parents=True)
        (res / f"{name}.csv").write_text("scenario\n")
        metrics = {"final_score_view": v / 100, "score_spatiotemporal_view": 0.1, "score_spatial_view": 0.2,
                   "score_weighted_spatial_view": 0.3, "score_mse_view": 0.4, "final_score_orig": 0.3,
                   "final_score_origround": 0.3}
        (res / f"{name}_metrics.json").write_text(json.dumps(metrics))
        log = work / f"{name}.log"
        log.write_text(f"Physics-IQ score (verified) for {name}: {v}\n")
        (work / "pre.json").write_text(json.dumps({"run_dir": "x", "n_files": 198}))
        run(str(ROOT / "scripts/eval/piq_summarize_run.py"), "--mode", "verified", "--name", name, "--work", str(work),
            "--log", str(log), "--precheck", str(work / "pre.json"), "--code", str(work / "code"),
            "--descriptions", str(work / "descriptions_base.csv"), "--bench_data", str(work / "physics-IQ-benchmark-verified"),
            "--run_dir", str(tmp_path / "gen" / name), "--rc", "0", "--t_start", "s", "--t_end", "e", "--wall_s", "1",
            "--nproc", "1", "--cpus", "all", "--results_dir", str(res), "--out", str(results / f"{name}__verified.json"))
        text = (results / f"{name}__verified.json").read_text()
        summary = json.loads(text)
        assert str(tmp_path) not in text and "host" not in summary["timing"]
        assert summary["files"]["csv"] == f"{name}__verified/{name}.csv" and summary["run_dir"] == name
    agg = tmp_path / "piq_code" / "physiq"                       # stand-in for the official aggregator: checks its CSVs
    agg.mkdir(parents=True)
    (agg / "aggregate_runs_from_csvs.py").write_text("import os, sys\nprint(all(os.path.isfile(p) for p in sys.argv[1:-2]))\n")
    run(str(ROOT / "scripts/s2eval_verified/agg_runs_v.py"), "--results-dir", str(results), "--mode", "verified",
        "--label", "m-bpp", "--piq-python", sys.executable, "--piq-code", str(tmp_path / "piq_code"), *names)
    out = json.loads((results / "agg__m-bpp__verified.json").read_text())
    assert out["official_aggregator"]["rc"] == 0 and out["official_aggregator"]["stdout"].strip() == "True"
    assert str(tmp_path) not in json.dumps(out)
    assert out["mean"] == pytest.approx(39.5)


def test_agg_runs_v_refuses_failed_run(tmp_path):
    (tmp_path / "m-op-run_01__orig_ec2c611.json").write_text(json.dumps({"status": "scorer_exit_1"}))
    res = subprocess.run([sys.executable, str(ROOT / "scripts/s2eval_verified/agg_runs_v.py"), "--results-dir", str(tmp_path),
                          "--mode", "orig_ec2c611", "--label", "m", "m-op-run_01"], capture_output=True, text=True)
    assert res.returncode != 0 and "not ok" in res.stderr


def test_piq_report_values_rounding_and_original_source(tmp_path):
    from scripts.eval import piq_report_values as R
    assert "%.2f" % 2.675 == "2.67" and R.round2(2.675) == "2.68"        # the float lies just below 2.675
    assert R.round2(statistics.mean([2.67, 2.68])) == "2.68" and R.round2(1.234999) == "1.23" and R.round2(7.0) == "7.00"
    for label, ver, orig in (("m", 5.0, 3.005), ("b", 4.0, 2.675)):
        (tmp_path / f"agg__{label}-bpp__verified.json").write_text(json.dumps({"mean": ver, "n": 4}))
        (tmp_path / f"agg__{label}-op__orig_ec2c611.json").write_text(json.dumps({"mean": orig, "n": 4}))
    cols = R.columns(tmp_path, "b", None, None)                    # no breakdown: mean of the printed per-run scores
    assert cols["Verified"][0] == 4.0 and cols["Original"][0] == 2.675 and "printed" in cols["Original"][2]
    cell = {"final_mean": 3.0049, "final_runs": [1, 2, 3, 4], "base_mean": 2.6751, "base_runs": [1, 2, 3, 4]}
    bd = {"results": {"original_op_ec2c611": {"rows": {"Overall": {"metrics": {"final_score_orig_ec2c611": cell}}}}}}
    (tmp_path / "breakdown_m_vs_b.json").write_text(json.dumps(bd))
    assert R.columns(tmp_path, "b", "base", bd)["Original"][0] == 2.6751       # the unrounded per-run mean
    out = run(str(ROOT / "scripts/eval/piq_report_values.py"), "--results-dir", str(tmp_path), "--label", "m", "--base", "b")
    assert "3.00" in out and "2.68" in out and "5.00" in out


# ------------------------------------------------------------------------------------------------ Morpheus
def morpheus_tree(root: Path, model: str, scores: dict) -> Path:
    """scores: (prompt_type, experiment, seed) -> (physical, dynamical, discarded)"""
    base = root / "morpheus/results/single_frame_conditioning" / model
    for (pt, exp, k), (phys, dyn, disc) in scores.items():
        d = base / pt / exp / str(k)
        d.mkdir(parents=True, exist_ok=True)
        (d / "combined_scores.json").write_text(json.dumps({"model": model, "experiment": exp, "prompt_type": pt,
                                                             "physical_score": phys, "statistical_score": dyn}))
        (d / "filtering_info.json").write_text(json.dumps({"trajectory_filtering": {"discard_details": {"0": {"is_discarded": disc}}}}))
    return base


def test_morph_aggregate_discarded_zero_and_best_prompt(tmp_path):
    scores = {("plain", "falling_ball", 0): (0.8, 0.6, False), ("plain", "spring", 0): (0.4, 0.2, True),
              ("enhanced", "falling_ball", 0): (0.9, 0.7, False), ("enhanced", "spring", 0): (0.5, 0.5, False)}
    base = morpheus_tree(tmp_path, "MP-a", scores)
    run(str(ROOT / "scripts/bench/morph_aggregate_br.py"), str(base))
    agg = json.loads((base / "aggregate.json").read_text())["MP-a"]
    plain = agg["plain"]["per_video_zeroed"]
    assert plain["physinv"] == pytest.approx(round((0.8 + 0.0) / 2, 4))      # discarded video counts 0
    assert plain["total"] == pytest.approx(round(((0.6 + 0.0) / 2 + (0.8 + 0.0) / 2) / 2, 4))
    assert agg["best_prompt_by_total"] == "enhanced"
    assert agg["plain"]["discard_rate"] == pytest.approx(0.5)


def test_morph_worker_check_requires_every_score(tmp_path):
    res, results, mm = tmp_path / "resized", tmp_path / "results", "MP-a"
    cond = res / "single_frame_conditioning" / mm
    mapping = [{"source": "x.mp4", "experiment": exp, "prompt_type": pt, "seed_index": k, "resized": str(cond / pt / exp / str(k))}
               for exp in ("spring", "non_holonomic_pendulum") for pt in ("plain", "enhanced") for k in range(2)]
    cond.mkdir(parents=True)
    (cond / "RESIZE_MAP.json").write_text(json.dumps(mapping))
    scored = [m for m in mapping if m["experiment"] == "spring"]           # the excluded experiment needs no score
    for m in scored[:-1]:
        d = results / "single_frame_conditioning" / mm / m["prompt_type"] / m["experiment"] / str(m["seed_index"])
        d.mkdir(parents=True)
        (d / "combined_scores.json").write_text("{}")
    cmd = [sys.executable, str(ROOT / "scripts/s2eval/morph_worker.py"), "check", "--resized", str(res), "--results", str(results),
           "--model", mm]
    r = subprocess.run(cmd + ["--expect", "4"], capture_output=True, text=True)
    assert r.returncode == 1 and "1 of 4 videos have no combined_scores.json" in r.stderr and "enhanced/spring/1" in r.stdout
    last = scored[-1]
    d = results / "single_frame_conditioning" / mm / last["prompt_type"] / last["experiment"] / str(last["seed_index"])
    d.mkdir(parents=True)
    (d / "combined_scores.json").write_text("{}")
    assert json.loads(run(*cmd[1:], "--expect", "4"))["scored"] == 4
    r = subprocess.run(cmd + ["--expect", "6"], capture_output=True, text=True)
    assert r.returncode == 1 and "expected 6" in r.stderr


def test_osp_paired_and_breakdown_morpheus(tmp_path):
    exps = ["falling_ball", "spring", "projectile"]
    sa = {("plain", e, k): (0.5 + 0.1 * i, 0.4, False) for i, e in enumerate(exps) for k in range(3)}
    sb = {("plain", e, k): (0.3 + 0.1 * i, 0.4, False) for i, e in enumerate(exps) for k in range(3)}
    for sc in (sa, sb):                                         # a worse enhanced prompt: the plain one is reported
        sc.update({("enhanced", e, k): (0.1, 0.1, False) for e in exps for k in range(3)})
    for model, sc in (("MP-a", sa), ("MP-b", sb)):
        run(str(ROOT / "scripts/bench/morph_aggregate_br.py"), str(morpheus_tree(tmp_path, model, sc)))
    run(str(ROOT / "scripts/s2eval/osp_paired.py"), "morpheus", "--root", str(tmp_path), "--a", "a", "--b", "b", "--n", "200")
    out = json.loads((tmp_path / "paired/paired_morpheus_a_vs_b.json").read_text())
    assert out["n"] == 9 and out["prompt_a"] == out["prompt_b"] == "plain"
    assert out["metrics"]["physinv"]["diff"] == pytest.approx(0.2)
    assert out["metrics"]["dyn"]["diff"] == pytest.approx(0.0)
    lo, hi = out["metrics"]["physinv"]["ci_videos"]
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)            # constant difference -> degenerate interval
    run(str(ROOT / "scripts/s2eval/osp_breakdown.py"), "morpheus", "--root", str(tmp_path), "--a", "a", "--b", "b")
    bd = json.loads((tmp_path / "paired/breakdown_morpheus_a_vs_b.json").read_text())
    assert bd["rows"]["overall"]["physinv"]["diff"] == pytest.approx(0.2)
    assert set(bd["rows"]) == {"overall", "Falling", "Spring", "Projectile"}


# ------------------------------------------------------------------------------------------------ PisaBench
def test_osp_paired_pisabench(tmp_path):
    rng = np.random.default_rng(1)
    la, lb = rng.uniform(0.05, 0.3, 20), rng.uniform(0.05, 0.3, 20)
    for model, l2 in (("a", la), ("b", lb)):
        rows = [{"config": {"pd_mask": f"/x/{model}/s0/real/{k:04d}/mask.npz"}, "l2": float(l2[k]), "chamfer_distance": float(2 * l2[k]),
                 "iou": float(1 - l2[k])} for k in range(20)]
        write_jsonl(tmp_path / f"pisabench/official/{model}/s0/results/per_video.jsonl", rows)
    run(str(ROOT / "scripts/s2eval/osp_paired.py"), "pisabench", "--root", str(tmp_path), "--a", "a", "--b", "b", "--n", "500")
    out = json.loads((tmp_path / "paired/paired_pisabench_a_vs_b.json").read_text())
    assert out["n"] == 20
    m = out["metrics"]["l2"]
    assert m["diff"] == pytest.approx(float(np.mean(la - lb)))
    assert m["ci_videos"][0] <= m["diff"] <= m["ci_videos"][1]
    assert out["metrics"]["iou"]["higher_is_better"] is True and m["higher_is_better"] is False


def test_pisa_fall_speed_helpers():
    from scripts.s2eval import pisa_fall_speed as pf
    t = np.arange(5, dtype=float)
    d = np.array([0.0, 1.0, 3.0, np.nan, 9.0])
    assert pf.crossing(t, d, 2.0) == pytest.approx(1.5)
    assert pf.crossing(t, d, 100.0) is None
    m = np.zeros((2, 4, 4), bool)
    m[0, 1, :] = True
    ys = pf.centroid_y(m)
    assert ys[0] == pytest.approx(1.0) and math.isnan(ys[1])


def test_pisa_export_geometry():
    from scripts.bench import pisa_official_export as pe
    assert pe.SQUARE_COLUMNS == (98, 350)
    frames = [np.full((256, 448, 3), k, np.uint8) for k in range(8)]          # 8 frames @ 7 fps
    out = pe.retime(frames, 7.0, 16.0, "nearest")
    assert len(out) == 17 and [int(f[0, 0, 0]) for f in out[:4]] == [0, 0, 1, 1]
    assert pe.scored_frames(17, 16.0, gt_frames=60) == 8                      # int(k * 120 / 16) < 60
    sq = pe.to_square(frames[3], 256)
    assert sq.shape == (256, 256, 3)


# ------------------------------------------------------------------------------------------------ RigidBench
def test_osp_breakdown_rigid(tmp_path):
    metrics = ["iou", "l2", "chamfer", "ate", "ate3d", "si_mse", "ssim", "lpips", "iddrift", "bgdrift"]
    for model, off in (("a", 0.0), ("b", 0.1)):
        samples = [{"sample_id": f"{k:05d}", "task_type": "free_fall" if k < 5 else "collision",
                    **{m: 0.2 + 0.01 * k + off for m in metrics}} for k in range(10)]
        agg = {m: float(np.mean([s[m] for s in samples])) for m in metrics}
        p = tmp_path / f"rigidbench/runs/{model}/results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"samples": samples, "aggregated": agg}))
    run(str(ROOT / "scripts/s2eval/osp_breakdown.py"), "rigid", "--root", str(tmp_path), "--a", "a", "--b", "b")
    out = json.loads((tmp_path / "paired/breakdown_rigid_a_vs_b.json").read_text())
    assert out["n"] == 10 and set(out["rows"]) == {"overall", "collision", "free_fall"}
    l2 = out["rows"]["overall"]["l2"]
    assert l2["diff"] == pytest.approx(-0.1) and l2["favourable"] is True and l2["significant"] is True
    x, y = out["check_overall_mean_vs_aggregated_a"]["l2"]
    assert x == pytest.approx(y)


# ------------------------------------------------------------------------------------------------ generation
def test_embeds_cover_full_id_list(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.eval import embeds_cover as EC
    cache = tmp_path / "cache"
    cache.mkdir()
    for sid, prompt in (("a", "p1"), ("b", "p2"), ("c", "p1")):
        torch.save({"prompt": prompt, "cond": torch.zeros(3)}, cache / f"{sid}.pt")
    ids, emb, neg, table = tmp_path / "ids.txt", tmp_path / "e.pt", tmp_path / "neg.txt", tmp_path / "table.json"
    ids.write_text("a\nb\nc\n")
    args = ["--embeds", str(emb), "--cache-dir", str(cache), "--ids", str(ids)]
    assert EC.main(args) == 1                                          # no file yet
    torch.save({"prompts": {"p1": torch.zeros(2)}, "negative": torch.zeros(2), "negative_prompt": "neg"}, emb)
    assert EC.main(args) == 1                                          # e.g. written by a smoke test over id a only
    table.write_text(json.dumps({"b": "p1"}))                          # the prompt table overrides the cache prompt
    assert EC.main(args + ["--prompt-table", str(table)]) == 0
    neg.write_text("neg\n")                                            # the launcher passes "$(cat file)": newline dropped
    assert EC.main(args + ["--prompt-table", str(table), "--negative-prompt-file", str(neg)]) == 0
    neg.write_text("other")
    assert EC.main(args + ["--prompt-table", str(table), "--negative-prompt-file", str(neg)]) == 1


# ------------------------------------------------------------------------------------------------ inputs
def test_morpheus_inputs_reproduce_reference_prompt_table(tmp_path):
    from PIL import Image
    inputs = ROOT / "scripts/eval/inputs"
    img = Image.new("RGB", (8, 8))
    for kind in ("plain", "enhanced"):
        for line in (inputs / f"morpheus_{kind}.jsonl").read_text().splitlines():
            rec = json.loads(line)
            p = tmp_path / "morph" / rec["image"].split("Morpheus/", 1)[1]
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p)
    run(str(ROOT / "scripts/bench/official_sp_inputs.py"), "morpho", "--src", str(inputs / "morpheus_plain.jsonl"),
        "--src", str(inputs / "morpheus_enhanced.jsonl"), "--morpheus-repo", str(tmp_path / "morph"), "--out", str(tmp_path / "out"))
    table = (tmp_path / "out/PROMPT_TABLE.json").read_bytes()
    assert hashlib.sha256(table).hexdigest() == MORPHEUS_PROMPT_TABLE_SHA256
    rows = [json.loads(x) for x in (tmp_path / "out/manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 96 and all(not any(chr(0x4E00) <= ch <= chr(0x9FFF) for ch in r["caption"]) for r in rows)   # Stage-1 reads English


def test_pisabench_inputs(tmp_path):
    from PIL import Image
    src = ROOT / "scripts/eval/inputs/pisabench_real.jsonl"
    ff = tmp_path / "ff"
    ff.mkdir()
    for line in src.read_text().splitlines():
        Image.new("RGB", (8, 8)).save(ff / f"pisa_real_{int(json.loads(line)['id']):04d}.png")
    run(str(ROOT / "scripts/bench/official_sp_inputs.py"), "pisafull", "--src", str(src), "--first-frames", str(ff), "--out", str(tmp_path / "o"))
    assert (tmp_path / "o/NEGATIVE_PROMPT.txt").read_text() == "no camera motion, no slow-motion"
    rows = [json.loads(x) for x in (tmp_path / "o/manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 361 and all(r["caption"].endswith("A video that conforms to the laws of physics.") for r in rows)


# ------------------------------------------------------------------------------------------------ MoProbe
def test_moplan_paired_a_statistics():
    import moplan_paired_A as P
    rec = lambda t, er, co, same: {"type": t, "ex_real": er, "cf_opp": co, "ex_opp": False, "same_excf": same}
    ga = {"f1": [rec("lr", True, True, False)], "f2": [rec("lr", True, False, True)], "f3": [rec("noun", False, True, False)]}
    gb = {"f1": [rec("lr", False, False, True)], "f2": [rec("lr", True, False, True)], "f3": [rec("noun", False, False, True)]}
    p = P.paired(ga, gb, P.same5, n=200)
    assert p["frames"] == 3 and p["a"] == pytest.approx(1 / 3) and p["b"] == pytest.approx(1.0)
    assert p["ci"][0] <= p["diff"] <= p["ci"][1]
    rs = [r for v in ga.values() for r in v]
    # dboth = P(both) - sum_d w_d p_ex,d p_cf,d
    assert P.dboth5(rs) == pytest.approx(1 / 3 - (2 / 3 * 1.0 * 0.5 + 1 / 3 * 0.0 * 1.0))
    assert P.dirright(rs) == pytest.approx(1.0)


def test_moprobe_h_rules_derive():
    import moprobe_h_rules as R
    d = {"matched": {"0000": 1, "0001": 2, "obj": 3}, "actor_cat": "0000", "patient_cat": "0001", "template_id": 36,
         "roles": ["hand", "actor", "patient"], "placeholders": ["the box", "the cup"]}
    P, iss = R.derive("relation", d, "A hand moves the box and the cup away from each other.", "A hand moves them closer.")
    assert P == {"A": 1, "B": 2, "want_ex": "apart", "want_cf": "closer"} and not iss
    P, _ = R.derive("lr", d, "A hand pushes the box from left to right.", "A hand pushes the box from right to left.")
    assert P == {"k": 1, "want_ex": "right", "want_cf": "left"}
    P, _ = R.derive("updown", d, "A hand lifts the box.", "A hand puts the box down")
    assert P["want_cf"] == "down" and P["want_ex"] == "up"


def test_moprobe_h_missing_items_and_stale_real_camera(tmp_path, capsys):
    import os
    import moprobe_h_rules as R
    rows = []
    for k, folder in enumerate(("items_bench", "items_cfset")):
        d = tmp_path / folder
        for sub in ("items", "masks", "ref"):
            (d / sub).mkdir(parents=True)
        rows.append({"frame": f"ssv2_{k}", "cond": "ex", "items_dir": str(d)})
    for r in rows[:1]:
        (Path(r["items_dir"]) / "items" / f"{r['frame']}.json").write_text("{}")
        (Path(r["items_dir"]) / "masks" / f"{r['frame']}.npz").write_bytes(b"")
    with pytest.raises(SystemExit, match="1 MoProbe-H frames have no built item"):
        R.require_items(rows)
    assert R.require_items(rows, allow_missing=True) == [(rows[1]["items_dir"], "ssv2_1")]
    assert "--allow-missing" in capsys.readouterr().err
    r1 = Path(rows[1]["items_dir"])
    (r1 / "items" / "ssv2_1.json").write_text("{}")
    (r1 / "masks" / "ssv2_1.npz").write_bytes(b"")
    assert R.require_items(rows) == []
    # stored real-camera verdicts: reused only while current
    tasks = [dict(ver="B", base=r["frame"], items_dir=r["items_dir"]) for r in rows]
    out = tmp_path / "real_cam.jsonl"
    assert R.real_cam_stale(out, tasks) == "no stored verdicts"
    write_jsonl(out, [dict(t, verdict="still") for t in tasks])
    old = out.stat().st_mtime - 100
    for r in rows:
        for f in Path(r["items_dir"]).rglob("*"):
            if f.is_file():
                os.utime(f, (old, old))
    assert R.real_cam_stale(out, tasks) is None
    assert R.real_cam_stale(out, tasks[:1]) == "other frames"
    write_jsonl(out, [dict(tasks[0], verdict="still"), dict(tasks[1], verdict="missing")])
    assert R.real_cam_stale(out, tasks) is None                        # its reference file still does not exist
    (r1 / "ref" / "ssv2_1.npz").write_bytes(b"")
    os.utime(r1 / "ref" / "ssv2_1.npz", (old, old))
    assert "stored as missing" in R.real_cam_stale(out, tasks)
    write_jsonl(out, [dict(t, verdict="still") for t in tasks])
    newer = out.stat().st_mtime + 100
    os.utime(r1 / "items" / "ssv2_1.json", (newer, newer))
    assert "newer" in R.real_cam_stale(out, tasks)


def test_moprobe_data_consistency():
    data = ROOT / "data/moprobe"
    B = json.loads((data / "finding_set_B.json").read_text())
    rows = [json.loads(x) for x in (data / "moprobe_h_rows.jsonl").read_text().splitlines()]
    rej = set(json.loads((data / "rejected_frames.json").read_text())["rejected"])
    frames = {r["frame"] for r in rows}
    assert set(B["H"]) <= frames | rej
    assert set(B["f1_frames"]) <= set(B["H"])
    for r in rows:
        assert r["part"] == "H" and r["cond"] in ("ex", "cf", "nx", "nt", "rex", "rcf") and "/" not in r["items_dir"]
    A = json.loads((data / "finding_set_A.json").read_text())
    assert sum(len(v) for v in A["P"].values()) == 110
    p_rows = [json.loads(x) for x in (data / "moprobe_p_rows.jsonl").read_text().splitlines()]
    assert {f for v in A["P"].values() for f in v} <= {r["frame"] for r in p_rows}
    # per-clip manual labels of generated clips are results: they are not distributed
    assert not (data / "confabulation_blind").exists()
