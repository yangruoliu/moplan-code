#!/usr/bin/env python3
"""Inventory of 27-channel (stage2_cond_v2) Stage-2 caches + evaluation exclusions for the single-run GT SFT.

Provenance tool: it produced the exclusion lists and the per-source id lists from which the frozen training lists in
data/ids/stage2_sft were made (data/ids/stage2_sft/exclusion_lists.json is its exclusion_lists.json).  Training does
not need it.  Its inputs are id lists / manifests of the data pipeline (no tensors are read).

Writes <out>/inventory.json, <out>/EXCLUSION_RECEIPT.json, <out>/exclusion_lists.json, <out>/ids/<pool>.txt (final
training ids per source) and <out>/phyco_new_pool.jsonl (PhyCo clips eligible for new caching).
Rule: a clip is dropped if its id is in any evaluation list; a P101 clip is also dropped when ANY camera of the same
trial is in a list.

usage: inventory.py OUT --p101-cam1-cache DIR --p101-otherviews DIR --p101-dev96 F --p101-rl-holdout F
         --p101-rl-holdout48 F --ssv2-cache DIR --ssv2-official-val F --ssv2-official-test F --phyco-root DIR
         --phyco-stage1-val F [--phyco-extra-val F ...] --phyco-rl-excluded F --phyco-quality F
         --phyco-cached-ids F [--phyco-local-cache DIR] --phyco-candidates F [--other-cache DIR ...]
  --p101-cam1-cache DIR    camera-1 cache: <sid>.pt, ids_train.txt, ids_train_<scenario>.txt, ids_dev*.txt, ids_test*.txt
  --p101-otherviews DIR    other-view lists: ids_all.txt, ids_train.txt, ids_train_<scenario>.txt, ids_dev*.txt, ids_test*.txt
  --ssv2-cache DIR         SSv2 cache: <sid>.pt, ids_strict_train.txt, ids_hold.txt, ids_strict_hold.txt
  --ssv2-official-val/test official SSv2 validation / test ids (ssv2_<id>, from the dataset's label files)
  --phyco-root DIR         PhyCo Kubric root: manifests_v1/{val,test_core}.jsonl, manifests/{val,test,...}.csv
"""
import argparse
import collections
import csv
import json
import re
from pathlib import Path


def read_ids(path):
    path = Path(path)
    return [l.strip() for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def pt_ids(folder):
    return sorted(p.stem for p in Path(folder).glob("*.pt") if not p.name.startswith("summary-"))


def group_of(sid):          # P101 trial = sample id without the camera suffix
    return sid.split("__")[0]


def phyco_id_from_video(path):
    m = re.search(r"extracted/([^/]+)/([^/]+)/([^/]+)/rgba\.mp4", path)
    return f"phyco_{m.group(1)}_{m.group(2)}_{m.group(3)}" if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--p101-cam1-cache", required=True)
    ap.add_argument("--p101-otherviews", required=True)
    ap.add_argument("--p101-dev96", required=True)
    ap.add_argument("--p101-rl-holdout", required=True)
    ap.add_argument("--p101-rl-holdout48", required=True)
    ap.add_argument("--ssv2-cache", required=True)
    ap.add_argument("--ssv2-official-val", required=True)
    ap.add_argument("--ssv2-official-test", required=True)
    ap.add_argument("--phyco-root", required=True)
    ap.add_argument("--phyco-stage1-val", required=True, help="Stage-1 PhyCo validation manifest (jsonl)")
    ap.add_argument("--phyco-extra-val", action="append", default=[], help="further PhyCo validation id lists")
    ap.add_argument("--phyco-rl-excluded", required=True, help="jsonl rows with sample_id, known_holdout")
    ap.add_argument("--phyco-quality", required=True, help="data-quality exclusions json ({'mapping': {sid: reason}})")
    ap.add_argument("--phyco-cached-ids", required=True, help="ids of the existing 29-frame PhyCo train cache")
    ap.add_argument("--phyco-local-cache", default="", help="a local subset of that cache (<sid>.pt), for the report only")
    ap.add_argument("--phyco-candidates", required=True, help="RL-pool candidate table (jsonl)")
    ap.add_argument("--other-cache", action="append", default=[], help="further caches to report overlaps for")
    a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); (OUT / "ids").mkdir(exist_ok=True)
    UL = Path(a.p101_otherviews)

    inv, excl = {}, {}
    # ---------------- P101 ----------------
    p1 = Path(a.p101_cam1_cache)
    cam1_all = set(pt_ids(p1))
    cam1_train = set(read_ids(p1 / "ids_train.txt"))
    ev = {}
    ev["p101_dev96"] = set(read_ids(a.p101_dev96))
    ev["p101_rl_holdout"] = set(read_ids(a.p101_rl_holdout))
    ev["p101_rl_holdout48"] = set(read_ids(a.p101_rl_holdout48))
    for f in sorted(p1.glob("ids_dev*.txt")) + sorted(p1.glob("ids_test*.txt")):
        ev.setdefault("p101_cam1_dev_test_lists", set()).update(read_ids(f))
    for f in sorted(UL.glob("ids_dev*.txt")) + sorted(UL.glob("ids_test*.txt")):
        ev.setdefault("p101_otherview_dev_test_lists", set()).update(read_ids(f))
    p101_eval_ids = set().union(*ev.values())
    p101_eval_groups = {group_of(s) for s in p101_eval_ids}
    cam1_keep = sorted(s for s in cam1_train if s not in p101_eval_ids and group_of(s) not in p101_eval_groups)
    r_all = set(read_ids(UL / "ids_all.txt"))
    r_train = set(read_ids(UL / "ids_train.txt"))
    r_keep = sorted(s for s in r_train if s in r_all and s not in p101_eval_ids and group_of(s) not in p101_eval_groups)
    inv["p101_cam1"] = {"cache": str(p1), "files": len(cam1_all), "train_list": len(cam1_train),
                        "train_in_eval_lists": len(cam1_train & p101_eval_ids),
                        "train_sharing_trial_with_eval": sum(group_of(s) in p101_eval_groups for s in cam1_train), "kept": len(cam1_keep)}
    inv["p101_otherviews"] = {"cache": str(UL), "files": len(r_all), "train_list": len(r_train),
                              "train_sharing_trial_with_eval": sum(group_of(s) in p101_eval_groups for s in r_train), "kept": len(r_keep),
                              "cameras": dict(collections.Counter(s.split("__")[1] for s in r_keep))}
    for scen in ("fall", "ramp", "spring", "multi", "liquid"):
        c1 = set(read_ids(p1 / f"ids_train_{scen}.txt")); rr = set(read_ids(UL / f"ids_train_{scen}.txt"))
        (OUT / "ids" / f"p101_{scen}.txt").write_text("".join(s + "\n" for s in sorted((c1 & set(cam1_keep)) | (rr & set(r_keep)))))
    (OUT / "ids/p101_cam1.txt").write_text("".join(s + "\n" for s in cam1_keep))
    (OUT / "ids/p101_otherviews.txt").write_text("".join(s + "\n" for s in r_keep))
    # ---------------- SSv2 ----------------
    s2 = Path(a.ssv2_cache)
    ss_all = set(pt_ids(s2)); ss_strict = set(read_ids(s2 / "ids_strict_train.txt"))
    ev_ss = {"ssv2_hold": set(read_ids(s2 / "ids_hold.txt")) | set(read_ids(s2 / "ids_strict_hold.txt")),
             "ssv2_official_validation": set(read_ids(a.ssv2_official_val)),
             "ssv2_official_test": set(read_ids(a.ssv2_official_test))}
    ss_eval = set().union(*ev_ss.values())
    ss_keep = sorted(s for s in ss_strict if s in ss_all and s not in ss_eval)
    inv["ssv2"] = {"cache": str(s2), "files": len(ss_all), "strict_train_list": len(ss_strict), "in_eval": len(ss_strict & ss_eval), "kept": len(ss_keep),
                   "official_split_check": "every kept id must be in the SSv2 train labels and in neither the validation nor the test labels"}
    (OUT / "ids/ssv2.txt").write_text("".join(s + "\n" for s in ss_keep))
    ev.update(ev_ss)
    # ---------------- PhyCo ----------------
    P = Path(a.phyco_root)
    ev_pc = collections.defaultdict(set)
    for name in ("val", "test_core"):
        for line in (P / f"manifests_v1/{name}.jsonl").read_text().splitlines():
            if line.strip():
                ev_pc[f"phyco_raw_{name}"].add(json.loads(line)["sample_id"])
    for name in ("val", "test", "val.clean12", "test.clean12", "val.buggycolor", "test.buggycolor"):
        f = P / f"manifests/{name}.csv"
        if f.is_file():
            for row in csv.DictReader(f.open()):
                sid = phyco_id_from_video(row["video_path"])
                if sid: ev_pc["phyco_legacy_val_test_csv"].add(sid)
    for line in Path(a.phyco_stage1_val).read_text().splitlines():
        if line.strip():
            ev_pc["phyco_stage1_val"].add(json.loads(line)["sample_id"])
    for extra in a.phyco_extra_val:
        ev_pc["phyco_stage1_val"] |= set(read_ids(extra))
    for line in Path(a.phyco_rl_excluded).read_text().splitlines():
        r = json.loads(line)
        if r.get("known_holdout"):
            ev_pc["phyco_rl_pool_known_holdout"].add(r["sample_id"])
    quality = json.loads(Path(a.phyco_quality).read_text())["mapping"]
    pc_eval = set().union(*ev_pc.values())
    cached_pc = set(read_ids(a.phyco_cached_ids)); local_pc = set(pt_ids(a.phyco_local_cache)) if a.phyco_local_cache else set()
    cand = [json.loads(l) for l in Path(a.phyco_candidates).read_text().splitlines() if l.strip()]
    cand_ids = {r["sample_id"] for r in cand}
    pc_exist_keep = sorted(s for s in cached_pc if s not in pc_eval and s not in quality)
    inv["phyco_existing"] = {"cache_ids": str(a.phyco_cached_ids), "files": len(cached_pc),
                             "files_local": len(local_pc), "local_subset_of_cached": local_pc <= cached_pc, "in_eval": len(cached_pc & pc_eval),
                             "in_quality_exclusions": len(cached_pc & set(quality)), "kept": len(pc_exist_keep),
                             "kept_local": len(set(pc_exist_keep) & local_pc), "kept_in_rl_candidates": len(set(pc_exist_keep) & cand_ids),
                             "scenarios": dict(collections.Counter(re.sub(r"_\d{4}-\d\d-\d\d.*", "", s[6:]) for s in pc_exist_keep))}
    (OUT / "ids/phyco_existing.txt").write_text("".join(s + "\n" for s in pc_exist_keep))
    (OUT / "ids/phyco_existing_fetch.txt").write_text("".join(s + "\n" for s in pc_exist_keep if s not in local_pc))
    new_pool = [r for r in cand if r["sample_id"] not in cached_pc and r["sample_id"] not in pc_eval and r["sample_id"] not in quality
                and not r.get("known_holdout") and not any(r["holdout_checks"].values()) and Path(r["stage1_target_path"]).is_file() is not None]
    inv["phyco_new_candidates"] = {"rl_pool_candidates": len(cand), "not_cached_and_eligible": len(new_pool),
                                   "scenarios": dict(collections.Counter(r["scenario"] for r in new_pool))}
    with (OUT / "phyco_new_pool.jsonl").open("w") as f:
        for r in new_pool:
            f.write(json.dumps({k: r[k] for k in ("sample_id", "scenario", "caption", "video_path", "stage1_target_path", "first_frame_path", "raw_manifest_split")}) + "\n")
    ev.update(ev_pc)
    # ---------------- other caches (same clips as above, measured cond_real): overlap report ----------------
    others = {}
    for d in a.other_cache:
        ids = set(pt_ids(d))
        others[d] = {"files": len(ids), "in_p101_pools": len(ids & (set(cam1_keep) | set(r_keep))), "in_phyco_existing": len(ids & set(pc_exist_keep)),
                     "in_eval": len(ids & (p101_eval_ids | pc_eval | ss_eval)), "p101_in_eval_trial": sum(group_of(s) in p101_eval_groups for s in ids if s.startswith("p101")),
                     "new_unique": len(ids - set(cam1_keep) - set(r_keep) - set(pc_exist_keep) - p101_eval_ids - pc_eval)}
    inv["other_caches"] = others
    inv["excluded_never_used"] = {"PisaBench": "evaluation", "Physics-IQ": "evaluation", "PhyCo Stage-1 val": "validation"}
    receipt = {"schema": "s2sft_exclusion_receipt_v1", "rule": "a clip is dropped if its id is in any list below; P101 clips are also dropped when ANY camera of the same trial (group_id) is in a list",
               "lists": {k: {"n": len(v)} for k, v in ev.items()},
               "sources": {"p101_dev96": str(a.p101_dev96), "p101_rl_holdout": str(a.p101_rl_holdout),
                           "p101_cam1_dev_test_lists": str(p1 / "ids_{dev,test}*.txt"), "p101_otherview_dev_test_lists": str(UL / "ids_{dev,test}*.txt"),
                           "ssv2_hold": str(s2 / "ids_hold.txt + ids_strict_hold.txt"), "ssv2_official_validation/test": "SSv2 labels validation.json/test.json",
                           "phyco_raw_val/test_core": str(P / "manifests_v1/{val,test_core}.jsonl"), "phyco_legacy_val_test_csv": str(P / "manifests/{val,test}*.csv"),
                           "phyco_stage1_val": str(a.phyco_stage1_val), "phyco_rl_pool_known_holdout": str(a.phyco_rl_excluded) + " (known_holdout)",
                           "phyco_quality (not eval; data quality)": str(a.phyco_quality)},
               "not_used_at_all": ["PisaBench", "Physics-IQ", "SSv2 validation / test", "RigidBench / Morpheus inputs"],
               "dropped_counts": {"p101_cam1_train": len(cam1_train) - len(cam1_keep), "p101_otherviews_train": len(r_train) - len(r_keep),
                                  "ssv2_strict_train": len(ss_strict) - len(ss_keep), "phyco_existing": len(cached_pc) - len(pc_exist_keep)}}
    (OUT / "inventory.json").write_text(json.dumps(inv, indent=1) + "\n")
    (OUT / "exclusion_lists.json").write_text(json.dumps({k: sorted(v) for k, v in ev.items()}) + "\n")
    (OUT / "EXCLUSION_RECEIPT.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(inv, indent=1)); print(json.dumps(receipt["dropped_counts"]))


if __name__ == "__main__":
    main()
