#!/usr/bin/env python3
"""Pick the additional PhyCo train clips to cache: scenario water-filling over the eligible pool (inventory's
phyco_new_pool.jsonl = RL-pool candidates minus every cached clip, every holdout list and the quality exclusions),
deterministic order (sha1 of the sample id).  With CANDIDATES (the RL-pool candidate table) it also writes a small
overlap manifest (clips that already have a 29-frame payload) for a consistency check.
Provenance tool: it picked the new PhyCo clips of the frozen list data/ids/stage2_sft/train/phyco.txt.
usage: select_phyco_new.py INVENTORY_DIR OUT_DIR N_NEW [CANDIDATES]
"""
import collections, hashlib, json, sys
from pathlib import Path

inv, out, n_new = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
pool = [json.loads(l) for l in (inv / "phyco_new_pool.jsonl").read_text().splitlines() if l.strip()]
for r in pool:
    if not Path(r["stage1_target_path"]).is_file() or not Path(r["video_path"]).is_file():
        r["missing"] = True
pool = [r for r in pool if not r.get("missing")]
key = lambda r: hashlib.sha1(r["sample_id"].encode()).hexdigest()  # noqa: E731
by = collections.defaultdict(list)
for r in sorted(pool, key=key):
    by[r["scenario"]].append(r)
quota, left = {s: 0 for s in by}, n_new
while left > 0:
    open_ = [s for s in by if quota[s] < len(by[s])]
    if not open_:
        break
    share = max(1, left // len(open_))
    for s in sorted(open_, key=lambda s: len(by[s])):
        take = min(share, len(by[s]) - quota[s], left)
        quota[s] += take
        left -= take
picked = [r for s in sorted(by) for r in by[s][: quota[s]]]
picked.sort(key=key)                     # shards interleave scenarios
(out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in picked))
(out / "ids.txt").write_text("".join(r["sample_id"] + "\n" for r in picked))
cand = [json.loads(l) for l in Path(sys.argv[4]).read_text().splitlines() if l.strip()] if len(sys.argv) > 4 else []
overlap = sorted((r for r in cand if r.get("old_s2_cache_path") and Path(r["old_s2_cache_path"]).is_file()), key=key)[:8]
(out / "check_manifest.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("sample_id", "scenario", "caption", "video_path", "stage1_target_path", "old_s2_cache_path")}) + "\n" for r in overlap))
rec = {"eligible_with_files": len(pool), "picked": len(picked), "quota": quota, "available": {s: len(v) for s, v in by.items()}}
(out / "SELECTION.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec, indent=1))
