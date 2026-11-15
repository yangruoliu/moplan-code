#!/usr/bin/env python3
"""Candidate table for the SSv2 TRAIN real pool: the physically relevant templates of the official
Something-Something v2 train split, grouped into event families, with a quota-balanced core of ~20k clips.

Reads ONLY labels/train.json.  Official validation clips never enter this pool (the MoProbe benchmark is built from
SSv2 validation): the output split is always "official_train", and with `--val-names` every id is additionally checked
against a list of validation tar member names.  The output schema is the one of the benchmark's SSv2 candidate table,
so its camera / box scripts (scripts/motionbench/ssv2_camera_prefilter.py, ssv2_boxes.py) can be reused unchanged:
  candidates_v0.csv    id, split, template_id, template, placeholders, label, groups  (+ family, pool, rank, noun)
  names.txt            tar member names (20bn-something-something-v2/<id>.webm) of every candidate (core + reserve)
  names_core.txt       the core only
  template_counts.csv  split, template_id, template, n   (+ family, pool)
  families.json        the derived whitelist (family -> template ids), quotas and per-family counts

The whitelist is re-derived from labels.json by keyword rules on the template text (FAMILY_RULES, first match wins,
after the fluid / camera exclusions), so it can be audited against the plan's list of 58 templates.  `groups` holds the
family name (one family per template).  The core is drawn per family by the quotas (round-robin over the family's
templates, ids in a seeded hash order) so that no family is dominated by one template; the rest of the whitelist is the
reserve (pool = "reserve", ranked after the core) and is used only if the camera / box gates leave the core short.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

SPLIT = "official_train"
MEMBER = "20bn-something-something-v2/{id}.webm"
DEFAULT_VAL_NAMES = ""

# Fluids and camera-motion templates are never candidates (checked on the normalised template text, before the rules).
EXCLUDE_ANY = ("camera", "pour", "spill", "water", "substance", "wet")

# Event families: ordered (family, regex) rules on the normalised, lower-case template text; first match wins.
# Static controls come first so that "without letting it roll" is static and not roll, "without letting it drop down"
# is lift and not fall-drop, etc.
FAMILY_RULES = [
    ("static", r"doesn't or almost doesn't move|without the stack collapsing|not enough for it to slide|doesn't glide down"
               r"|stays where it is|without it falling down|almost falls off but doesn't|without letting it roll"
               r"|pretending to throw|doesn't fall down|touching \(without moving\)|pretending to pick something up"
               r"|pretending to poke"),
    ("lift", r"lifting something up completely without|lifting up one end of something without"),
    ("fall_drop", r"^dropping|falling like a rock|, then letting it drop down|not supported and falls down"
                  r"|cannot support it so it falls down|falls off the table|until it falls down|^pushing something off of"),
    ("slope_slide", r"slanted surface, so it slides down|until it starts sliding down|until it falls off"),
    ("roll", r"roll"),
    ("tip_topple", r"falls over|falls on its side|^tipping something over|stack collapses|falls out"),
    ("collide", r"collid|deflected|^hitting something with something"),
    ("spin", r"spin"),
    ("throw", r"^throwing"),
    ("push_slide", r"^pushing something from|^pulling something from (left|right)|slightly moves|^pushing something with something"),
]
FAMILIES = [f for f, _ in FAMILY_RULES]
STATIC_FAMILY = "static"

# Core quotas (share of --target).  As given they sum to 1.10, so they are normalised to
# sum to 1 with the static controls pinned at >= MIN_STATIC_SHARE (normalise_quotas).
DEFAULT_QUOTAS = {"push_slide": 0.20, "roll": 0.12, "fall_drop": 0.18, "tip_topple": 0.10, "slope_slide": 0.05,
                  "collide": 0.08, "spin": 0.07, "throw": 0.05, "lift": 0.05, "static": 0.20}
MIN_STATIC_SHARE = 0.20


def normalise_quotas(quotas: dict, min_static: float = MIN_STATIC_SHARE) -> dict:
    """Shares summing to 1; if the static share is below `min_static` it is raised to it and the other families are
    scaled down proportionally."""
    total = sum(quotas.values())
    q = {k: v / total for k, v in quotas.items()}
    static = q.get(STATIC_FAMILY, 0.0)
    if STATIC_FAMILY in q and static < min_static:
        others = 1.0 - static
        q = {k: (min_static if k == STATIC_FAMILY else v * (1.0 - min_static) / others) for k, v in q.items()}
    return q


def norm_template(t: str) -> str:
    return t.replace("[", "").replace("]", "")


def family_of(template: str) -> str | None:
    """Event family of a template (normalised or bracketed text), or None when it is not whitelisted."""
    text = norm_template(template).strip().lower()
    if any(word in text for word in EXCLUDE_ANY):
        return None
    for family, pattern in FAMILY_RULES:
        if re.search(pattern, text):
            return family
    return None


def norm_noun(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    return re.sub(r"^(a|an|the|some|two|three|several|many|one)\s+", "", s)


def derive_whitelist(labels: dict) -> dict:
    """labels.json (template -> id string) -> {family: [template ids]} for every family (main() refuses empty ones)."""
    fam = collections.defaultdict(list)
    for template, tid in labels.items():
        f = family_of(template)
        if f is not None:
            fam[f].append(int(tid))
    return {f: sorted(fam.get(f, [])) for f in FAMILIES}


def order_key(seed: int, vid: str) -> str:
    return hashlib.sha256(f"ssv2-train-{seed}:{vid}".encode()).hexdigest()


def draw_core(rows_by_family: dict, quotas: dict, target: int, seed: int) -> dict:
    """{id: rank} of the core: per family min(quota * target, available), round-robin over the family's templates in a
    seeded hash order.  Rank is the position in the draw (0 = first), per family."""
    core = {}
    for family, rows in rows_by_family.items():
        want = min(int(round(quotas.get(family, 0.0) * target)), len(rows))
        by_template = collections.defaultdict(list)
        for r in rows:
            by_template[r["template_id"]].append(r)
        for lst in by_template.values():
            lst.sort(key=lambda r: order_key(seed, r["id"]))
        queues = [by_template[t] for t in sorted(by_template)]
        rank = 0
        while rank < want:
            progressed = False
            for q in queues:
                if q and rank < want:
                    core[q.pop(0)["id"]] = rank
                    rank += 1
                    progressed = True
            if not progressed:
                break
    return core


def build(labels_zip: str, out_dir: str, val_names: str | None, target: int, seed: int, quotas: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(labels_zip)
    labels = json.load(z.open("labels/labels.json"))
    whitelist = derive_whitelist(labels)
    tid2family = {tid: f for f, ids in whitelist.items() for tid in ids}
    id2template = {int(v): norm_template(k) for k, v in labels.items()}
    tid_of = {norm_template(k): int(v) for k, v in labels.items()}

    train = json.load(z.open("labels/train.json"))
    val_ids = {json.loads(json.dumps(r["id"])) for r in json.load(z.open("labels/validation.json"))}
    if val_names:
        for line in Path(val_names).read_text().splitlines():
            line = line.strip()
            if line:
                val_ids.add(Path(line).stem)

    rows_by_family = collections.defaultdict(list)
    n_total = 0
    for r in train:
        tid = tid_of[norm_template(r["template"])]
        f = tid2family.get(tid)
        if f is None:
            continue
        vid = str(r["id"])
        assert vid not in val_ids, f"official validation id {vid} in train.json"
        ph = r.get("placeholders", [])
        rows_by_family[f].append({
            "id": vid, "split": SPLIT, "template_id": tid, "template": id2template[tid],
            "placeholders": json.dumps(ph, ensure_ascii=False), "label": r.get("label", ""), "groups": f,
            "family": f, "noun": norm_noun(ph[0]) if ph else "",
        })
        n_total += 1
    core = draw_core(rows_by_family, quotas, target, seed)
    rows = []
    for f in FAMILIES:
        fam_rows = rows_by_family.get(f, [])
        n_core = sum(1 for r in fam_rows if r["id"] in core)
        reserve = [r for r in fam_rows if r["id"] not in core]
        reserve.sort(key=lambda r: order_key(seed, r["id"]))
        for r in fam_rows:
            if r["id"] in core:
                r["pool"], r["rank"] = "core", core[r["id"]]
        for i, r in enumerate(reserve):
            r["pool"], r["rank"] = "reserve", n_core + i
        rows.extend(sorted(fam_rows, key=lambda r: r["rank"]))
    # core first (interleaved over families by rank) so a prefix of the table is a balanced subset
    rows.sort(key=lambda r: (r["pool"] != "core", r["rank"], FAMILIES.index(r["family"])))
    assert all(r["split"] == SPLIT for r in rows)
    assert not {r["id"] for r in rows} & val_ids

    fields = ["id", "split", "template_id", "template", "placeholders", "label", "groups", "family", "pool", "rank", "noun"]
    with open(out / "candidates_v0.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out / "names.txt").write_text("".join(MEMBER.format(id=r["id"]) + "\n" for r in rows))
    (out / "names_core.txt").write_text("".join(MEMBER.format(id=r["id"]) + "\n" for r in rows if r["pool"] == "core"))
    cnt = collections.Counter((r["split"], r["template_id"], r["template"], r["family"], r["pool"]) for r in rows)
    with open(out / "template_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "template_id", "template", "family", "pool", "n"])
        for key, n in sorted(cnt.items(), key=lambda kv: (kv[0][3], kv[0][1], kv[0][4])):
            w.writerow(list(key) + [n])
    per_family = {}
    for f in FAMILIES:
        fam_rows = rows_by_family.get(f, [])
        per_family[f] = {"templates": whitelist[f], "whitelisted": len(fam_rows),
                         "core": sum(1 for r in fam_rows if r["pool"] == "core"),
                         "quota": quotas.get(f, 0.0),
                         "by_template": {str(t): sum(1 for r in fam_rows if r["template_id"] == t) for t in whitelist[f]}}
    summary = {"train_rows": len(train), "whitelisted": n_total, "templates": sum(len(v) for v in whitelist.values()),
               "core": len(core), "target": target, "seed": seed, "families": per_family,
               "template_names": {str(t): id2template[t] for ids in whitelist.values() for t in ids}}
    (out / "families.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-zip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-names", default=DEFAULT_VAL_NAMES,
                    help="optional SSv2 validation tar member names to assert disjointness; '' (default) skips the check")
    ap.add_argument("--target", type=int, default=20000, help="core size the quotas are applied to")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--quotas", default=",".join(f"{k}={v}" for k, v in DEFAULT_QUOTAS.items()))
    args = ap.parse_args()
    quotas = normalise_quotas({k: float(v) for k, v in (kv.split("=") for kv in args.quotas.split(","))})
    s = build(args.labels_zip, args.out_dir, args.val_names or None, args.target, args.seed, quotas)
    empty = [f for f, d in s["families"].items() if not d["templates"]]
    if empty:
        raise SystemExit(f"no template matched for families {empty}: the keyword rules drifted from labels.json")
    print(f"train rows {s['train_rows']}; whitelisted {s['whitelisted']} clips in {s['templates']} templates; "
          f"core {s['core']} (target {s['target']})")
    print(f"  {'family':12s} {'quota':>6s} {'core':>6s} {'whitelist':>9s}  templates")
    for f, d in s["families"].items():
        print(f"  {f:12s} {d['quota']:6.2f} {d['core']:6d} {d['whitelisted']:9d}  "
              + ", ".join(f"{t}:{d['by_template'][str(t)]}" for t in d["templates"]))
    static_share = s["families"][STATIC_FAMILY]["core"] / max(1, s["core"])
    print(f"  static share of core: {static_share:.3f}")


if __name__ == "__main__":
    main()
