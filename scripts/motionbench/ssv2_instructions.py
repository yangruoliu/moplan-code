#!/usr/bin/env python3
"""Step 5 of the MoProbe SSv2 build: deterministic instruction writing from the template table.

make_instruction(template_id, placeholders, variant) -> str
  variant 'neutral'  : Physics-IQ style (scene + subject + initial action, no outcome words)
  variant 'explicit' : adds direction / outcome words (T2 only)
Noun normalisation: lower-case, strip leading articles, prefix 'the ' (SSv2 placeholders are
bare nouns like 'cube', 'a paper', 'the remote').
CLI: --preview N writes a CSV of N random candidates with both variants for manual QA.
"""
import argparse, csv, json, random, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = json.load(open(HERE / "ssv2_templates.json"))["templates"]
CF = json.load(open(HERE / "ssv2_templates.json"))["counterfactual_pairs"]
OUTCOME_WORDS = re.compile(r"\b(falls?|fell|slides?|rolls?|collide|spins?|drops?|tips?|stops?|deflect|halt|stays?|does not|doesn't|nothing)\b", re.I)


def norm_noun(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(a|an|the|some|two|three|several|many|one)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return f"the {s}" if s else "the object"


ORDINALS = ("first", "second", "third")


def distinct_names(placeholders):
    """Normalised nouns; identical placeholders become 'the first potato' / 'the second potato' (SSv2 often repeats a
    noun: 'Moving potato and potato so they collide' would otherwise read 'the potato and the potato')."""
    names = [norm_noun(p) for p in placeholders]
    for i, n in enumerate(names):
        same = [j for j, m in enumerate(names) if m == n]
        if len(same) > 1 and n.startswith("the "):
            for rank, j in enumerate(same[:3]):
                names[j] = f"the {ORDINALS[rank]} {norm_noun(placeholders[j])[4:]}"
    return names + ["the object"] * 3


def make_instruction(template_id, placeholders, variant="neutral"):
    t = TABLE[str(template_id)]
    names = distinct_names(placeholders)
    txt = t[variant].format(A=names[0], B=names[1], C=names[2])
    txt = txt[0].upper() + txt[1:]
    if not txt.endswith("."):
        txt += "."
    return txt


def roles_for(template_id, n_placeholders):
    """Return (actor_placeholder, patient_placeholder, direction, static_caption, event, rigid)."""
    t = TABLE[str(template_id)]
    return dict(actor=t.get("actor"), patient=t.get("patient"), dir=t.get("dir"),
                static_caption=bool(t.get("static_caption", False)), event=t["event"],
                rigid=bool(t.get("rigid", True)))


def counterfactual_for(template_id, placeholders):
    """Alternative instruction for the same first frame, or None."""
    c = CF.get(str(template_id))
    if not c:
        return None
    if "alt_template" in c:
        return dict(kind=c["kind"], template_id=int(c["alt_template"]),
                    text=make_instruction(c["alt_template"], placeholders, "explicit"))
    names = distinct_names(placeholders)
    txt = c["alt"].format(A=names[0], B=names[1], C=names[2])
    return dict(kind=c["kind"], template_id=None, text=txt[0].upper() + txt[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--preview", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(args.candidates)) if r["split"] == "official_val"]
    rows = [r for r in rows if r["template_id"] in TABLE]
    random.Random(args.seed).shuffle(rows)
    leaks = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "template_id", "template", "placeholders", "neutral", "explicit", "cf_kind", "cf_text", "neutral_has_outcome_word"])
        for r in rows[:args.preview]:
            ph = json.loads(r["placeholders"])
            neu = make_instruction(r["template_id"], ph, "neutral")
            exp = make_instruction(r["template_id"], ph, "explicit")
            cf = counterfactual_for(r["template_id"], ph) or {}
            leak = bool(OUTCOME_WORDS.search(neu))
            leaks += leak
            w.writerow([r["id"], r["template_id"], r["template"], r["placeholders"], neu, exp, cf.get("kind", ""), cf.get("text", ""), int(leak)])
    print(f"wrote {min(args.preview, len(rows))} previews; neutral instructions with outcome words: {leaks}")
    # whole-table audit: every neutral pattern must be free of outcome words
    bad = {k: v["neutral"] for k, v in TABLE.items() if OUTCOME_WORDS.search(v["neutral"])}
    print("neutral patterns containing outcome words:", bad if bad else "none")


if __name__ == "__main__":
    main()
