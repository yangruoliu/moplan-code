"""Showcase helper: render the gallery, check the assets, list and verify regenerated panels.

Every case lives in assets/showcase/<NN>_<id>/ with a meta.json that is the single source of truth.

  python assets/showcase/showcase.py build              # rewrite the site page (repo-root index.html) and the README table
  python assets/showcase/showcase.py check              # verify every asset against the sha256 in meta.json
  python assets/showcase/showcase.py commands [--case N] [--arm moplan|base]   # print generate.py commands
  python assets/showcase/showcase.py verify OUT_DIR     # compare regenerated panels with the references

`verify` looks for OUT_DIR/<NN>_<id>/<key>.npy and OUT_DIR/<NN>_<id>/<key>.mp4, where <key> is the panel key in
meta.json (moplan, base, A_moplan, B_base, ...). These are the paths used by the commands in meta.json.
Panels whose reference is marked `"verifiable": false` (the RigidBench cases, whose references are hashes of the
1280x704 evaluation exports, not of what generate.py writes) are reported as NOT-VERIFIABLE and do not fail.
Standard library only.
"""
import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROUPS = [
    ("Inertia", "Inertia: motion follows the first frame",
     "Image-to-video models tend to produce the motion the first frame suggests, whatever the text says. MoPlan "
     "predicts the motion plan from the first frame and the text, so the named object moves in the named direction. "
     "Two-row examples pair the real instruction (A) with an opposite one (B) on the same first frame and seed."),
    ("Confabulation", "Confabulation: models invent agents and objects",
     "When the cause of a motion is not visible, the untrained model tends to invent one: a hand, a tool or new "
     "objects. MoPlan moves only the objects that are in the first frame."),
    ("Entanglement", "Entanglement: the camera moves with the object",
     "The real camera is fixed. Image-to-video models often move the camera with the object; in these examples "
     "MoPlan keeps the camera still while the object moves."),
    ("Physics-IQ qualitative", "General physical events (Physics-IQ)",
     "Out-of-domain real experiments from Physics-IQ. The real video is the recorded continuation after the "
     "conditioning frame."),
    ("New results", "Further examples",
     "Further examples on rendered RigidBench scenes (no recorded continuation)."),
]
INTRO = ("Each example uses the same first frame, the same text and the same random seed for both models, under the "
         "same generation protocol (448×256). The base model is the untrained Wan2.2-I2V-A14B; MoPlan is our final "
         "model, which predicts an object-level motion plan from the first frame and the text and generates the "
         "video conditioned on it. Where it is available, the real continuation is shown on the left.")
SELECTION = ("Examples were chosen by watching the videos, not by score alone: we kept only cases where the physical "
             "event is right and the difference from the base model is visible.")
LEGEND = ["left: real video (when shown)", "middle: untrained Wan2.2, same seed", "right: MoPlan"]
FOOTER = ("Videos are shown at the resolution and frame rate used for evaluation (448×256; 8 fps for Physics-IQ, "
          "7 fps otherwise).")
TABLE_START, TABLE_END = "<!-- cases:start -->", "<!-- cases:end -->"


def load_cases():
    cases = [json.loads(p.read_text()) for p in sorted(HERE.glob("[0-9][0-9]_*/meta.json"))]
    for c in cases:
        assert c["folder"] == f"{c['number']:02d}_{c['id']}", c["folder"]
    return sorted(cases, key=lambda c: c["number"])


def generated_panels(case):
    for row in case["layout"]["rows"]:
        for p in row["panels"]:
            if p["arm"] in ("moplan", "base"):
                yield row, p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def panel_summary(case):
    rows = case["layout"]["rows"]
    labels = {"real": "Real", "base": "Wan2.2", "moplan": "MoPlan", "still": "First frame"}
    parts = []
    for row in rows:
        s = ", ".join(labels[p["arm"]] for p in row["panels"])
        parts.append(f"{row['row']}: {s}" if row["row"] else s)
    return "; ".join(parts)


def seeds(case):
    out = []
    for row, p in generated_panels(case):
        if p["arm"] == "moplan":
            out.append(f"{row['row']}: {p['seed']}" if row["row"] else str(p["seed"]))
    return ", ".join(out)


def readme_table(cases):
    lines = ["| # | Group | Title | Benchmark | Sample id(s) | Panels | Seed |",
             "|---|---|---|---|---|---|---|"]
    for c in cases:
        ids = ", ".join(f"`{s}`" for s in c["sample_ids"])
        title = f"[{c['title']}]({c['folder']}/meta.json)"
        lines.append(f"| {c['number']} | {c['finding_group']} | {title} | {c['benchmark']} | {ids} | "
                     f"{panel_summary(c)} | {seeds(c)} |")
    return "\n".join(lines)


CSS = """
:root { --fg: #1d1d1f; --muted: #5f6368; --bg: #ffffff; --card: #f6f7f9; --line: #dfe1e5; --chip: #e8eaed; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e8eaed; --muted: #a0a4a8; --bg: #16181b; --card: #202327; --line: #33373c; --chip: #2c3035; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 system-ui, sans-serif; }
main { max-width: 1400px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.7rem; margin: 0 0 8px; }
h2 { font-size: 1.3rem; margin: 40px 0 6px; padding-top: 12px; border-top: 1px solid var(--line); }
p { margin: 6px 0; }
.muted { color: var(--muted); }
.legend span { display: inline-block; background: var(--chip); border-radius: 12px; padding: 2px 10px;
  margin: 4px 6px 0 0; font-size: 0.9rem; }
nav a { margin-right: 14px; color: inherit; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
  margin: 18px 0; }
.card h3 { margin: 0 0 8px; font-size: 1.1rem; }
.num { color: var(--muted); margin-right: 6px; }
video, img.strip { display: block; max-width: 100%; height: auto; border-radius: 6px; background: #000; }
img.strip { background: transparent; margin-top: 8px; }
details { margin-top: 8px; }
summary { cursor: pointer; color: var(--muted); }
table { border-collapse: collapse; width: 100%; margin-top: 6px; font-size: 0.9rem; }
th, td { border: 1px solid var(--line); padding: 4px 6px; text-align: left; vertical-align: top; }
pre { white-space: pre-wrap; word-break: break-all; font-size: 0.8rem; background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
footer { margin-top: 40px; color: var(--muted); font-size: 0.9rem; }
"""


def card_html(c, prefix=""):
    e = html.escape
    f = c["files"]
    fo = prefix + c["folder"]
    v = f["video"]
    out = [f'<article class="card" id="case-{c["number"]:02d}">',
           f'<h3><span class="num">#{c["number"]}</span>{e(c["title"])}</h3>',
           f'<video controls loop muted playsinline preload="none" width="{v["width"]}" height="{v["height"]}" '
           f'poster="{e(fo)}/{e(f["poster"]["path"])}">'
           f'<source src="{e(fo)}/{e(v["path"])}" type="video/mp4"></video>',
           f'<p>{e(c["caption"])}</p>']
    rows = []
    for row, p in generated_panels(c):
        rows.append("<tr>" + "".join(f"<td>{e(str(x))}</td>" for x in (
            row["row"] or "", p["label"], p["sample_id"], p["seed"], p["prompt"])) + "</tr>")
    out.append('<details><summary>Prompts, seeds and inputs</summary>'
               f'<p class="muted">{e(c["benchmark"])}: {e(c["benchmark_detail"])}. '
               f'First frame: <a href="{e(fo)}/{e(f["first_frame"]["path"])}">'
               f'{e(f["first_frame"]["path"])}</a> ({e(c["first_frame"]["source"])}).</p>'
               '<table><tr><th>Row</th><th>Panel</th><th>Sample id</th><th>Seed</th><th>Prompt</th></tr>'
               + "".join(rows) + "</table>"
               f'<p class="muted">Seed rule: {e(c["seed_rule"])}.</p>'
               '<p class="muted">Regenerate (from the repository root):</p>'
               f'<pre>{e(chr(10).join(c["regenerate"]))}</pre>'
               f'<p class="muted"><a href="{e(fo)}/meta.json">meta.json</a></p></details>')
    s = f["strip"]
    out.append('<details><summary>Frames over time</summary>'
               f'<img class="strip" loading="lazy" src="{e(fo)}/{e(s["path"])}" width="{s["width"]}" '
               f'height="{s["height"]}" alt="frames over time: {e(c["title"])}"></details>')
    out.append("</article>")
    return "\n".join(out)


def index_html(cases, prefix="", header=None):
    e = html.escape
    parts = ['<!doctype html>', '<html lang="en">', '<head>', '<meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>MoPlan showcase</title>', f'<style>{CSS}</style>', '</head>', '<body>', '<main>']
    parts += header if header else ['<h1>MoPlan: video comparisons with the untrained base model</h1>']
    parts += [
             f'<p>{e(INTRO)}</p>', f'<p class="muted">{e(SELECTION)}</p>',
             '<p class="legend">' + "".join(f"<span>{e(x)}</span>" for x in LEGEND) + '</p>', '<nav>']
    used = [g for g in GROUPS if any(c["finding_group"] == g[0] for c in cases)]
    for key, heading, _ in used:
        nums = [c["number"] for c in cases if c["finding_group"] == key]
        rng = f"#{nums[0]}" if len(nums) == 1 else f"#{nums[0]}–{nums[-1]}"
        parts.append(f'<a href="#{e(key.lower().replace(" ", "-"))}">{e(heading)} ({rng})</a>')
    parts.append('</nav>')
    for key, heading, text in used:
        parts.append(f'<h2 id="{e(key.lower().replace(" ", "-"))}">{e(heading)}</h2>')
        parts.append(f'<p>{e(text)}</p>')
        parts += [card_html(c, prefix) for c in cases if c["finding_group"] == key]
    parts += [f'<footer>{e(FOOTER)}</footer>', '</main>', '</body>', '</html>', '']
    return "\n".join(parts)


def root_html(cases, prefix=""):
    """Video-only page (served at the site root): numbered comparison videos with a one-line label each."""
    e = html.escape
    parts = ['<!doctype html>', '<html lang="en">', '<head>', '<meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>Video comparisons</title>', f'<style>{CSS}</style>', '</head>', '<body>', '<main>',
             '<h1>Video comparisons</h1>',
             '<p class="muted">Panels from left to right: real video (when available), untrained Wan2.2 '
             '(same first frame, prompt and seed), MoPlan.</p>']
    source = {"MoProbe-H": "Something-Something v2", "Something-Something v2 validation frame": "Something-Something v2",
              "MoProbe-P": "PisaBench", "Physics-IQ Verified": "Physics-IQ"}
    for c in cases:
        f = c["files"]; v = f["video"]; fo = prefix + c["folder"]
        title = c["title"].split(" (")[0]
        parts += [f'<article class="card" id="case-{c["number"]:02d}">',
                  f'<h3><span class="num">#{c["number"]}</span>{e(source.get(c["benchmark"], c["benchmark"]))}: {e(title)}</h3>',
                  f'<video controls loop muted playsinline preload="metadata" width="{v["width"]}" height="{v["height"]}" '
                  f'poster="{e(fo)}/{e(f["poster"]["path"])}"><source src="{e(fo)}/{e(v["path"])}" type="video/mp4"></video>']
        if c.get("page_note"):
            parts.append(f'<p class="muted">{e(c["page_note"])}</p>')
        parts.append('</article>')
    parts += ['</main>', '</body>', '</html>', '']
    return "\n".join(parts)


def cmd_build(_args):
    cases = load_cases()
    (HERE.parent.parent / "index.html").write_text(root_html(cases, prefix="assets/showcase/"))
    readme = HERE / "README.md"
    text = readme.read_text()
    a, b = text.index(TABLE_START) + len(TABLE_START), text.index(TABLE_END)
    readme.write_text(text[:a] + "\n" + readme_table(cases) + "\n" + text[b:])
    print(f"wrote the site page (index.html at the repository root) and the README table ({len(cases)} cases)")


def cmd_check(_args):
    bad = 0
    total = 0
    listed = {HERE / "README.md", HERE / "showcase.py"}
    for c in load_cases():
        d = HERE / c["folder"]
        listed.add(d / "meta.json")
        for key, f in c["files"].items():
            p = d / f["path"]
            listed.add(p)
            if not p.is_file():
                print(f"MISSING {c['folder']}/{f['path']}")
                bad += 1
                continue
            total += p.stat().st_size
            if p.stat().st_size != f["bytes"] or sha256(p) != f["sha256"]:
                print(f"MISMATCH {c['folder']}/{f['path']}")
                bad += 1
    extra = [p for p in HERE.rglob("*") if p.is_file() and p not in listed and "__pycache__" not in p.parts]
    for p in extra:
        print(f"UNLISTED {p.relative_to(HERE)}")
    print(f"{'OK' if not bad else 'FAILED'}: {len(load_cases())} cases, {total / 1e6:.1f} MB of media, "
          f"{bad} problems, {len(extra)} unlisted files")
    return 1 if bad else 0


def cmd_commands(args):
    cases = load_cases()
    numbers = [c["number"] for c in cases]
    if args.case is not None and args.case not in numbers:
        print(f"error: there is no case {args.case}; the cases are {', '.join(map(str, numbers))}", file=sys.stderr)
        return 2
    for c in cases:
        if args.case is not None and c["number"] != args.case:
            continue
        for _row, p in generated_panels(c):
            if args.arm and p["arm"] != args.arm:
                continue
            print(p["command"])
    return 0


def cmd_verify(args):
    root = Path(args.out_dir)
    n_match = n_diff = n_missing = n_unverifiable = 0
    for c in load_cases():
        for _row, p in generated_panels(c):
            ref = p["reference"]
            stem = root / c["folder"] / p["key"]
            if ref.get("verifiable") is False:
                # the reference is not of a file generate.py writes, so a hash comparison would always fail
                n_unverifiable += 1
                print(f"NOT-VERIFIABLE {c['folder']}/{p['key']} (reference is an evaluation export: {ref['kind']})")
                continue
            found = False
            for ext in (".npy", ".mp4"):
                f = stem.with_suffix(ext)
                if not f.is_file():
                    continue
                found = True
                h = sha256(f)
                if ext == ".npy":
                    want = {"npy": ref.get("sha256")}
                else:
                    want = {"raw mp4": ref.get("raw_mp4_sha256"), "trimmed mp4": ref.get("trimmed_mp4_sha256")}
                hit = [k for k, v in want.items() if v and v == h]
                if not any(want.values()):
                    print(f"NO-REF   {c['folder']}/{p['key']}{ext} (this panel has a reference for the other format)")
                elif hit:
                    n_match += 1
                    print(f"MATCH    {c['folder']}/{p['key']}{ext} ({hit[0]})")
                else:
                    n_diff += 1
                    print(f"DIFFERS  {c['folder']}/{p['key']}{ext}")
            if not found:
                n_missing += 1
    print(f"{n_match} match, {n_diff} differ, {n_missing} panels without output, "
          f"{n_unverifiable} not verifiable (reference is an evaluation export)")
    return 1 if n_diff else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="rewrite index.html and the README case table from meta.json")
    sub.add_parser("check", help="verify the assets against meta.json")
    c = sub.add_parser("commands", help="print the generate.py commands of the generated panels")
    c.add_argument("--case", type=int, default=None, help="only this case number (an unknown number is an error)")
    c.add_argument("--arm", choices=["moplan", "base"], default=None, help="only this model")
    v = sub.add_parser("verify", help="compare regenerated panels with the reference hashes")
    v.add_argument("out_dir", help="directory with <NN>_<id>/<key>.npy or .mp4 files (default commands: out/showcase)")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "check": cmd_check, "commands": cmd_commands, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
