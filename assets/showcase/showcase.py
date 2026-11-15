"""Showcase helper: render the video page, check the assets, list and verify regenerated panels.

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
:root { --fg: #1d1d1f; --muted: #5f6368; --bg: #ffffff; --card: #f6f7f9; --line: #dfe1e5; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e8eaed; --muted: #a0a4a8; --bg: #16181b; --card: #202327; --line: #33373c; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 system-ui, sans-serif; }
main { max-width: 1400px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.7rem; margin: 0 0 8px; }
p { margin: 6px 0; }
.muted { color: var(--muted); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
  margin: 18px 0; }
.card h3 { margin: 0 0 8px; font-size: 1.1rem; }
.num { color: var(--muted); margin-right: 6px; }
video { display: block; max-width: 100%; height: auto; border-radius: 6px; background: #000; }
"""


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
                if not any(stem.with_suffix(ext).is_file() for ext in (".npy", ".mp4")):
                    n_missing += 1
                    print(f"NO-OUTPUT {c['folder']}/{p['key']}")
                    continue
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
