#!/usr/bin/env python3
"""Does a prompt-embedding file of scripts/infer_stage2_cond_v2.py --phase embed hold every prompt a run will look up?

launch/eval_bench_generate.sh keeps one embedding file per benchmark ($EVAL_ROOT/<bench>/prompt_embeds.pt) for every
model, and the high / low phases look each clip's generator prompt up in it.  This check runs before any model is
loaded: exit 0 when the file exists and holds the generator prompt of every id (the --prompt-table entry of the id, else
the "prompt" of its condition cache, as infer_stage2_cond_v2.py resolves it) and was encoded with the run's negative
prompt (the text of --negative-prompt-file, else Wan2.2's official one, as infer_stage2_cond_v2.py resolves it); exit 1
(with the reason on stderr) otherwise, so the caller encodes it again.
The caches are memory-mapped, so only their small fields are read.

usage: embeds_cover.py --embeds E.pt --cache-dir CACHE --ids IDS.txt [--prompt-table T.json] [--negative-prompt-file N.txt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def load(path: Path, **kwargs):
    try:
        return torch.load(path, map_location="cpu", mmap=True, **kwargs)
    except (RuntimeError, TypeError):                            # not a zip-format file (or an old torch): no memory map
        return torch.load(path, map_location="cpu", **kwargs)


def default_negative_prompt() -> str:
    """Wan2.2's official negative prompt: what infer_stage2_cond_v2.py uses without --negative-prompt"""
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts.infer_stage2_oracle import WAN_BENCHMARK_NEGATIVE_PROMPT
    return WAN_BENCHMARK_NEGATIVE_PROMPT


def missing_reason(embeds: Path, cache_dir: Path, ids: list[str], table: dict, negative: str | None) -> str | None:
    """None when the file covers the run, else why not (negative: the run's negative prompt; None = Wan2.2's default)"""
    if not embeds.is_file():
        return f"{embeds} does not exist"
    have = load(embeds, weights_only=True)
    if have.get("negative_prompt") != (negative or default_negative_prompt()):
        return f"{embeds} was encoded with another negative prompt than the run's"
    need = {table[sid] if sid in table else load(cache_dir / f"{sid}.pt", weights_only=False)["prompt"] for sid in ids}
    lack = need - set(have["prompts"])
    if lack:
        return f"{embeds} lacks {len(lack)} of the {len(need)} prompts of the {len(ids)} ids"
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeds", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ids", required=True, help="sample ids, one per line")
    p.add_argument("--prompt-table", default="", help="json sample_id -> generator prompt (as --prompt-table of the run)")
    p.add_argument("--negative-prompt-file", default="", help="the run's --negative-prompt, as a text file")
    a = p.parse_args(argv)
    ids = [line.strip() for line in Path(a.ids).read_text().splitlines() if line.strip()]
    present = lambda f: bool(f) and Path(f).is_file() and Path(f).stat().st_size > 0   # noqa: E731  (the launcher's [ -s f ])
    table = json.loads(Path(a.prompt_table).read_text()) if present(a.prompt_table) else {}
    negative = None                                              # the launcher passes "$(cat FILE)": trailing newlines dropped
    if present(a.negative_prompt_file):
        negative = Path(a.negative_prompt_file).read_text().rstrip("\n")
    reason = missing_reason(Path(a.embeds), Path(a.cache_dir), ids, table, negative)
    if reason:
        print(reason, file=sys.stderr)
        return 1
    print(f"{a.embeds} covers the {len(ids)} ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
