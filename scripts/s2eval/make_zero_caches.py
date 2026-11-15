#!/usr/bin/env python3
"""CPU, zero-flow ablation: copy finished Stage-2 evaluation caches with cond_pred REPLACED by that cache's cond_zero
(the same Stage-1 objects rendered at rest) after the hold-known rule (hold_known_beyond_horizon.py: latents past the
4 s Stage-1 horizon copy the last known latent, as cond_pred already does in held caches).  Every other key (image
latent, prompt, cond_real / cond_null, ...) is kept, so infer_stage2_cond_v2.py runs unchanged with arm pred.
Sources are never modified; non-.pt files are copied, sub-directories hard-linked; a receipt ZERO_PLAN_RECEIPT.json is
written per cache.

Layout: <ev>/s1/<src>/caches/<bench>  ->  <ev>/s1/<dst>/caches/<bench>
usage: make_zero_caches.py --ev DIR --src HELD_TAG [--dst zeroh] [--benches KEY[,KEY...]]
       (launch/eval_bench_prepare.sh passes the one cache key of its benchmark, see bench_key in launch/eval_common.sh)
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hold_known_beyond_horizon import hold  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ev", type=Path, required=True, help="evaluation root holding s1/<tag>/caches/<bench>")
    p.add_argument("--src", required=True, help="tag of the held final-model caches")
    p.add_argument("--dst", default="zeroh", help="tag of the zero-flow caches to write")
    p.add_argument("--benches", default="piq,pisa,p101,rigid,morph,miqA")
    a = p.parse_args(argv)
    dst_root = a.ev / "s1" / a.dst
    for bench in a.benches.split(","):
        src, dst = a.ev / "s1" / a.src / "caches" / bench, dst_root / "caches" / bench
        if (dst / "ZERO_PLAN_RECEIPT.json").is_file():
            print(f"{bench}: complete"); continue
        if dst.exists():
            shutil.rmtree(dst)
        tmp = dst.with_name(bench + ".partial"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        n = filled = 0
        for f in sorted(src.iterdir()):
            if f.suffix != ".pt":
                if f.is_file():
                    shutil.copy2(f, tmp / f.name)
                elif f.is_dir():
                    shutil.copytree(f, tmp / f.name, copy_function=os.link)
                continue
            d = torch.load(f, map_location="cpu", weights_only=False)
            z = d["cond_zero"]
            y, _, k = hold(z.float())
            d["cond_pred"] = y.to(z.dtype)
            d["zero_plan_as_pred"] = True
            torch.save(d, tmp / f.name)
            n += 1; filled += int(k > 0)
        (tmp / "ZERO_PLAN_RECEIPT.json").write_text(json.dumps({"src": str(src), "rule": "cond_pred := hold_known(cond_zero)",
                                                                 "files": n, "held_files": filled}, indent=1) + "\n")
        tmp.rename(dst)
        print(f"{bench}: {n} caches ({filled} held past the horizon)")
    dst_root.mkdir(parents=True, exist_ok=True)
    print("ZERO_CACHES_DONE")


if __name__ == "__main__":
    main()
