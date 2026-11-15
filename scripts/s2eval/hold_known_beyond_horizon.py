"""Make Stage-1-prediction conditions follow the training contract beyond the Stage-1 horizon.

Stage-1 predicts up to 4 s; the 27-channel legacy27 renderer then marks later latents "unknown" (channel 20 `known` = 0,
positions merely held).  The generator was trained on 41-frame windows whose late latents are the HELD plan with
known = 1 (scenes at rest extended with their last frame), so known = 0 never occurs in training and the generator
changes abruptly right after the last known latent.  This tool rewrites a condition cache so that every latent after
the last one that carries `known` holds that latent's full condition (positions, known, availability, extras):
"the predicted end state persists", exactly the training-side padding.  The source cache is never modified.
All released MoPlan results read conditions processed this way (generate.py applies the same rule in memory).

Edge case (kept as is, because the released outputs were made with this code): the last known latent is the last one
whose `known` channel is non-zero anywhere, and `known` is drawn only on the objects' transformed footprints.  If every
planned object (static ones included) is outside the frame from some latent time before 4 s on, those latents have
known = 0 everywhere, so the hold starts at an earlier latent and overwrites later latents, including ones up to 4 s,
with that earlier condition.  The intended rule holds from the last Stage-1 horizon (4 s) on.  The receipt's `last_known_hist`
shows where the hold started; see docs/INFERENCE.md.

usage: python hold_known_beyond_horizon.py SRC_CACHE_DIR DST_CACHE_DIR [--keys cond_pred]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch

KNOWN = 20  # cond_v2 legacy27: channels 0-19 base, 20 known, 21 availability, 22 selector, 23/24 raw flow, 25 motion_prob, 26 source


def hold(x: torch.Tensor):
    """x: (27, L, H, W).  Returns (new x, last known latent, number of latents filled).

    The last known latent is found from the footprint of channel 20, not from the latent times (see the module
    docstring for the case where every object leaves the frame before 4 s)."""
    k = x[KNOWN].flatten(1).abs().sum(-1)                       # (L,)
    known = torch.nonzero(k > 0).flatten()
    if len(known) == 0:
        return x, -1, 0
    last = int(known[-1])
    if last == x.shape[1] - 1:
        return x, last, 0
    y = x.clone()
    y[:, last + 1:] = x[:, last:last + 1]
    return y, last, x.shape[1] - 1 - last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--keys", default="cond_pred")
    a = p.parse_args()
    keys = a.keys.split(",")
    a.dst.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "filled_files": 0, "filled_latents": 0, "last_known_hist": {}}
    for f in sorted(a.src.iterdir()):
        if f.suffix != ".pt":
            if f.is_file():
                shutil.copy2(f, a.dst / f.name)          # ids lists, receipts, manifests
            elif f.is_dir() and not (a.dst / f.name).exists():
                shutil.copytree(f, a.dst / f.name, copy_function=os.link)   # e.g. scenes_v5 (reward scenes): hard links
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        n_fill = 0
        for key in keys:
            if key in d and hasattr(d[key], "shape") and d[key].dim() == 4 and d[key].shape[0] == 27:
                y, last, n = hold(d[key].float())
                d[key] = y.to(d[key].dtype)
                n_fill += n
                stats["last_known_hist"][str(last)] = stats["last_known_hist"].get(str(last), 0) + 1
        d["hold_known_beyond_horizon"] = True
        torch.save(d, a.dst / f.name)
        stats["files"] += 1
        stats["filled_files"] += int(n_fill > 0)
        stats["filled_latents"] += n_fill
    stats.update(src=a.src.resolve().name, keys=keys, rule="latents after the last known latent copy that latent's full 27-ch condition")
    (a.dst / "HOLD_KNOWN_RECEIPT.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps({k: v for k, v in stats.items()}))


if __name__ == "__main__":
    main()
