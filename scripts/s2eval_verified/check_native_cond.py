#!/usr/bin/env python
"""GPU check (~1 min): the native profile's condition path reproduces a Stage-2 condition cache of the same views.

For a few Physics-IQ views it renders the condition exactly as NativeGenerator.condition does (switch frame -> PIL LANCZOS
448x256 -> Wan VAE posterior mean of (frame ++ zeros) -> plan from the Stage-1 export at 256x448 -> render_v2("pred")
-> fp16), but at the clip length of the cache (e.g. 37 frames @7 fps), and compares with a condition cache made from the
same op exports (<cache>/<sid>.pt: image_cond_latent and cond_pred).
usage: check_native_cond.py --exports DIR --cache DIR --switch DIR --wan WAN_MODEL_DIR --ids 0001_...,0002_... --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exports", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--switch", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--wan", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from diffusers import AutoencoderKLWan
    from PIL import Image
    from scripts import motion_plan as mp
    from scripts.cache_stage2_oracle import preprocess
    from scripts.control import plan_condition as pc
    vae = AutoencoderKLWan.from_pretrained(Path(a.wan) / "vae", torch_dtype=torch.float32).to("cuda").eval()
    rows = []
    for sid in a.ids.split(","):
        cache = torch.load(Path(a.cache) / f"{sid}.pt", map_location="cpu", weights_only=False)
        export = torch.load(Path(a.exports) / f"{sid}.pt", map_location="cpu", weights_only=False)
        frames_n, fps, h, w = int(cache["frames"]), float(cache["fps"]), int(cache["height"]), int(cache["width"])
        frame = next(Path(a.switch).glob(f"{sid[:4]}_switch-frames_*.jpg"))
        image = Image.open(frame).convert("RGB").resize((w, h), Image.LANCZOS)
        frames = np.zeros((frames_n, h, w, 3), np.uint8)
        frames[0] = np.asarray(image)
        video = preprocess(frames, h, w).to("cuda")
        image_video = torch.zeros_like(video)
        image_video[:, :, :1] = video[:, :, :1]
        with torch.no_grad():
            latent = vae.encode(image_video).latent_dist.mode()[0].float().cpu()
        times = pc.latent_times(frames_n, fps)
        plan = mp.from_stage1_export(export, (h, w), native_hw=(2160, 3840))
        cond = pc.render_v2("pred", plan, latent, times, (h, w), export=export, native_hw=(2160, 3840)).half().float()
        ref_lat, ref = cache["image_cond_latent"].float(), cache["cond_pred"].float()
        same_lat = pc.render_v2("pred", plan, ref_lat, times, (h, w), export=export, native_hw=(2160, 3840)).half().float()
        rows.append({"sid": sid, "frames": frames_n, "fps": fps, "times": [round(t, 4) for t in times],
                     "times_cache": cache["frame_seconds"],
                     "image_latent_max_abs_diff": float((latent - ref_lat).abs().max()),
                     "image_latent_ref_absmax": float(ref_lat.abs().max()),
                     "cond_max_abs_diff_per_channel_group": {"warped_latent_0_15": float((cond[:16] - ref[:16]).abs().max()),
                                                             "flow_mask_vis_16_19": float((cond[16:20] - ref[16:20]).abs().max()),
                                                             "extra_20_26": float((cond[20:] - ref[20:]).abs().max())},
                     "cond_from_cache_latent_identical": bool(torch.equal(same_lat, ref))})
        print(json.dumps(rows[-1]), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
