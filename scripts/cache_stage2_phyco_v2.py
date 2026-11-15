#!/usr/bin/env python
"""Stage-2 PhyCo cache, v2: deployment-consistent geometry + Stage-1 predicted conditions.

Differences from cache_stage2_phyco.py:

* content geometry.  The Stage-1 grid is a 160x160 square holding the 16:9 frame letterboxed
  (content rows 35..124).  v1 stretched that whole square onto the bar-free 256x448 latent, so
  the condition said the object sat at latent rows 7..25 while the video had it at 0..31.
  Here the content box is cropped out of the grid and resampled onto the inner box of the
  256x448 work frame before ``make_motion_condition`` -- exactly what the deployment (Physics-IQ)
  cache does, so training and Physics-IQ conditions share one geometry.
* conditions per clip, all built through the same ``make_motion_condition``:
    motion_condition                      ground truth (targets)
    motion_condition_pred__<field>        the Stage-1 prediction read out as <field>
                                          (scripts/piq_gt_v2/stage1_pred_fields.py: raw / soft /
                                          binary_<tau>); mask = |flow| > 1.5 px (any horizon) dilated
                                          7x7, visibility = sigmoid(logits) > .5 -- the deployment rule
    motion_condition_pred_shuffled__<field>   (--shuffle) the same prediction with every instance's
                                          flow vectors rotated by one angle per instance: same support,
                                          same per-pixel magnitude, same background, wrong direction;
                                          future mask / visibility / warped latent rebuilt along it
    motion_condition_zero                 zero displacement with an empty mask, what the deployment
                                          chain produces when Stage-1 abstains
    motion_condition_pred                 alias of the first <field> (legacy single-field readers)
* --ids-file / --video-root: clips are read from <root>/<id>.mp4, not from target["video_path"].
* --captions: the prompt is the manifest caption Stage-1 was given (reviewed captions), not the
  caption stored in the target, so both stages read the same text for a clip.
* --base-cache: reuse video_latent / image_cond_latent / ground-truth and zero conditions of an existing
  v2 cache (they do not depend on Stage-1) and rebuild only the predicted conditions and the prompt --
  no VAE, runs on CPU when the prompt table (--prompt-table) covers the captions.

Everything the Stage-2 trainer reads (video_latent / image_cond_latent / motion_condition* /
prompt_embeds) keeps the v1 names and dtypes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_jepa.video import forward_splat_mask
from scripts.cache_stage2_oracle import make_motion_condition
from scripts.piq_gt_v2.stage1_pred_fields import canonical, displacement_jump, resolve_field, rotate_objects

CACHE_FORMAT_VERSION = 1          # == scripts.cache_stage2_phyco.CACHE_FORMAT_VERSION (kept import-free of cv2)
GEOMETRY = "content_box_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default="", help="Stage-1 target cache (7 or 14 anchors); optional with --base-cache")
    parser.add_argument("--ids-file", default="")
    parser.add_argument("--video-root", default="", help="directory with <sample_id>.mp4")
    parser.add_argument("--predicted-flows", default="", help="eval_stage1_flowdit.py export dir (<id>.pt)")
    parser.add_argument("--pred-field", default="flow_task_object_gated", help="single readout (legacy spelling accepted)")
    parser.add_argument("--pred-fields", default="", help="comma list of readouts, e.g. raw,soft,binary_0.5,binary_0.7")
    parser.add_argument("--pred-source", default="", help="free-text provenance of the prediction")
    parser.add_argument("--output", default="")
    parser.add_argument("--model-dir", default=os.environ.get("WAN_MODEL_DIR", ""),
                        help="Wan2.2-I2V-A14B-Diffusers directory (default: $WAN_MODEL_DIR)")
    parser.add_argument("--captions", action="append", default=[],
                        help="manifest jsonl (sample_id, caption): the prompt of each clip; repeatable")
    parser.add_argument("--prompt-table", default="", help="caption -> embedding table (see --encode-prompt-table)")
    parser.add_argument("--encode-prompt-table", default="",
                        help="only encode the unique --captions into this table and exit")
    parser.add_argument("--base-cache", default="", help="existing v2 cache to take latents and GT/zero conditions from")
    parser.add_argument("--native-size", default="", help="HxW of the source clips when --video-root is not given")
    parser.add_argument("--shuffle", action="store_true", help="also write the direction-shuffled control conditions")
    parser.add_argument("--shuffle-seed", type=int, default=20261011)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--frames", type=int, default=29)
    parser.add_argument("--span-seconds", type=float, default=4.0)
    parser.add_argument("--source-fps", type=float, default=24.0)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--move-px", type=float, default=1.5, help="deployment mask threshold (work-frame px)")
    parser.add_argument("--geometry", default="content_box_v2", choices=("content_box_v2", "v1_stretch"),
                        help="v1_stretch = control: the whole letterboxed 160x160 Stage-1 grid stretched onto the work frame, "
                             "as cache_stage2_phyco.py (v1) did")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def inner_box(height: int, width: int, native_h: int, native_w: int) -> tuple[int, int, int, int]:
    """(top, left, inner_h, inner_w) of a native_h x native_w frame letterboxed into height x width."""
    scale = min(height / native_h, width / native_w)
    inner_h, inner_w = round(native_h * scale), round(native_w * scale)
    return (height - inner_h) // 2, (width - inner_w) // 2, inner_h, inner_w


def grid_to_work(field: torch.Tensor, grid_box, work_box, height: int, width: int,
                 mode: str, scale_values: bool) -> torch.Tensor:
    """Crop the content box out of the Stage-1 grid [T,C,side,side] and resample it onto the inner
    box of the work frame; displacement values are scaled per axis with the coordinates."""
    g_top, g_left, g_h, g_w = grid_box
    w_top, w_left, w_h, w_w = work_box
    content = field[:, :, g_top: g_top + g_h, g_left: g_left + g_w].float()
    resized = F.interpolate(content, size=(w_h, w_w), mode=mode, **({"align_corners": False} if mode == "bilinear" else {}))
    if scale_values:
        resized[:, 0] *= w_w / g_w
        resized[:, 1] *= w_h / g_h
    out = field.new_zeros((field.shape[0], field.shape[1], height, width), dtype=torch.float32)
    out[:, :, w_top: w_top + w_h, w_left: w_left + w_w] = resized
    return out


def dynamic_source_mask(flow: torch.Tensor, visibility: torch.Tensor, move_px: float) -> torch.Tensor:
    """Deployment rule (Physics-IQ cache): reliable nontrivial motion, dilated 7x7."""
    reliable_magnitude = flow.square().sum(1).sqrt() * visibility[:, 0]
    mask = reliable_magnitude.amax(0, keepdim=True) > move_px
    return F.max_pool2d(mask.float()[None], kernel_size=7, stride=1, padding=3)[0]


def native_size(path: Path) -> tuple[int, int]:
    import cv2
    capture = cv2.VideoCapture(str(path))
    size = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture.release()
    if min(size) <= 0:
        raise RuntimeError(f"cannot read {path}")
    return size


def nearest_anchor(horizons: list[float], when: float) -> int:
    index = min(range(len(horizons)), key=lambda h: abs(horizons[h] - when))
    if abs(horizons[index] - when) > 0.05:
        raise ValueError(f"no anchor near {when:.3f}s (have {horizons})")
    return index


def load_captions(paths: list[str]) -> dict[str, str]:
    captions: dict[str, str] = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if captions.get(row["sample_id"], row["caption"]) != row["caption"]:
                    raise ValueError(f"{row['sample_id']}: conflicting captions across --captions manifests")
                captions[row["sample_id"]] = row["caption"]
    return captions


class PromptEncoder:
    """umT5 embeddings exactly as the Stage-2 caches have always stored them (unpadded, bf16)."""

    def __init__(self, model_dir: str, prompt_length: int, table_path: str = "") -> None:
        self.model_dir, self.prompt_length = model_dir, int(prompt_length)
        self.table: dict[str, torch.Tensor] = {}
        if table_path:
            stored = torch.load(table_path, map_location="cpu", weights_only=True)
            if int(stored["prompt_length"]) != self.prompt_length:
                raise ValueError("prompt table was encoded with a different --prompt-length")
            self.table = dict(stored["embeddings"])
        self.tokenizer = self.encoder = None
        self.encoded = 0

    def _load(self) -> None:
        from transformers import T5TokenizerFast, UMT5EncoderModel
        if not self.model_dir:
            raise RuntimeError("a caption is missing from the prompt table and no --model-dir was given")
        root = Path(self.model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5TokenizerFast.from_pretrained(root / "tokenizer", local_files_only=True)
        self.encoder = UMT5EncoderModel.from_pretrained(
            root / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
        ).to(self.device).eval()

    def __call__(self, prompt: str) -> torch.Tensor:
        if prompt not in self.table:
            if self.encoder is None:
                self._load()
            tokens = self.tokenizer([prompt], padding="max_length", max_length=self.prompt_length, truncation=True,
                                    add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
            attention = tokens.attention_mask.to(self.device)
            length = max(int(attention.sum()), 1)
            with torch.no_grad():
                self.table[prompt] = self.encoder(
                    input_ids=tokens.input_ids.to(self.device), attention_mask=attention
                ).last_hidden_state[0, :length].cpu().to(torch.bfloat16)
            self.encoded += 1
        return self.table[prompt]


def dilate_assignment(masks: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    """[K,H,W] bool instance masks grown into the unclaimed pixels around them (first slot wins).

    The flow field was upsampled bilinearly, so an object's vectors bleed a pixel or two past its
    nearest-neighbour mask; the halo has to rotate with the object it belongs to."""
    grown = F.max_pool2d(masks.float()[None], kernel_size=kernel, stride=1, padding=kernel // 2)[0] > 0
    assignment, free = masks.clone(), ~masks.any(dim=0)
    for slot in range(masks.shape[0]):
        take = grown[slot] & free
        assignment[slot] |= take
        free &= ~take
    return assignment


def main() -> None:
    args = parse_args()
    geometry_tag = GEOMETRY if args.geometry == "content_box_v2" else "v1_stretch_control"
    captions = load_captions(args.captions)
    if args.encode_prompt_table:
        encoder = PromptEncoder(args.model_dir, args.prompt_length,
                                args.prompt_table if args.prompt_table and Path(args.prompt_table).is_file() else "")
        unique = sorted(set(captions.values()))
        for caption in unique:
            encoder(caption)
        torch.save({"prompt_length": args.prompt_length, "embeddings": {c: encoder.table[c] for c in unique},
                    "encoder": "umt5 text_encoder of " + str(args.model_dir), "dtype": "bfloat16"},
                   args.encode_prompt_table)
        print(json.dumps({"prompt_table": args.encode_prompt_table, "captions": len(unique), "encoded_now": encoder.encoded}))
        return

    for name in ("ids_file", "predicted_flows", "output"):
        if not getattr(args, name):
            raise SystemExit(f"--{name.replace('_', '-')} is required")
    base_dir = Path(args.base_cache) if args.base_cache else None
    if base_dir is None and not (args.targets and args.video_root and args.model_dir):
        raise SystemExit("without --base-cache, --targets, --video-root and --model-dir are required")
    if base_dir is None and not torch.cuda.is_available():
        raise RuntimeError("Stage-2 caching from the raw clips requires CUDA (VAE); use --base-cache on CPU")
    ids = [line.strip() for line in Path(args.ids_file).read_text().splitlines() if line.strip()]
    if args.limit:
        ids = ids[: args.limit]
    ids = ids[args.shard_index :: args.num_shards]
    fields = [canonical(f) for f in (args.pred_fields.split(",") if args.pred_fields else [args.pred_field]) if f.strip()]
    if len(set(fields)) != len(fields) or not fields:
        raise SystemExit(f"bad --pred-fields: {fields}")
    print(f"{len(ids)} clips on shard {args.shard_index}/{args.num_shards}; fields {fields}; "
          f"shuffle={args.shuffle}; base={'yes' if base_dir else 'no'}; captions={len(captions)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() and base_dir is None else "cpu")
    model_dir = Path(args.model_dir) if args.model_dir else Path("")
    vae = None
    if base_dir is None:
        from diffusers import AutoencoderKLWan
        from scripts.cache_stage2_phyco import preprocess, read_letterboxed
        vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32).to(device).eval()
    encode_prompt = PromptEncoder(args.model_dir, args.prompt_length, args.prompt_table)
    fixed_native = tuple(int(v) for v in args.native_size.lower().split("x")) if args.native_size else None

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seconds = [args.span_seconds * i / (args.frames - 1) for i in range(args.frames)]
    selected = list(range(0, args.frames, 4))
    future_seconds = [seconds[i] for i in selected[1:]]
    height, width = args.height, args.width
    completed = skipped = failed = 0
    errors: list[dict] = []
    stats = {field: {"peak_px": 0.0, "mask_fraction": 0.0, "abstain": 0, "jump_max": 0.0} for field in fields}
    shuffle_stats = {"magnitude_max_abs_diff_px": 0.0, "outside_max_abs_diff_px": 0.0, "support_mismatch": 0,
                     "rotated_fraction_of_support": 0.0, "direction_cosine": 0.0, "clips": 0}
    prompt_sources: dict[str, int] = {}

    for sample_id in ids:
        output = output_dir / f"{sample_id}.pt"
        if output.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            pred = torch.load(Path(args.predicted_flows) / f"{sample_id}.pt", map_location="cpu", weights_only=True)
            side = int(pred["flow"].shape[-1])
            video_path = Path(args.video_root) / f"{sample_id}.mp4" if args.video_root else None
            if video_path is not None:
                native_h, native_w = native_size(video_path)
                if fixed_native and fixed_native != (native_h, native_w):
                    raise ValueError(f"native size {(native_h, native_w)} != --native-size {fixed_native}")
            elif fixed_native:
                native_h, native_w = fixed_native
            else:
                raise ValueError("need --video-root or --native-size for the content geometry")
            grid_box = inner_box(side, side, native_h, native_w)        # (35, 0, 90, 160) for 768x432
            work_box = inner_box(height, width, native_h, native_w)     # (2, 0, 252, 448)
            if args.geometry == "v1_stretch":                           # control: bars and all, stretched per axis
                grid_box, work_box = (0, 0, side, side), (0, 0, height, width)

            if base_dir is not None:
                base = torch.load(base_dir / f"{sample_id}.pt", map_location="cpu", weights_only=True)
                if (base.get("geometry") != geometry_tag or int(base["height"]) != height or int(base["width"]) != width
                        or int(base["frames"]) != args.frames or list(base["latent_frame_indices"]) != selected):
                    raise ValueError("base cache payload has a different geometry or frame sampling")
                video_latent, image_cond_latent = base["video_latent"], base["image_cond_latent"].float()
                gt_condition, zero_condition = base["motion_condition"], base["motion_condition_zero"]
                meta = {key: base.get(key, "") for key in ("scenario", "label_source", "flow_source", "instance_source",
                                                           "caption_source")}
                gt_peak = float(base.get("pred_summary", {}).get("gt_peak_px", float("nan")))
                fallback_prompt, fallback_embeds = base["prompt"], base["prompt_embeds"]
                decode = base.get("decode", {})
            else:
                target = torch.load(Path(args.targets) / f"{sample_id}.pt", map_location="cpu", weights_only=True)
                frames = read_letterboxed(video_path, height, width, seconds, args.source_fps)
                # decode fingerprint: the clips carry no colour metadata, so a different swscale build can decode
                # them a few levels off; whoever scores generations against the raw clip can check that it
                # decodes the frames this cache was built from
                decode = {"frames_sha1": hashlib.sha1(frames.tobytes()).hexdigest(), "opencv": __import__("cv2").__version__,
                          "mean_rgb": [round(float(v), 4) for v in frames.reshape(-1, 3).mean(0)]}
                with torch.no_grad():
                    video = preprocess(frames).to(device)
                    video_latent = vae.encode(video).latent_dist.mode()[0].cpu().half()
                    image_video = torch.cat((video[:, :, :1], video.new_zeros(*video.shape[:2], video.shape[2] - 1,
                                                                                *video.shape[3:])), dim=2)
                    image_cond_latent = vae.encode(image_video).latent_dist.mode()[0].cpu()
                # ground truth: the anchors nearest to the sampled latent frames
                gt_horizons = [float(h) for h in target["horizon_seconds"]]
                gt_index = [nearest_anchor(gt_horizons, when) for when in future_seconds]
                gt_flow = grid_to_work(target["flow"].float()[gt_index], grid_box, work_box, height, width, "bilinear", True)
                gt_vis = grid_to_work(target["visibility"].float()[gt_index], grid_box, work_box, height, width, "nearest", False)
                gt_future = grid_to_work(target["future_mask"].float()[gt_index], grid_box, work_box, height, width, "nearest", False)
                gt_object = grid_to_work(target["object_mask"].float()[None], grid_box, work_box, height, width, "nearest", False)[0]
                gt_target = {"flow": gt_flow, "visibility": gt_vis, "object_mask": gt_object, "future_mask": gt_future}
                zero_target = {"flow": torch.zeros_like(gt_flow), "visibility": torch.ones_like(gt_vis),
                               "object_mask": torch.zeros_like(gt_object), "future_mask": torch.zeros_like(gt_future)}
                build = lambda built: make_motion_condition(                                 # noqa: E731
                    image_cond_latent, built, source_height=height, source_width=width,
                    selected_indices=list(range(len(selected))), future_indices=list(range(1, len(selected))),
                ).cpu().half()
                gt_condition, zero_condition = build(gt_target), build(zero_target)
                image_cond_latent_store = image_cond_latent.half()
                meta = {"scenario": target["scenario"], "label_source": target["label_source"],
                        "flow_source": target.get("flow_source", ""), "instance_source": target.get("instance_source", ""),
                        "caption_source": target.get("caption_source", "")}
                gt_peak = float(gt_flow.square().sum(1).sqrt().max())
                fallback_prompt = target.get("caption", "") or f"A physical simulation of {target['scenario']}."
                fallback_embeds = None

            def condition_of(built: dict) -> torch.Tensor:
                return make_motion_condition(
                    image_cond_latent, built, source_height=height, source_width=width,
                    selected_indices=list(range(len(selected))), future_indices=list(range(1, len(selected))),
                ).cpu().half()

            # Stage-1 prediction: stored normalised by the eval side at its own horizons
            pred_horizons = [float(h) for h in pred["horizon_seconds"]]
            pred_index = [nearest_anchor(pred_horizons, when) for when in future_seconds]
            eval_side = float(pred.get("eval_side", side))
            vis_logits = grid_to_work(pred["visibility_logits"].float()[pred_index], grid_box, work_box, height, width, "bilinear", False)
            pred_vis = (torch.sigmoid(vis_logits) > 0.5).float()
            pred_vis[:, :, : work_box[0]] = 0.0
            pred_vis[:, :, work_box[0] + work_box[2]:] = 0.0
            assignment = None
            if args.shuffle:
                slots = pred["activity_input_instance_masks"].float()[None]                       # [1, K, S, S]
                assignment = dilate_assignment(grid_to_work(slots, grid_box, work_box, height, width, "nearest", False)[0] > 0.5)

            conditions, contracts, pred_summary, shuffle_summary = {}, {}, {}, {}
            for field in fields:
                flow_norm, contract = resolve_field(pred, field)
                pred_flow = grid_to_work(flow_norm[pred_index] * eval_side, grid_box, work_box, height, width, "bilinear", True)
                pred_object = dynamic_source_mask(pred_flow, pred_vis, args.move_px)
                pred_future = forward_splat_mask(pred_object[None], pred_flow[None])[0]
                conditions[f"motion_condition_pred__{field}"] = condition_of(
                    {"flow": pred_flow, "visibility": pred_vis, "object_mask": pred_object, "future_mask": pred_future})
                contracts[field] = contract
                peak = float(pred_flow.square().sum(1).sqrt().max())
                pred_summary[field] = {"peak_px": peak, "mask_fraction": float(pred_object.mean()),
                                       "abstain": bool(peak < 0.5), "gt_peak_px": gt_peak,
                                       "jump": displacement_jump(pred_flow, pred_object[0])}
                stats[field]["peak_px"] += peak
                stats[field]["mask_fraction"] += float(pred_object.mean())
                stats[field]["abstain"] += int(peak < 0.5)
                stats[field]["jump_max"] = max(stats[field]["jump_max"], pred_summary[field]["jump"])
                if assignment is not None:
                    # direction shuffle: rotate each instance's vectors, keep support/visibility, rebuild the rest
                    rotated, angles = rotate_objects(pred_flow, assignment, sample_id, seed=args.shuffle_seed)
                    rotated_future = forward_splat_mask(pred_object[None], rotated[None])[0]
                    conditions[f"motion_condition_pred_shuffled__{field}"] = condition_of(
                        {"flow": rotated, "visibility": pred_vis, "object_mask": pred_object, "future_mask": rotated_future})
                    magnitude, rotated_magnitude = pred_flow.square().sum(1).sqrt(), rotated.square().sum(1).sqrt()
                    inside = assignment.any(dim=0)
                    support = pred_object[0] > 0
                    weight = (magnitude * (support & inside)[None]).clamp_min(0)
                    cosine = ((pred_flow * rotated).sum(1) / (magnitude * rotated_magnitude).clamp_min(1e-6) * weight).sum() \
                        / weight.sum().clamp_min(1e-6)
                    shuffle_summary[field] = {
                        "angles_deg": {str(k): round(v, 2) for k, v in angles.items()},
                        "magnitude_max_abs_diff_px": float((magnitude - rotated_magnitude).abs().max()),
                        "outside_max_abs_diff_px": float((rotated - pred_flow).abs().amax(dim=(0, 1))[~inside].max()) if bool((~inside).any()) else 0.0,
                        "support_recomputed_equal": bool(torch.equal(dynamic_source_mask(rotated, pred_vis, args.move_px), pred_object)),
                        "rotated_fraction_of_support": float((support & inside).sum() / support.sum().clamp_min(1)),
                        "direction_cosine": float(cosine),
                    }
            if assignment is not None:
                first = shuffle_summary[fields[0]]
                shuffle_stats["clips"] += 1
                for field in fields:
                    item = shuffle_summary[field]
                    shuffle_stats["magnitude_max_abs_diff_px"] = max(shuffle_stats["magnitude_max_abs_diff_px"], item["magnitude_max_abs_diff_px"])
                    shuffle_stats["outside_max_abs_diff_px"] = max(shuffle_stats["outside_max_abs_diff_px"], item["outside_max_abs_diff_px"])
                    shuffle_stats["support_mismatch"] += int(not item["support_recomputed_equal"])
                shuffle_stats["rotated_fraction_of_support"] += first["rotated_fraction_of_support"]
                shuffle_stats["direction_cosine"] += first["direction_cosine"]

            if captions:
                if sample_id not in captions:
                    raise KeyError("no caption in the --captions manifests")
                prompt, prompt_source = captions[sample_id], "captions_manifest"
            else:
                prompt, prompt_source = fallback_prompt, "base_cache" if base_dir is not None else "target"
            if fallback_embeds is not None and prompt == fallback_prompt:
                prompt_embeds, prompt_source = fallback_embeds, prompt_source + "+base_embedding"
            else:
                prompt_embeds = encode_prompt(prompt)
            prompt_sources[prompt_source] = prompt_sources.get(prompt_source, 0) + 1

            payload = {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "geometry": geometry_tag,
                "sample_id": sample_id,
                "scenario": meta["scenario"],
                "video_latent": video_latent,
                "image_cond_latent": image_cond_latent.half() if base_dir is not None else image_cond_latent_store,
                "motion_condition": gt_condition,
                "motion_condition_zero": zero_condition,
                **conditions,
                "motion_condition_pred": conditions[f"motion_condition_pred__{fields[0]}"],   # alias (shared storage)
                "prompt_embeds": prompt_embeds,
                "prompt": prompt,
                "prompt_source": prompt_source,
                "caption_source": meta["caption_source"],
                "frames": args.frames,
                "span_seconds": args.span_seconds,
                "height": height,
                "width": width,
                "native_size": [native_h, native_w],
                "decode": decode,
                "latent_frame_indices": selected,
                "horizon_seconds": future_seconds,
                "label_source": meta["label_source"],
                "flow_source": meta["flow_source"],
                "instance_source": meta["instance_source"],
                "condition_scope": "all_moving_instances",
                "condition_channels": ["warped_first_frame_latent[16]", "flow_xy[2]", "mask[1]", "visibility[1]"],
                "pred_source": args.pred_source,
                "pred_field": fields[0],
                "pred_fields": fields,
                "pred_contracts": contracts,
                "pred_horizon_seconds": [pred_horizons[i] for i in pred_index],
                "pred_summary": pred_summary,
                "shuffle_summary": shuffle_summary,
                "base_cache": str(base_dir) if base_dir is not None else "",
            }
            temporary = output.with_suffix(f".tmp-{os.getpid()}")
            torch.save(payload, temporary)
            temporary.replace(output)
            completed += 1
            if completed % 200 == 0:
                print(f"  {completed} written, {skipped} skipped, {failed} failed", flush=True)
        except Exception as error:                                    # noqa: BLE001
            failed += 1
            if failed <= 5:
                print(f"[fail] {sample_id}: {error!r}", flush=True)
            errors.append({"sample_id": sample_id, "error": repr(error)})

    count = max(completed, 1)
    summary = {"completed": completed, "skipped": skipped, "failed": failed, "shard": args.shard_index,
               "num_shards": args.num_shards, "geometry": geometry_tag, "fields": fields,
               "field_stats": {field: {"mean_peak_px": value["peak_px"] / count, "mean_mask_fraction": value["mask_fraction"] / count,
                                       "abstain_fraction": value["abstain"] / count, "jump_max": value["jump_max"]}
                               for field, value in stats.items()},
               "shuffle": ({**shuffle_stats,
                            "rotated_fraction_of_support": shuffle_stats["rotated_fraction_of_support"] / max(shuffle_stats["clips"], 1),
                            "direction_cosine": shuffle_stats["direction_cosine"] / max(shuffle_stats["clips"], 1)}
                           if args.shuffle else None),
               "prompt_sources": prompt_sources, "prompts_encoded_now": encode_prompt.encoded, "errors": errors[:20]}
    (output_dir / f"summary-shard{args.shard_index:03d}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}))


if __name__ == "__main__":
    main()
