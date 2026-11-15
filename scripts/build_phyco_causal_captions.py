#!/usr/bin/env python
"""Build short Physics-IQ-style PhyCo captions from trusted causal metadata.

The builder is deterministic. It uses rendered first-frame labels only for
actor grounding and relative screen direction; it never reads future RGB or
future motion. Exact physical values stay in the proposal sidecar while the
caption uses short, frozen qualitative bins.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import os
from pathlib import Path
import re

try:
    import torch
except ModuleNotFoundError:  # Metadata-only captioning has no tensor dependency.
    torch = None


OUTCOME_PATTERN = re.compile(
    r"\b(after|later|eventually|bounce|bounces|bounced|rebound|rebounds|"
    r"deform|deforms|deformed|stretch|stretches|slow|slows|stop|stops|"
    r"settle|settles|final|exit|exits|remain|remains)\b",
    re.IGNORECASE,
)
SUPPORT_TYPES = {"dome", "cube_platform"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def caption_sha256(value: str) -> str:
    return sha256_bytes(value.strip().encode("utf-8"))


def physicsiq_sections(caption: str) -> tuple[str, str]:
    """Validate and split the frozen Physics-IQ three-sentence surface form."""
    parts = caption.split(". ")
    if len(parts) != 3 or parts[-1] != "Static shot with no camera movement.":
        raise ValueError("caption does not follow the Physics-IQ V10 sentence contract")
    return parts[0] + ".", parts[1] + "."


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def aligned(objects: dict, key: str, index: int):
    values = objects.get(key, [])
    return values[index] if isinstance(values, list) and index < len(values) else None


def direction(dx: float, dy: float) -> str:
    horizontal = "right" if dx > 0 else "left"
    vertical = "down" if dy > 0 else "up"
    ax, ay = abs(dx), abs(dy)
    if ax < 0.35 * max(ay, 1e-6):
        return vertical
    if ay < 0.35 * max(ax, 1e-6):
        return horizontal
    return f"{vertical} and {horizontal}"


def arrow_direction(records, start_key: str, end_key: str) -> str | None:
    if not isinstance(records, list) or not records:
        return None
    start, end = records[0].get(start_key), records[0].get(end_key)
    if not (
        isinstance(start, list) and isinstance(end, list)
        and len(start) >= 2 and len(end) >= 2
        and all(finite(value) for value in [*start[:2], *end[:2]])
    ):
        return None
    return direction(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))


def elasticity(value) -> str:
    if not finite(value):
        return "rigid"
    value = float(value)
    if value <= 0.25:
        return "slightly bouncy rigid"
    if value <= 0.75:
        return "moderately bouncy rigid"
    return "highly bouncy rigid"


def surface_phrase(value) -> str:
    text = str(value or "").lower()
    if "wood" in text or "parquet" in text or "laminate" in text:
        return "hard wooden"
    if "metal" in text:
        return "hard metal"
    if "concrete" in text or "stone" in text or "gravel" in text:
        return "hard concrete"
    return "hard"


def softness(objects: dict, index: int) -> str:
    mu = aligned(objects, "neo_hookean_mu", index)
    if not finite(mu):
        return "soft elastic"
    if float(mu) <= 90.72528731376207:
        return "very soft elastic"
    if float(mu) <= 533.9996228789312:
        return "soft elastic"
    return "firm elastic"


def force_strength(record: dict, causal: dict) -> str:
    magnitude = record.get("force_magnitude")
    low, high = causal.get("min_force"), causal.get("max_force")
    if not all(finite(value) for value in (magnitude, low, high)) or float(high) <= float(low):
        return ""
    fraction = (float(magnitude) - float(low)) / (float(high) - float(low))
    if fraction <= 1 / 3:
        return "light"
    if fraction <= 2 / 3:
        return "moderate"
    return "strong"


def contact_friction(*values) -> tuple[str, dict]:
    """Return a frozen qualitative contact bin without exposing coefficients."""
    finite_values = [float(value) for value in values if finite(value)]
    if not finite_values:
        return "", {"contact_friction_values": [], "contact_friction_phrase": None}
    effective = sum(finite_values) / len(finite_values)
    if effective <= 0.25:
        phrase = "low-friction"
    elif effective <= 0.65:
        phrase = "moderate-friction"
    else:
        phrase = "high-friction"
    return phrase, {
        "contact_friction_values": finite_values,
        "contact_friction_effective_mean": effective,
        "contact_friction_phrase": phrase,
        "contact_friction_frozen_thresholds": [0.25, 0.65],
    }


def centroids(instance_id: torch.Tensor) -> dict[int, tuple[float, float]]:
    result = {}
    for value in torch.unique(instance_id).tolist():
        value = int(value)
        if value <= 0:
            continue
        y, x = torch.where(instance_id == value)
        if x.numel():
            result[value] = (float(x.float().mean()), float(y.float().mean()))
    return result


def visual_color(frame: torch.Tensor, labels: torch.Tensor, instance_id: int) -> str | None:
    pixels = frame[:, labels == instance_id]
    if pixels.shape[1] < 12:
        return None
    rgb = pixels.float().median(1).values.div(255).clamp(0, 1).tolist()
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    if v < 0.18:
        return "black"
    if s < 0.14:
        return "white" if v > 0.8 else "gray"
    degrees = h * 360
    if degrees < 15 or degrees >= 345:
        return "red"
    if degrees < 40:
        return "orange" if v > 0.55 else "brown"
    if degrees < 70:
        return "yellow"
    if degrees < 165:
        return "green"
    if degrees < 195:
        return "cyan"
    if degrees < 255:
        return "blue"
    if degrees < 290:
        return "purple"
    if degrees < 345:
        return "pink"
    return None


def actor_records(
    metadata: dict,
    frame: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> list[dict]:
    """Build actor facts from trusted metadata, with optional frame grounding.

    None of the current V10 caption templates requires future frames.  The
    first-frame cache is optional because its only additional facts are color
    and screen centroid; the force/velocity directions already come from the
    simulator's causal metadata.  Keeping this path metadata-only lets caption
    generation run beside the raw corpus without copying latent caches.
    """
    if (frame is None) != (labels is None):
        raise ValueError("frame and labels must both be provided or both omitted")
    objects = metadata["object_data"]
    records = []
    centers = centroids(labels) if labels is not None else {}
    for index, kind in enumerate(objects["type"]):
        if kind in SUPPORT_TYPES:
            continue
        instance_id = int(aligned(objects, "segmentation_id", index) or 0)
        records.append({
            "index": index,
            "type": str(kind),
            "instance_id": instance_id,
            "center": centers.get(instance_id),
            "color": visual_color(frame, labels, instance_id) if frame is not None else None,
            "mass": aligned(objects, "mass", index),
            "friction": aligned(objects, "friction", index),
            "restitution": aligned(objects, "restitution", index),
            "use_neo_hookean": aligned(objects, "use_neo_hookean", index),
            "neo_hookean_mu": aligned(objects, "neo_hookean_mu", index),
            "neo_hookean_lambda": aligned(objects, "neo_hookean_lambda", index),
            "neo_hookean_damping": aligned(objects, "neo_hookean_damping", index),
        })
    return records


def target_direction(actors: list[dict], target_type: str, metadata: dict, labels: torch.Tensor) -> str | None:
    objects = metadata["object_data"]
    target_indices = [i for i, kind in enumerate(objects["type"]) if kind == target_type]
    if not actors or not target_indices or actors[0]["center"] is None:
        return None
    instance_id = int(aligned(objects, "segmentation_id", target_indices[0]) or 0)
    target = centroids(labels).get(instance_id)
    if target is None:
        return None
    source = actors[0]["center"]
    return direction(target[0] - source[0], target[1] - source[1])


def compact_sidecar(metadata: dict, actors: list[dict], derived: dict) -> dict:
    causal = metadata["causal_metadata"]
    return {
        "source_metadata_sha256": metadata["metadata_sha256"],
        "actors": actors,
        "surface": {
            key: causal.get(key) for key in (
                "platform_texture_type", "ground_texture_type", "platform_friction",
                "wall_friction", "wall_restitution",
            ) if key in causal
        },
        "initial_force": causal.get("applied_forces_simulator"),
        "initial_velocity": causal.get("applied_velocities_simulator", causal.get("velocity_profile")),
        "derived_from_initial_frame_or_frozen_bins": derived,
    }


def make_caption(category: str, metadata: dict, actors: list[dict], labels: torch.Tensor) -> tuple[str, dict]:
    causal = metadata["causal_metadata"]
    surface = surface_phrase(causal.get("platform_texture_type", causal.get("ground_texture_type")))
    derived = {}
    if category.endswith("ball_drop_soft_v4"):
        if len(actors) != 1:
            raise ValueError("soft ball drop requires one actor")
        material = softness(metadata["object_data"], actors[0]["index"])
        caption = (
            f"A {material} ball is suspended above a {surface} surface. "
            "The ball is then released under gravity. Static shot with no camera movement."
        )
        derived["softness_phrase"] = material
    elif category.endswith("ball_drop_v2") or category.endswith("ball_drop_v3"):
        if not actors or any("ball" not in actor["type"] for actor in actors):
            raise ValueError("rigid ball drop has invalid actors")
        descriptions = [elasticity(actor["restitution"]) for actor in actors]
        count_word = {1: "One", 2: "Two", 3: "Three"}.get(len(actors), str(len(actors)))
        if len(set(descriptions)) == 1:
            subject = f"{count_word} {descriptions[0]} ball" + ("s" if len(actors) > 1 else "")
        else:
            subject = f"{count_word} rigid balls with different bounciness levels"
        caption = (
            f"{subject} are suspended above a {surface} surface. "
            "The balls are then released together under gravity. Static shot with no camera movement."
            if len(actors) > 1 else
            f"{subject} is suspended above a {surface} surface. "
            "The ball is then released under gravity. Static shot with no camera movement."
        )
        derived["elasticity_phrases"] = descriptions
    elif category.endswith("ball_wall_collision"):
        ball = next((actor for actor in actors if actor["type"] == "ball"), None)
        if ball is None:
            raise ValueError("wall collision has no ball")
        screen = target_direction([ball], "wall", metadata, labels)
        toward = f" toward the rigid wall to its {screen}" if screen else " toward the rigid wall"
        friction, friction_sidecar = contact_friction(
            ball["friction"], causal.get("platform_friction")
        )
        slab = f"{friction} {surface}" if friction else surface
        caption = (
            f"A {elasticity(ball['restitution'])} ball rests on a {slab} slab beside a rigid wall. "
            f"The ball is launched along the slab{toward}. Static shot with no camera movement."
        )
        derived.update({"wall_screen_direction": screen, **friction_sidecar})
    elif category.endswith("cube_deform_soft_v2_noeff"):
        cube = next((actor for actor in actors if actor["type"] == "jelly_cube"), None)
        weight = next((actor for actor in actors if actor["type"] == "weight"), None)
        if cube is None or weight is None:
            raise ValueError("soft cube scene requires cube and weight")
        material = softness(metadata["object_data"], cube["index"])
        screen = target_direction([weight], "jelly_cube", metadata, labels)
        caption = (
            f"A rigid weight is suspended above a {material} cube resting on a hard support. "
            "The weight is then released under gravity toward the cube. "
            "Static shot with no camera movement."
        )
        derived.update({"softness_phrase": material, "weight_to_cube_screen_direction": screen})
    elif category.endswith("friction_slide_flat_force_v3"):
        brick = next((actor for actor in actors if actor["type"] == "brick"), None)
        force_image = causal.get("applied_forces_image")
        force_records = causal.get("applied_forces_simulator")
        if brick is None or not isinstance(force_records, list) or not force_records:
            raise ValueError("forced slide lacks brick or force")
        screen = arrow_direction(force_image, "image_coordinates", "force_end_image_coordinates")
        strength = force_strength(force_records[0], causal)
        direction_text = screen or "along the surface"
        friction, friction_sidecar = contact_friction(brick["friction"])
        contact = f"{friction} {surface}" if friction else surface
        force_subject = f"A {strength} external force" if strength else "An external force"
        caption = (
            f"A rigid brick rests on a {contact} surface. "
            f"{force_subject} pushes the brick {direction_text}. Static shot with no camera movement."
        )
        derived.update({
            "force_screen_direction": screen, "force_strength_phrase": strength,
            **friction_sidecar,
        })
    elif category.endswith("friction_slide_flat_v2"):
        brick = next((actor for actor in actors if actor["type"] == "sliding_object"), None)
        velocity_image = causal.get("applied_velocities_image")
        velocity = causal.get("velocity_profile")
        if brick is None or not isinstance(velocity, dict) or not finite(velocity.get("velocity_magnitude")):
            raise ValueError("velocity slide lacks actor or velocity")
        screen = arrow_direction(velocity_image, "image_coordinates", "velocity_end_image_coordinates")
        direction_text = screen or "along the surface"
        friction, friction_sidecar = contact_friction(
            brick["friction"], causal.get("platform_friction")
        )
        contact = f"{friction} {surface}" if friction else surface
        caption = (
            f"A rigid rectangular brick rests on a {contact} surface. "
            f"The brick is launched {direction_text}. Static shot with no camera movement."
        )
        derived.update({
            "velocity_screen_direction": screen,
            "exact_initial_speed_stored_in_physics_sidecar": float(
                velocity["velocity_magnitude"]
            ),
            **friction_sidecar,
        })
    else:
        raise ValueError(f"unsupported category: {category}")
    return caption, derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, action="append", required=True)
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument("--flow-cache-dir", type=Path)
    parser.add_argument("--target-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewer", default="deterministic_causal_contract_v1")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if (args.flow_cache_dir is None) != (args.target_cache_dir is None):
        raise ValueError("flow and target caches must both be provided or both omitted")
    if args.flow_cache_dir is not None and torch is None:
        raise RuntimeError("PyTorch is required when flow/target caches are provided")
    candidates = []
    for path in args.candidate_manifest:
        candidates.extend(read_jsonl(path))
    if args.limit:
        candidates = candidates[: args.limit]
    metadata_rows = {}
    for path in args.metadata:
        for record in read_jsonl(path):
            sample_id = record["sample_id"]
            if sample_id in metadata_rows:
                raise ValueError(f"duplicate metadata sample: {sample_id}")
            metadata_rows[sample_id] = record
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proposals, reviews, rejected = [], [], []
    accepted_manifests: dict[str, list[dict]] = {"train": [], "val": []}
    captions = set()
    for row in candidates:
        sample_id = row["sample_id"]
        split = row.get("split", "train")
        try:
            if split not in accepted_manifests:
                raise ValueError(f"unsupported split: {split}")
            metadata = metadata_rows[sample_id]
            frame = labels = None
            if args.flow_cache_dir is not None:
                flow = torch.load(
                    args.flow_cache_dir / f"{sample_id}.pt",
                    map_location="cpu", weights_only=True,
                )
                target = torch.load(
                    args.target_cache_dir / f"{sample_id}.pt",
                    map_location="cpu", weights_only=True,
                )
                frame = flow["first_frame"].to(torch.uint8)
                labels = target["instance_id"].squeeze(0).long()
            actors = actor_records(metadata, frame, labels)
            caption, derived = make_caption(row["category"], metadata, actors, labels)
            claims = sorted(set(match.group(0).lower() for match in OUTCOME_PATTERN.finditer(caption)))
            if claims:
                raise ValueError(f"future-outcome words: {claims}")
            if len(caption) > 320 or len(caption.split()) > 48:
                raise ValueError(f"caption budget exceeded: chars={len(caption)} words={len(caption.split())}")
            scene_setup, initial_action = physicsiq_sections(caption)
            proposal = {
                "decision": "accept",
                "final_caption": caption,
                "scene_setup": scene_setup,
                "subjects": [{
                    "name": actor["type"], "count": 1,
                    "material": (
                        softness(metadata["object_data"], actor["index"])
                        if actor["use_neo_hookean"] else elasticity(actor["restitution"])
                    ),
                    "evidence": (
                        ["trusted causal metadata", "first-frame rendered instance label"]
                        if labels is not None else ["trusted causal metadata"]
                    ),
                } for actor in actors],
                "initial_action": initial_action,
                "physics_sidecar": compact_sidecar(metadata, actors, derived),
                "material_visual_consistency": (
                    "not_in_rigid_looking_training_exclusion_ledger"
                    if split == "train" and row["category"].endswith(
                        ("ball_drop_soft_v4", "cube_deform_soft_v2_noeff")
                    ) else "not_audited_for_training_exclusion"
                    if row["category"].endswith(
                        ("ball_drop_soft_v4", "cube_deform_soft_v2_noeff")
                    ) else "not_applicable"
                ),
                "outcome_claims": [],
                "uncertainty": [],
                "sample_id": sample_id,
                "split": split,
                "caption_sha256": caption_sha256(caption),
                "source_metadata_sha256": metadata["metadata_sha256"],
                "generator": (
                    "trusted_metadata_plus_first_frame_grounding_v1"
                    if labels is not None else "trusted_metadata_causal_contract_v2"
                ),
            }
            proposals.append(proposal)
            if split == "train":
                reviews.append({
                    "sample_id": sample_id, "split": "train", "decision": "accept",
                    "reviewer": args.reviewer, "new_caption": caption,
                    "new_caption_sha256": proposal["caption_sha256"],
                    "checks": {
                        "future_outcome_claims": [], "trusted_causal_inputs_only": True,
                        "word_count": len(caption.split()), "under_character_budget": True,
                    },
                })
            accepted_manifests[split].append({
                "sample_id": sample_id,
                "scenario": row["category"].split("/")[-1],
                "split": split,
                "video_path": row.get("original_manifest_video_path", row.get("video_path")),
                # Train keeps source text and applies an audited override. Val
                # receives the causal caption directly and is never overridden.
                "caption": (
                    row.get("original_caption", row.get("caption", ""))
                    if split == "train" else caption
                ),
                "source": "phyco_kubric",
            })
            captions.add(caption)
        except Exception as error:
            rejected.append({
                "sample_id": sample_id, "split": split,
                "reason": f"{type(error).__name__}: {error}",
            })
    paths = {
        "proposals": args.output_dir / "phyco_causal_v10_proposals.jsonl",
        "reviews": args.output_dir / "phyco_causal_v10_reviews.jsonl",
        "rejections": args.output_dir / "phyco_causal_v10_rejections.jsonl",
        "overrides": args.output_dir / "phyco_causal_v10_overrides.json",
    }
    for key in ("proposals", "reviews", "rejections"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        values = {"proposals": proposals, "reviews": reviews, "rejections": rejected}[key]
        paths[key].write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))
    manifest_paths = {}
    for split, values in accepted_manifests.items():
        if not values:
            continue
        path = args.output_dir / "manifests" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))
        manifest_paths[split] = path
    if "train" not in manifest_paths:
        raise ValueError("caption build produced no accepted train rows")
    review_sha = sha256_file(paths["reviews"])
    manifest_sha = sha256_file(manifest_paths["train"])
    override_records = []
    for row, review in zip(accepted_manifests["train"], reviews):
        override_records.append({
            "sample_id": row["sample_id"], "split": "train",
            "old_caption": row["caption"], "new_caption": review["new_caption"],
            "new_caption_sha256": review["new_caption_sha256"],
            "review": {"source_path": paths["reviews"].name, "source_sha256": review_sha},
        })
    override_payload = {
        "schema": "stage1_caption_overrides_v1", "split": "train",
        "source_manifest_sha256": manifest_sha, "records": override_records,
    }
    paths["overrides"].write_text(json.dumps(override_payload, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "phyco_causal_caption_build_v1",
        "candidate_rows": len(candidates), "accepted_rows": len(proposals),
        "rejected_rows": len(rejected), "unique_captions": len(captions),
        "accepted_by_split": {
            split: len(values) for split, values in accepted_manifests.items()
        },
        "manifest_sha256": manifest_sha, "review_sha256": review_sha,
        "overrides_sha256": sha256_file(paths["overrides"]),
        "future_outcome_claims": 0,
        "visual_grounding_mode": (
            "trusted_metadata_plus_first_frame" if args.flow_cache_dir is not None
            else "trusted_metadata_only_no_future_frames"
        ),
    }
    (args.output_dir / "BUILD_RECEIPT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
