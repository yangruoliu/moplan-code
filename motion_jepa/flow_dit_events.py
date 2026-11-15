"""Matched-event training: identical observations, audited alternative futures.

Pairs are supplied by a data recipe, never inferred from caption similarity or
from a model prediction. Labels supervise both members independently; this loss
also supervises their difference. Paraphrases must have identical labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .flow_dit_data import caption_key, collate
from .flow_dit_coarse import object_coarse_terms, _mean


class EventPairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, manifest, prompt_table, prompt_embeddings=None):
        self.dataset = dataset
        payload = json.loads(Path(manifest).read_text())
        if payload.get("schema") != "stage1_event_pairs_v1":
            raise ValueError("unsupported event pair schema")
        self.pairs = payload["pairs"]
        self.index = {row.sample_id: i for i, row in enumerate(dataset.rows)}
        if len(self.index) != len(dataset.rows):
            raise ValueError("event training requires globally unique sample IDs")
        self.prompts = (prompt_embeddings if prompt_embeddings is not None else
                        torch.load(prompt_table, map_location="cpu", weights_only=True, mmap=True)["embeddings"])
        for sid, alternatives in self.pairs.items():
            if sid not in self.index or not alternatives:
                raise ValueError(f"event base missing from training split: {sid}")
            for item in alternatives:
                if item["relation"] not in ("invariant", "counterfactual"):
                    raise ValueError("unknown event relation")
                if item.get("sample_id", sid) not in self.index:
                    raise ValueError("event partner is not in the same training split")
                if caption_key(item["caption"]) not in self.prompts:
                    raise ValueError("event prompt was not encoded")
        self.contract = {"schema": payload["schema"], "base_rows": len(self.index),
                         "paired_rows": len(self.pairs),
                         "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest()}
        digest = hashlib.sha256()
        with Path(prompt_table).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        self.contract["prompt_table_sha256"] = digest.hexdigest()
        self.contract["choice"] = "sample_id_sha256_mod_alternatives_v1"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        base = self.dataset[index]
        alternatives = self.pairs.get(base["sample_id"])
        if not alternatives:
            return base
        # A deterministic choice preserves the trainer's exact data-cursor
        # replay contract, including persistent worker prefetch after resume.
        choice = int(hashlib.sha256(base["sample_id"].encode()).hexdigest()[:8], 16)
        spec = alternatives[choice % len(alternatives)]
        partner_id = spec.get("sample_id", base["sample_id"])
        partner = dict(base) if partner_id == base["sample_id"] else self.dataset[self.index[partner_id]]
        # These are observations, including the causal VAE's first latent. The
        # latter check catches accidental future-label leakage into the prior.
        for key in ("first_frame", "image_cond_latent", "instance_masks", "instance_id", "horizon_seconds"):
            if key not in base or key not in partner or not torch.equal(base[key], partner[key]):
                raise ValueError(f"event pair does not share {key}: {base['sample_id']} / {partner_id}")
        if not torch.equal(base["flow_latent"][:, :1], partner["flow_latent"][:, :1]):
            raise ValueError("event pair has different conditioning priors")
        common = (base["flow_valid"] > .5) & (partner["flow_valid"] > .5)
        difference = torch.where(common, base["flow"] - partner["flow"], 0)
        if spec["relation"] == "invariant" and bool((difference.abs() > 1e-6).any()):
            raise ValueError("paraphrase changes the labeled future")
        if spec["relation"] == "counterfactual" and not bool((difference.abs() > 1e-6).any()):
            raise ValueError("counterfactual has no observable target difference")
        partner["prompt_embeds"] = self.prompts[caption_key(spec["caption"])].float()
        return {**base, "event_pair_sample": partner, "event_pair_relation": spec["relation"]}


def collate_event_pairs(batch, prompt_pad_length):
    flat, pairs, relations = [], [], []
    for item in batch:
        index = len(flat)
        flat.append(item)
        if "event_pair_sample" in item:
            pairs.append([index, len(flat)])
            relations.append(item["event_pair_relation"])
            flat.append(item["event_pair_sample"])
    result = collate(flat, prompt_pad_length)
    result["event_pair_indices"] = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    result["event_pair_relations"] = relations
    return result


class EventParaphraseValidation(torch.utils.data.Dataset):
    """A separate held-out wording domain, using exactly the same visual cases."""
    def __init__(self, dataset, manifest):
        self.dataset = dataset
        payload = json.loads(Path(manifest).read_text())
        if payload.get("schema") != "stage1_event_pairs_v1":
            raise ValueError("unknown event validation schema")
        self.rows = dataset.rows
        self.captions = {}
        for row in self.rows:
            specs = payload["pairs"][row.sample_id]
            spec = next(s for s in specs if s["relation"] == "invariant")
            if spec.get("sample_id", row.sample_id) != row.sample_id:
                raise ValueError("validation paraphrases must retain their original labels")
            self.captions[row.sample_id] = spec["caption"]
        self.conditioning_captions = self.captions
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        item = self.dataset[index]
        item["prompt_embeds"] = self.dataset.prompt_table[caption_key(self.captions[item["sample_id"]])].float()
        return item


def event_difference_loss(prediction, batch, config):
    # Empty differentiable reduction avoids 0 * NaN on unobserved pixels.
    zero = prediction["flow"].float().reshape(-1)[:0].sum()
    if config.get("profile") == "minimal5_direction_v1":
        # The unified field-difference term is already included by the
        # minimal5 dispatch. Do not add historical event auxiliaries again.
        return zero, {}
    weights = {name: float(config.get("event_" + name, 0))
               for name in ("invariance", "displacement", "relative_displacement", "direction", "velocity", "affine", "static_role",
                            "direction_margin", "segment_direction", "field_direction")}
    if any(not torch.isfinite(torch.tensor(v)) or v < 0 for v in weights.values()):
        raise ValueError("event weights must be finite and nonnegative")
    if not any(weights.values()):
        return zero, {}
    indices = batch.get("event_pair_indices")
    if indices is None:
        raise ValueError("event loss requires the event-pair data loader")
    if indices.numel() == 0:
        return zero, {"event_pairs": zero.detach()}
    indices = indices.to(prediction["flow"].device)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("event indices must be [P,2]")
    left, right = indices.unbind(1)
    g = batch["flow"].float()
    valid = (batch["flow_valid"][left] > .5) & (batch["flow_valid"][right] > .5)
    target = torch.where(valid, g[left] - g[right], 0)
    ids = batch["instance_id"][left, 0]
    if not torch.equal(ids, batch["instance_id"][right, 0]):
        raise ValueError("paired object identities disagree")
    # Include EVERY first-frame object, including the tool and stationary
    # distractors. Mover-only masks would leave false positive motion unpenalized.
    unique = [torch.unique(row[row > 0]) for row in ids]
    slots = max((len(row) for row in unique), default=0)
    if slots == 0:
        raise ValueError("event pairs have no labeled first-frame objects")
    masks = g.new_zeros((len(left), slots, *g.shape[-2:]))
    for i, values in enumerate(unique):
        masks[i, :len(values)] = ids[i][None] == values[:, None, None]
    horizons = batch["horizon_seconds"][left]
    if not torch.equal(horizons, batch["horizon_seconds"][right]):
        raise ValueError("paired physical times disagree")
    relations = batch["event_pair_relations"]
    if len(relations) != len(left) or any(x not in ("invariant", "counterfactual") for x in relations):
        raise ValueError("event relations do not match pairs")
    invariant = torch.tensor([x == "invariant" for x in relations], device=g.device)
    if bool((target[invariant].abs() > 1e-6).any()):
        raise ValueError("invariance labels disagree")
    total, logs = zero, {"event_pairs": g.new_tensor(len(left))}
    for name, field, default in (("raw", "flow", .5), ("gated", "flow_task_object_gated", 1.)):
        field_weight = float(config.get("event_" + name + "_weight", default))
        if field_weight == 0:
            continue
        if not 0 < field_weight < float("inf"):
            raise ValueError("invalid event field weight")
        delta = torch.where(valid, prediction[field][left].float() - prediction[field][right].float(), 0)
        with torch.autocast(device_type=g.device.type, enabled=False):
            # GT difference specifies magnitude and sign. Opposite prompts are
            # never blindly repelled when their actual futures are identical.
            cf = object_coarse_terms(delta, target, masks, valid.float(), horizons,
                                     config, (~invariant).float())
            inv = object_coarse_terms(delta, target, masks, valid.float(), horizons,
                                      config, invariant.float())
        value = weights["invariance"] * (inv["displacement"] + inv["affine"] + .25 * inv["velocity"])
        for term in ("displacement", "relative_displacement", "direction", "velocity", "affine",
                     "direction_margin", "segment_direction", "field_direction"):
            value = value + weights[term] * cf[term]
            logs[f"event_{name}_{term}"] = cf[term].detach()
        if weights["static_role"]:
            # A false tool motion shared by BOTH captions cancels out of the
            # difference. Penalize it absolutely, with equal object budgets.
            member = torch.cat((left, right))
            support = torch.cat((masks, masks))[:, :, None] * (batch["flow_valid"][member, None, :, 0] > .5)
            certified = support * (batch["static_certified"][member, None, :, 0] > .5)
            area = support.sum((-2, -1)); static_area = certified.sum((-2, -1))
            static_object = (area >= 4) & (static_area >= .95 * area)
            pixels = prediction[field][member].float() * g.new_tensor([g.shape[-1],g.shape[-2]])[None,None,:,None,None]
            safe_pixels = torch.where(batch["flow_valid"][member] > .5, pixels, 0)
            magnitude = (safe_pixels.square().sum(2) + .25).sqrt() - .5
            object_error = (certified * magnitude[:, None]).sum((-2,-1)) / static_area.clamp_min(1)
            role = _mean(object_error / float(config.get("coarse_motion_scale_px",16.)), static_object, None)
            value = value + weights["static_role"] * role
            logs[f"event_{name}_static_role"] = role.detach()
        logs[f"event_{name}_invariance"] = inv["displacement"].detach()
        total = total + field_weight * value
    return total, logs
