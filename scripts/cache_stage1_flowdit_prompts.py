#!/usr/bin/env python
"""Build the shared umT5 prompt table for Stage-1 WanFlowDiT.

Captions repeat across rows (many rows share a scene and its caption), so one
shared table keyed by sha256(caption) is far smaller than per-sample storage.

The empty string is encoded too and recorded as ``negative_key``: it is the
unconditional branch for classifier-free guidance, and it must be the *same*
text that ``train.caption_dropout`` substitutes, or CFG interpolates toward a
prompt the model never saw as a dropout target.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_jepa.flow_dit_config import apply_overrides, load_config
from motion_jepa.flow_dit_data import caption_key, read_captions
from motion_jepa.caption_overrides import load_caption_overrides, validate_override_prompt_embedding
from scripts.stage1_paths import placeholder_values, resolve_placeholders


SPLITS = ("train", "val", "internal_test", "test_core")
NEGATIVE_PROMPT = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache the shared Stage-1 prompt table")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "stage1" / "train.yaml"),
                        help="config; ${...} path placeholders are filled from MOPLAN_DATA / WAN_MODEL_DIR")
    parser.add_argument("--split", action="append", choices=SPLITS)
    parser.add_argument(
        "--from-targets",
        action="store_true",
        help="collect captions from the target cache (PhyCo) rather than a "
             "manifest; PhyCo ships one caption per scenario, so the table is "
             "tiny compared with Physion's 1,865 per-sample captions",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source", default="primary",
                        help="primary, secondary or the name of a data.additional_sources entry: select this "
                             "source's manifest, target cache, prompt table and reviewed overrides")
    parser.add_argument("--base-prompt-table", type=Path,
                        help="Copy original embeddings unchanged and encode only missing reviewed caption keys")
    parser.add_argument("--output", type=Path,
                        help="Explicit new output table; required for incremental caption caching")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def collect_from_targets(target_dir: Path) -> dict[str, str]:
    """Return ``{caption_hash: caption}``, the keying the dataset looks up by.

    Keying by sample_id embeds 40k rows of nine distinct strings and, worse,
    the lookup side hashes the caption text -- every fetch would miss.
    """
    import torch

    unique: dict[str, str] = {}
    empty = 0
    for path in sorted(target_dir.glob("*.pt")):
        if path.name.startswith("summary-"):
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        caption = (payload.get("caption") or "").strip()
        if not caption:
            empty += 1
            continue
        unique[caption_key(caption)] = caption
    if empty:
        raise SystemExit(f"{empty} targets have empty captions; fix the target cache first")
    unique[caption_key(NEGATIVE_PROMPT)] = NEGATIVE_PROMPT
    return unique


def collect_captions(
    manifest_dir: Path, splits: list[str], *, caption_overrides: str | Path | None = None,
) -> dict[str, str]:
    """Return ``{caption_hash: caption}`` over the requested splits."""
    unique: dict[str, str] = {}
    replacements: dict[str, str] = {}
    if caption_overrides and "train" in splits:
        manifest = manifest_dir / "train.jsonl"
        replacements = load_caption_overrides(
            caption_overrides, manifest=manifest, original_captions=read_captions(manifest),
        ).captions
    for split in splits:
        path = manifest_dir / f"{split}.jsonl"
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("caption")
                if split == "train" and replacements:
                    text = replacements.get(record["sample_id"], text)
                caption = (text or "").strip()
                if not caption:
                    raise ValueError(
                        f"{path}:{record.get('sample_id')} has no caption; "
                        "point manifest_dir at a manifest with a caption per row"
                    )
                unique[caption_key(caption)] = caption
    unique[caption_key(NEGATIVE_PROMPT)] = NEGATIVE_PROMPT
    return unique


def load_table(path: Path) -> tuple[dict, str]:
    content = path.read_bytes()
    return torch.load(io.BytesIO(content), map_location="cpu", weights_only=True), hashlib.sha256(content).hexdigest()


def _valid_matrix(value) -> bool:
    return (isinstance(value, torch.Tensor) and value.ndim == 2
            and all(value.shape) and bool(torch.isfinite(value).all()))


def override_reuse_errors(table: dict, captions: dict[str, str], overrides,
                          pad_length: int, *, base: dict | None = None,
                          base_sha256: str | None = None) -> list[str]:
    """Inspect tensor bytes/shapes, not just the advertised pad/key metadata."""
    errors = []
    embeddings = table.get("embeddings")
    if not isinstance(embeddings, dict):
        return ["embeddings is not a mapping"]
    if table.get("negative_key") != caption_key(NEGATIVE_PROMPT):
        errors.append("negative_key does not identify the unconditional caption")
    incremental = table.get("caption_override_incremental", {})
    compatible_pad = (type(table.get("prompt_pad_length")) is int and table["prompt_pad_length"] == pad_length)
    if isinstance(incremental, dict):
        compatible_pad |= (incremental.get("schema") == "stage1_incremental_caption_prompts_v1"
                           and incremental.get("training_prompt_pad_length") == pad_length
                           and incremental.get("override_file_sha256") == overrides.contract["override_file_sha256"])
    if not compatible_pad:
        errors.append("encoding/training pad metadata mismatch")
    widths = set()
    override_keys = {caption_key(text) for text in overrides.captions.values()}
    for key in captions:
        value = embeddings.get(key)
        if not _valid_matrix(value):
            errors.append(f"missing or invalid prompt matrix: {key}")
            continue
        widths.add(value.shape[1])
        if key in override_keys:
            try:
                validate_override_prompt_embedding(value, pad_length=pad_length, label=key)
            except ValueError as exc:
                errors.append(str(exc))
    if len(widths) > 1:
        errors.append("required prompt embedding widths disagree")
    if base is not None:
        if not isinstance(incremental, dict) or incremental.get("base_prompt_table_sha256") != base_sha256:
            errors.append("incremental base provenance mismatch")
        if table.get("negative_key") != base["negative_key"]:
            errors.append("original negative key changed")
        for key, value in base["embeddings"].items():
            actual = embeddings.get(key)
            if not isinstance(actual, torch.Tensor) or actual.dtype != value.dtype or not torch.equal(actual, value):
                errors.append(f"original base embedding changed: {key}")
                break
    return errors


def incremental_caption_plan(base: dict, captions: dict[str, str], overrides,
                             pad_length: int) -> dict[str, str]:
    embeddings = base.get("embeddings")
    if (not isinstance(embeddings, dict) or base.get("negative_key") != caption_key(NEGATIVE_PROMPT)
            or base.get("negative_key") not in embeddings):
        raise ValueError("incremental base lacks the original unconditional prompt contract")
    if (type(base.get("prompt_pad_length")) is not int or base["prompt_pad_length"] < 1
            or any(not _valid_matrix(value) for value in embeddings.values())):
        raise ValueError("incremental base has invalid prompt metadata or matrices")
    if len({value.shape[1] for value in embeddings.values()}) != 1:
        raise ValueError("incremental base prompt widths disagree")
    override_keys = {caption_key(text) for text in overrides.captions.values()}
    missing_original = set(captions) - override_keys - set(embeddings)
    if missing_original:
        raise ValueError("base table lacks original conditioning keys; incremental mode only encodes new override keys")
    for key in override_keys & set(embeddings):
        # Replacing this key would change a base tensor used by another row.
        validate_override_prompt_embedding(embeddings[key], pad_length=pad_length, label=key)
    return {key: captions[key] for key in sorted(override_keys - set(embeddings))}


@torch.inference_mode()
def encode_captions(captions: dict[str, str], model_dir: Path, pad_length: int) -> tuple[dict, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("prompt caching requires a CUDA GPU for umT5")
    from transformers import T5TokenizerFast, UMT5EncoderModel
    device = torch.device("cuda")
    tokenizer = T5TokenizerFast.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_dir / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True,
    ).to(device).eval()
    embeddings, token_counts = {}, {}
    truncated = 0
    for key, caption in sorted(captions.items(), key=lambda item: item[1]):
        tokens = tokenizer([caption], padding="max_length", max_length=pad_length,
                           truncation=True, add_special_tokens=True,
                           return_attention_mask=True, return_tensors="pt")
        attention = tokens.attention_mask.to(device)
        length = int(attention.sum())
        if length >= pad_length:
            truncated += 1
        hidden = text_encoder(input_ids=tokens.input_ids.to(device), attention_mask=attention).last_hidden_state[0, :max(length, 1)]
        embeddings[key] = hidden.cpu().to(torch.bfloat16)
        token_counts[key] = length
    if truncated:
        # Preserve the existing conservative >= pad rejection before any write.
        raise SystemExit(f"{truncated} captions hit the {pad_length}-token pad length; raise data.prompt_pad_length")
    tokenizer_files = {name: hashlib.sha256((model_dir / "tokenizer" / name).read_bytes()).hexdigest()
                       for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model")
                       if (model_dir / "tokenizer" / name).is_file()}
    encoder_config = model_dir / "text_encoder/config.json"
    provenance = {"tokenizer_path": f"{Path(model_dir).resolve().name}/tokenizer",      # Wan folder name, no machine path
                  "tokenizer_files_sha256": tokenizer_files,
                  "text_encoder_path": f"{Path(model_dir).resolve().name}/text_encoder",
                  "text_encoder_config_sha256": hashlib.sha256(encoder_config.read_bytes()).hexdigest(),
                  "max_length": pad_length, "padding": "max_length", "truncation": True,
                  "add_special_tokens": True, "dtype": "bfloat16",
                  "acceptance_rule": "attention token count including EOS strictly less than pad_length",
                  "token_counts": token_counts}
    return embeddings, provenance


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    config = resolve_placeholders(config, placeholder_values(), strict=False)
    paths = dict(config["paths"])
    if args.source == "secondary":
        source = config["data"].get("secondary")
        if not source:
            raise ValueError("--source secondary requires data.secondary")
        paths.update({key: source[key] for key in ("manifest_dir", "target_cache_dir", "prompt_table")})
        paths["caption_overrides"] = source.get("caption_overrides")
    elif args.source != "primary":
        source = next((item for item in config["data"].get("additional_sources", [])
                       if item.get("name") == args.source), None)
        if source is None:
            raise ValueError(f"--source {args.source!r} names no data.additional_sources entry")
        paths.update({key: source[key] for key in ("manifest_dir", "target_cache_dir", "prompt_table")})
        paths["caption_overrides"] = source.get("caption_overrides")
    splits = args.split or list(SPLITS)
    if args.from_targets and paths.get("caption_overrides"):
        raise ValueError("reviewed caption overrides require manifest collection; omit --from-targets")
    captions = (collect_from_targets(Path(paths["target_cache_dir"]))
                if args.from_targets
                else collect_captions(Path(paths["manifest_dir"]), splits,
                                      caption_overrides=paths.get("caption_overrides")))

    configured_output = Path(paths["prompt_table"])
    output = args.output or configured_output
    pad_length = int(config["data"]["prompt_pad_length"])
    overrides = None
    if paths.get("caption_overrides") and "train" in splits:
        manifest = Path(paths["manifest_dir"]) / "train.jsonl"
        overrides = load_caption_overrides(paths["caption_overrides"], manifest=manifest,
                                           original_captions=read_captions(manifest))
    base, base_sha, encoding_captions = None, None, captions
    if args.base_prompt_table:
        if overrides is None or args.output is None:
            raise ValueError("incremental caption caching requires reviewed train overrides and explicit --output")
        if (output.resolve() == args.base_prompt_table.resolve()
                or (output.exists() and output.samefile(args.base_prompt_table))):
            raise ValueError("incremental --output must be a new path, never the original base table")
        base, base_sha = load_table(args.base_prompt_table)
        encoding_captions = incremental_caption_plan(base, captions, overrides, pad_length)
    if output.is_file() and not args.overwrite:
        existing = torch.load(output, map_location="cpu", weights_only=True)
        if overrides is not None:
            errors = override_reuse_errors(existing, captions, overrides, pad_length,
                                           base=base, base_sha256=base_sha)
        else:
            present = set(existing.get("embeddings", {}))
            errors = [] if int(existing.get("prompt_pad_length", -1)) == pad_length and set(captions) <= present else ["pad_length changed or captions missing"]
        if not errors:
            print(json.dumps({"status": "exists", "path": str(output)}))
            return
        print(json.dumps({"status": "rebuild", "path": str(output), "reasons": errors}))
    if overrides is not None and base is None:
        if args.output is None or (configured_output.exists() and
                (output.resolve() == configured_output.resolve()
                 or (output.exists() and output.samefile(configured_output)))):
            raise ValueError("writing reviewed captions requires explicit new --output; use --base-prompt-table to preserve original embeddings")
    embeddings = dict(base["embeddings"]) if base is not None else {}
    encoding_provenance = None
    if encoding_captions:
        encoded, encoding_provenance = encode_captions(encoding_captions, Path(paths["wan_model"]), pad_length)
        if set(encoded) != set(encoding_captions):
            raise ValueError("encoder did not return exactly the requested new caption keys")
        for key, value in encoded.items():
            validate_override_prompt_embedding(value, pad_length=pad_length, label=key)
            if base is not None and value.shape[1] != next(iter(base["embeddings"].values())).shape[1]:
                raise ValueError("new encoding width differs from original base prompt table")
        embeddings.update(encoded)
    payload = {"embeddings": embeddings, "negative_key": caption_key(NEGATIVE_PROMPT),
               "prompt_pad_length": pad_length, "splits": splits, "num_captions": len(embeddings)}
    if base is not None:
        # A mixed table must not relabel the old 160-token encodings as 64-token.
        payload = {**base, **payload, "prompt_pad_length": base["prompt_pad_length"],
                   "negative_key": base["negative_key"]}
        payload["caption_override_incremental"] = {
            **overrides.contract,
            "schema": "stage1_incremental_caption_prompts_v1",
            "base_prompt_table": args.base_prompt_table.name,          # file name plus sha256, no machine path
            "base_prompt_table_sha256": base_sha,
            "base_prompt_pad_length": base["prompt_pad_length"],
            "training_prompt_pad_length": pad_length,
            "added_caption_keys": sorted(encoding_captions),
            "reused_override_caption_keys": sorted({caption_key(t) for t in overrides.captions.values()} & set(base["embeddings"])),
            "new_encoding": encoding_provenance,
            "original_embeddings_preserved": True,
        }
        errors = override_reuse_errors(payload, captions, overrides, pad_length,
                                       base=base, base_sha256=base_sha)
        if errors:
            raise ValueError("invalid incremental prompt table: " + "; ".join(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"path": str(output), "num_captions": len(embeddings),
                      "newly_encoded_captions": len(encoding_captions),
                      "truncated_at_pad_length": 0, "pad_length": pad_length,
                      "base_prompt_table_sha256": base_sha}))


if __name__ == "__main__":
    main()
