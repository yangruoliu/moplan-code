"""Load built benchmark items (items/<sid>.json, masks/<sid>.npz, ref/<sid>.npz) into the containers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .containers import Item, Reference


def load_item_json(path) -> Dict:
    return json.load(open(path))


CONTACT_EVENTS = {"hit_with", "collide", "collide_deflect", "collide_halt", "deflect", "push_onto", "throw_against", "knock_over"}


def trusted_flags(d: Dict) -> list:
    """An instance is trusted when it is an annotated object (matched to a box / point, or given a non-static role)."""
    matched = set(int(v) for v in (d.get("matched") or {}).values())
    return [bool(r in ("actor", "patient", "hand") or k in matched) for k, r in enumerate(d["roles"])]


def item_from_json(d: Dict) -> Item:
    return Item(sid=d["sid"], source=d["source"], group_id=d.get("group_id", d["sid"]), roles=list(d["roles"]),
                instruction=d.get("instruction", ""), event_tags=[d.get("event", "")] + list(d.get("event_tags", [])),
                rigid_support=bool(d.get("rigid_support", True)), static_caption=bool(d.get("static_caption", False)),
                hidden_mover=bool(d.get("hidden_mover", False)), counterfactual_sibling=d.get("counterfactual_sibling"),
                instr_dir=d.get("instr_dir"),
                actor_patient=d.get("actor_patient") if d.get("event") in CONTACT_EVENTS else None,
                trusted=trusted_flags(d),
                meta={k: d.get(k) for k in ("template_id", "track", "direction_expected", "flags", "t0_frame", "fps", "n_frames", "geometry")}).validate()


def load_masks(path) -> np.ndarray:
    z = np.load(path)
    shape = tuple(int(v) for v in z["shape"])
    bits = np.unpackbits(z["masks"], axis=1)[:, : shape[1] * shape[2]]
    return bits.reshape(shape).astype(bool)


def load_built(items_dir, sid: str) -> Tuple[Dict, Item, np.ndarray, Optional[Reference]]:
    items_dir = Path(items_dir)
    d = load_item_json(items_dir / "items" / f"{sid}.json")
    item = item_from_json(d)
    masks = load_masks(items_dir / "masks" / f"{sid}.npz")
    ref_path = items_dir / "ref" / f"{sid}.npz"
    ref = Reference.from_npz(ref_path) if ref_path.is_file() else None
    return d, item, masks, ref


def list_sids(items_dir) -> list:
    return sorted(p.stem for p in (Path(items_dir) / "items").glob("*.json"))
