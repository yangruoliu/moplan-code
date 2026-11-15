"""Recover semantic task IDs from FOT event recipes, including held objects.

No prediction or flow threshold is used. Include the instruction's actor and
every explicitly simulated participant. Static hold still selects its actor;
generic all-static scenes select their full described set. Input SAM masks
and analytic flow targets remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def variant_ids(scene, variant):
    identifiers = {o['id'] for o in scene['objects']}
    actor = variant.get('mover')
    if isinstance(actor, bool) or not isinstance(actor, int) or actor not in identifiers:
        raise ValueError('event recipe has no valid visible actor')
    if variant.get('contrast') == 'absent':
        raise ValueError('absent actor cannot be mapped to a visible object')
    # Generic static recipes describe the whole scene; a hold event has an
    # explicit stationary named actor. Do not invent a noun label for generic
    # "nothing moves" based on its randomly assigned recipe mover field.
    if variant['kind'] == 'static' and variant.get('event') != 'hold':
        return sorted(identifiers)
    selected = {actor} | {m['id'] for m in variant['motions']}
    if not selected <= identifiers:
        raise ValueError('event participant missing from scene')
    return sorted(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    manifests = {p.stem:[json.loads(s) for s in p.read_text().splitlines()]
                 for p in (args.data_root/'manifests').glob('*.jsonl')}
    rows = {r['sample_id']:r for rr in manifests.values() for r in rr}
    mapping = {}
    for path in sorted((args.data_root/'scenes').glob('*.json')):
        scene = json.loads(path.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for variant in scene['variants']:
            sid = variant['sample_id']
            if sid not in rows: continue
            if sid in mapping: raise ValueError('duplicate event sample '+sid)
            if rows[sid]['caption'] != variant['caption']:
                raise ValueError('event caption mismatch '+sid)
            mapping[sid] = dict(sample_id=sid, scenario=rows[sid]['scenario'],
                mover_instance_ids=variant_ids(scene,variant), source_scene_sha256=digest,
                event=variant.get('event',variant['kind']))
    if not rows or set(mapping) != set(rows):
        raise ValueError('incomplete event mapping')
    result = dict(schema='stage1_supervision_mover_ids_v1', rows=len(mapping),
        mapping=mapping, policy='recipe actor plus simulated participants; stationary hold actor is positive; generic static scene includes all named objects',
        manifest_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.data_root/'manifests').glob('*.jsonl')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=len(mapping),split_rows={k:len(v) for k,v in manifests.items()},output=str(args.output))))


if __name__ == '__main__': main()
