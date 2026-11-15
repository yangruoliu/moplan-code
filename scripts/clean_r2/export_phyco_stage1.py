#!/usr/bin/env python3
"""Export frozen Stage-1 predictions for the PhyCo conditions of the RAFT round.

The PhyCo RAFT conditions are training clips of the Stage-1 checkpoint's own `phyco_anchor` source, so the export runs
scripts/eval_stage1_flowdit.py on that source (train split, filtered to the ids) with one sample, one sampling step,
guidance 1, no shrinkage and the first sample kept.  The flows land in <output>/flows/<sid>.pt; they are the input of
scripts/clean_r2/refresh_predictions.py, which renders the RAFT condition caches.

The checkpoint's resolved_config.yaml (or --config) must list the `phyco_anchor` source in data.additional_sources
with paths that exist on this machine (point them there with --set key=value, passed through to the evaluator).
The id files are merged into <output>/ids.txt.  One receipt per shard, <output>/admission-<shard>.json, records the
checkpoint, the ids and the command; an output folder that already holds exports of another checkpoint or id list is
refused.

usage: export_phyco_stage1.py --checkpoint CK --ids data/ids/raft/ids_phyco.txt data/ids/raft/ids_phycoB.txt --output DIR
                              [--config CK/resolved_config.yaml] [--set key=value ...] [--shard-index i --num-shards n] [--dry-run]
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from motion_jepa.flow_dit_export import checkpoint_provenance  # noqa: E402

SOURCE = 'phyco_anchor'


def merged_ids(paths) -> list[str]:
    """Ids of all files in order, every id once (a repeated id is an error)."""
    ids = [sid for path in paths for sid in Path(path).read_text().split()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty or repeated condition ids')
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True, help='Stage-1 checkpoint directory (trainer_state.json, backend_params.pt, decoder.pt)')
    p.add_argument('--ids', type=Path, nargs='+', required=True, help='condition id files, one id per line')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=None, help='Stage-1 config (default <checkpoint>/resolved_config.yaml)')
    p.add_argument('--set', action='append', default=[], dest='overrides', help='key=value config override for the evaluator')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not 0 <= a.shard_index < a.num_shards:
        p.error('invalid shard index/count')
    checkpoint = a.checkpoint.resolve()
    config_path = (a.config or checkpoint / 'resolved_config.yaml').resolve()
    config = yaml.safe_load(config_path.read_text())
    ids = merged_ids(a.ids)
    ids_text = ''.join(f'{sid}\n' for sid in ids)
    ids_path = a.output.resolve() / 'ids.txt'
    sources = [s for s in config['data'].get('additional_sources', []) if s['name'] == SOURCE]
    if len(sources) != 1:
        raise ValueError(f'{config_path} lists no {SOURCE} training source')
    # The evaluator reads the complete train manifest of the source (caption contract) and filters to the ids afterwards.
    command = [sys.executable, str(ROOT / 'scripts' / 'eval_stage1_flowdit.py'),
               '--config', str(config_path), '--checkpoint', str(checkpoint),
               '--source', SOURCE, '--split', 'train', '--ids', str(ids_path),
               '--output-dir', str(a.output), '--visualizations', '0',
               '--num-samples', '1', '--num-steps', '1', '--guidance-scale', '1',
               '--shrinkage-px', '0', '--reduce', 'first',
               '--shard-index', str(a.shard_index), '--num-shards', str(a.num_shards)]
    for item in a.overrides:
        command += ['--set', item]
    contract = {'stage1_provenance': checkpoint_provenance(checkpoint),
                'ids_sha256': hashlib.sha256(ids_text.encode()).hexdigest(),
                'selected_independent_conditions': len(ids), 'ids': ids[a.shard_index::a.num_shards],
                'horizon_seconds': sources[0].get('horizon_seconds'), 'command': command}
    print(json.dumps(contract, indent=2), flush=True)
    if a.dry_run:
        return
    a.output.mkdir(parents=True, exist_ok=True)
    existing = list(a.output.glob('admission-*.json'))
    if not existing and any((a.output / 'flows').glob('*.pt')):
        raise ValueError('existing exports have no receipt: use a fresh output directory')
    for path in existing:
        previous = json.loads(path.read_text())
        for key in ('stage1_provenance', 'ids_sha256'):
            if previous.get(key) != contract[key]:
                raise ValueError('output belongs to a different checkpoint or id list')
    marker = a.output / f'admission-{a.shard_index:04d}.json'
    if marker.exists() and json.loads(marker.read_text()) != contract:
        raise ValueError('output shard contract changed')
    if ids_path.is_file() and ids_path.read_text() != ids_text:
        raise ValueError(f'{ids_path} holds another id list')
    ids_path.write_text(ids_text)
    marker.write_text(json.dumps(contract, indent=2) + '\n')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
