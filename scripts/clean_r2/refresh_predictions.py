#!/usr/bin/env python3
"""Replace only frozen Stage1 prediction conditions in existing Stage2 caches.

Real video latents, image latents, captions and measured supervision are reused
unchanged: cond_pred / cond_zero / cond_null (and the 20-channel cond_matched20_*)
are re-rendered from new Stage-1 exports with plan_condition.render_v2, using the
cache's own fp16 image latent and frame times.  A fresh output directory is
required, and every export must name the Stage-1 checkpoint given here
(--allow-smoke-checkpoint skips the checkpoint checks).  Of the checkpoint itself only trainer_state.json is
checked (consumed_coverage: >= 20,000 updated scenes); its weights, prompt contract and geometry are not verified
here, so pass the released stage1/ folder (tools/weights_manifest.json lists its sha256).  The reward scenes
<cache>/scenes_v5/<sid>.npz of the source cache are linked into the output (--no-scenes for
evaluation caches that have none).

usage: refresh_predictions.py --cache SRC --exports FLOWS --ids IDS --output DST --stage1-checkpoint CK
                              [--purpose evaluation --manifest M --no-scenes] [--selection-receipt R]
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import motion_plan as mp
from scripts.control import plan_condition as pc
from motion_jepa.stage2_condition import NATIVE5_NAMES, render_native5


def verify_export_provenance(export, sample_id, checkpoint, state_sha256, marker_sha256):
    expected = {"checkpoint": checkpoint.resolve().name,                 # as motion_jepa/flow_dit_export.checkpoint_provenance
                "checkpoint_state_sha256": state_sha256,
                "checkpoint_complete_sha256": marker_sha256}
    if export.get('sample_id') != sample_id or export.get('stage1_provenance') != expected:
        raise ValueError(f'{sample_id}: export does not identify the admitted frozen Stage1 checkpoint')


def refresh_payload(payload, export, native_hw=None, *, native5=False, real_plan=None):
    result = dict(payload)
    size = (int(payload['height']), int(payload['width']))
    times, image = payload['frame_seconds'], payload['image_cond_latent'].float()
    plan = mp.from_stage1_export(export, size, native_hw=native_hw)
    plan.sample_id = payload['sample_id']
    for kind in ('pred', 'zero', 'null'):
        result[f'cond_{kind}'] = pc.render_v2(kind, plan, image, times, size, export=export, native_hw=native_hw).half()
        base_plan = mp.zeroed(plan) if kind == 'zero' else (None if kind == 'null' else plan)
        result[f'cond_matched20_{kind}'] = pc.render_base(base_plan, image, times, size).half()
        if native5:
            result[f'cond_native5_{kind}'] = render_native5(plan, image, times, size, kind=kind).half()
    if native5:
        if real_plan is None:
            raise ValueError('native5 replay requires the saved measured plan')
        result['cond_native5_real'] = render_native5(real_plan, image, times, size, kind='real').half()
        result['native5_channels'] = list(NATIVE5_NAMES)
    result['plan_pred_summary'] = json.dumps(mp.describe(plan))
    result['slots'] = [obj.slot for obj in plan.objects]
    return result, plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'exports', 'ids', 'output', 'stage1-checkpoint'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--native-hw', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'))
    p.add_argument('--manifest', type=Path, help='first_frame_path determines per-video native dimensions')
    p.add_argument('--native5', action='store_true')
    p.add_argument('--real-plans', type=Path)
    p.add_argument('--selection-receipt', type=Path, help='budgeted selection from the large eligible training pool')
    p.add_argument('--purpose', choices=('training', 'evaluation'), default='training')
    p.add_argument('--allow-smoke-checkpoint', action='store_true')
    p.add_argument('--no-scenes', action='store_true',
                   help='evaluation caches without reward scenes: do not require or link <cache>/scenes_v5/<sid>.npz')
    a = p.parse_args()
    state_path = a.stage1_checkpoint / 'trainer_state.json'
    state = json.loads(state_path.read_text())
    coverage = state.get('consumed_coverage', {})
    if not a.allow_smoke_checkpoint and coverage.get('unique_scenes_updated', 0) < 20000:
        raise ValueError('formal RL requires a frozen Stage1 checkpoint with >=20,000 actually updated scenes')
    state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
    marker = a.stage1_checkpoint / 'checkpoint_complete.json'
    marker_sha256 = hashlib.sha256(marker.read_bytes()).hexdigest() if marker.is_file() else None
    if a.output.exists():
        raise FileExistsError('use a new output directory; prior prediction caches are immutable')
    ids = list(dict.fromkeys(a.ids.read_text().split()))
    selection = json.loads(a.selection_receipt.read_text()) if a.selection_receipt else None
    if not a.allow_smoke_checkpoint and a.purpose == 'training':
        if not selection or int(selection.get('eligible_independent_videos', 0)) < 20000:
            raise ValueError('formal RL conditions must come from the admitted large training pool')
        if not ids or not set(ids) <= set(selection.get('selected_ids', [])):
            raise ValueError('prediction IDs differ from the declared budgeted training selection')
    if a.manifest:
        records = (list(csv.DictReader(a.manifest.open())) if a.manifest.suffix == '.csv'
                   else [json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()])
        rows = {row['sample_id']: row for row in records}
        if len(rows) != len(records):
            raise ValueError('manifest repeats a sample ID')
    else:
        rows = {}
    if a.purpose == 'evaluation' and (not rows or not set(ids) <= rows.keys()):
        raise ValueError('evaluation predictions require an explicit manifest covering every held-out ID')
    a.output.mkdir(parents=True)
    (a.output / 'plans').mkdir()
    if not a.no_scenes:
        (a.output / 'scenes_v5').mkdir()
    evidence = []
    for sid in ids:
        original = a.cache / f'{sid}.pt'
        export_path = a.exports / f'{sid}.pt'
        scene = a.cache / 'scenes_v5' / f'{sid}.npz'
        if not a.no_scenes and not scene.is_file():
            raise FileNotFoundError(scene)
        payload = torch.load(original, map_location='cpu', weights_only=False)
        export = torch.load(export_path, map_location='cpu', weights_only=False)
        if not a.allow_smoke_checkpoint:
            verify_export_provenance(export, sid, a.stage1_checkpoint, state_sha256, marker_sha256)
        native_hw = a.native_hw
        if sid in rows:
            from PIL import Image
            with Image.open(rows[sid]['first_frame_path']) as first:
                native_hw = (first.height, first.width)
        real = mp.load_plan(a.real_plans / f'{sid}.npz') if a.native5 and a.real_plans else None
        updated, plan = refresh_payload(payload, export, native_hw, native5=a.native5, real_plan=real)
        updated['stage1_checkpoint'] = a.stage1_checkpoint.resolve().name     # names only; the sha256 are in the receipt
        updated['stage1_export'] = export_path.name
        if sid in rows:
            updated['first_frame_path'] = rows[sid]['first_frame_path']
        torch.save(updated, a.output / f'{sid}.pt')
        mp.save_plan(a.output / 'plans' / f'{sid}.npz', plan)
        if not a.no_scenes:
            (a.output / 'scenes_v5' / f'{sid}.npz').symlink_to(scene.resolve())
        evidence.append({'sample_id': sid, 'source_cache': original.name,
                         'export_sha256': hashlib.sha256(export_path.read_bytes()).hexdigest()})
    (a.output / 'ids_train.txt').write_text(''.join(sid + '\n' for sid in ids))
    (a.output / 'PREDICTION_RECEIPT.json').write_text(json.dumps({
        'stage1_checkpoint': a.stage1_checkpoint.resolve().name,
        'stage1_state_sha256': state_sha256, 'stage1_checkpoint_complete_sha256': marker_sha256,
        'coverage': coverage, 'smoke_only': a.allow_smoke_checkpoint,
        'purpose': a.purpose, 'evaluation_only': a.purpose == 'evaluation',
        'manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest() if a.manifest else None,
        'selection': selection,
        'selection_receipt_sha256': hashlib.sha256(a.selection_receipt.read_bytes()).hexdigest() if a.selection_receipt else None,
        'independent_conditions': len(ids), 'generated_videos': 0, 'samples': evidence}, indent=2) + '\n')
    print(f'READY {len(ids)} frozen prediction conditions')


if __name__ == '__main__':
    main()
