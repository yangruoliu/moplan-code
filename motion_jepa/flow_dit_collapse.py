"""Evaluation-only motion collapse diagnostics, independent of training loss.

Never use a predicted gate to restrict the reference support. Call on both
raw and selected fields, so closing the gate is counted as failing to move.
The bounded moving utility is a candidate-scoring probe, NOT an RL reward:
static instructions, identity and text consistency need separate admission.
"""
from __future__ import annotations
import math
import torch
from .flow_dit_minimal import _inputs, _segments


@torch.no_grad()
def segment_collapse_statistics(predicted, batch, config):
    with torch.autocast(device_type=predicted.device.type, enabled=False):
        p,g,masks,known,_,_ = _inputs({'flow':predicted},batch,None)
        ps,gs,moving = _segments(p,g,masks,known,config)
        pn,gn = ps.norm(dim=-1),gs.norm(dim=-1)
        cosine = (ps*gs).sum(-1)/(pn*gn).clamp_min(1e-8)
        ratio = pn/gn.clamp_min(1e-8)
        object_times = moving.any(2)
        objects = object_times.any(-1)
        eligible = objects.any(-1)
        def case_sum(value):
            cells = (value*moving).sum(2)/moving.sum(2).clamp_min(1)
            per_object = (cells*object_times).sum(-1)/object_times.sum(-1).clamp_min(1)
            per_case = (per_object*objects).sum(-1)/objects.sum(-1).clamp_min(1)
            return float((per_case*eligible).sum())
        # Bounded magnitude agreement is zero at no motion and decreases for
        # over-speed too. Static pixels cannot add positive moving reward.
        amplitude_agreement = torch.minimum(ratio,1/ratio.clamp_min(1e-8)).clamp(0,1)
        utility = 2*cosine.clamp(0,1)*amplitude_agreement - 1
        enough = pn>=float(config.get('direction_min_progress_px',.5))
        values = dict(nearzero_fraction=~enough, underspeed_fraction=ratio<.25,
                      overspeed_fraction=ratio>2, reverse_fraction=(cosine<0)&(pn>1e-8),
                      magnitude_ratio=ratio,
                      joint_success=(cosine>=math.cos(math.pi/6))&enough&(ratio>=.5)&(ratio<=2),
                      moving_utility=utility)
        result = dict(eligible_cases=float(eligible.sum()))
        result.update({k+'_case_sum':case_sum(v) for k,v in values.items()})
        return result


def summarize_segment_collapse(rows):
    if not rows: return {'eligible_cases':0.}
    total = {k:sum(row[k] for row in rows) for k in rows[0]}
    count = total['eligible_cases']
    return dict(eligible_cases=count,**{k.removesuffix('_case_sum'):v/count if count else None
                                      for k,v in total.items() if k.endswith('_case_sum')})


def collapse_guard_failures(initial,current,*,max_fraction_increase=.05,min_retained_magnitude=.8):
    """Extra stops on full-validation cumulative pixel statistics already logged.

    Thresholds compare each domain/readout to its own step0; they do not mean
    the initial magnitude was adequate. Segment case-macro metrics above are
    reported separately, never confused with these pixel-micro statistics.
    """
    result=[]
    for prefix in ('','event_paraphrase_','phyco_anchor_'):
        for field in ('raw','task_object_gated'):
            for suffix in ('moving_nearzero_fraction','moving_reverse_fraction_all',
                           'moving_pred_to_gt_magnitude'):
                key=f'metric/{prefix}{field}_motion/{suffix}'
                a,b=initial.get(key),current.get(key)
                if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not math.isfinite(a+b):
                    result.append(dict(metric=key,reason='missing_or_nonfinite_collapse_metric'))
                elif suffix=='moving_pred_to_gt_magnitude':
                    if b < a*min_retained_magnitude:
                        result.append(dict(metric=key,initial=a,current=b,reason='motion_magnitude_collapse'))
                elif b > a+max_fraction_increase:
                    result.append(dict(metric=key,initial=a,current=b,reason='motion_zero_or_reverse_regression'))
    return result


def raw_motion_guard(initial, current, *, strict_final=False):
    """Motion-only admission: selector recall/false activity cannot stop it.

    Inputs are unprefixed validation summaries, or metric/-prefixed history
    rows. Direction is segment/case macro; speed and nearzero use the existing
    cumulative moving-pixel definition, unchanged from the reference run.
    Missing/nonfinite values fail closed. Static-only domains have no raw
    motion denominator and are intentionally excluded.
    """
    failures, selectors = [], {}
    def value(row, key):
        return row.get(key, row.get('metric/'+key))
    for prefix in ('', 'event_paraphrase_', 'phyco_anchor_'):
        checks = (
            (f'{prefix}minimal5_raw_direction/direction30_accuracy', 'min_delta', 0. if strict_final else -.03),
            (f'{prefix}raw_motion/moving_pred_to_gt_magnitude', 'min_ratio', .9),
            (f'{prefix}raw_motion/moving_nearzero_fraction', 'max_delta', 0. if strict_final else .03),
        )
        for key, kind, bound in checks:
            a, b = value(initial, key), value(current, key)
            valid = all(isinstance(v, (float,int)) and math.isfinite(v) for v in (a,b))
            if not valid:
                failures.append(dict(metric=key, reason='missing_or_nonfinite'))
                continue
            failed = ((b < a+bound-1e-12) if kind == 'min_delta' else
                      (b < a*bound-1e-12) if kind == 'min_ratio' else (b > a+bound+1e-12))
            if failed:
                failures.append(dict(metric=key, initial=a, current=b, rule=kind, bound=bound))
        selectors[prefix or 'primary'] = {
            suffix: dict(initial=value(initial, f'{prefix}task_object_gated_motion/{suffix}'),
                         current=value(current, f'{prefix}task_object_gated_motion/{suffix}'))
            for suffix in ('moving_recall', 'static_falseactive_fraction')}
    return dict(passed=not failures, failures=failures,
                selector_policy='record_only_never_stops_motion', selector=selectors)


def full_training_admission(initial, current):
    """Stricter gate at training step 200, including paired text discrimination."""
    result = raw_motion_guard(initial, current, strict_final=True)
    failures = list(result['failures'])
    step = current.get('step')
    if step != 200:
        failures.append(dict(reason='requires_completed_step200', step=step))
    # Ratio alone can improve by driving hold toward zero while release also
    # collapses. Require the paired GT-normalized absolute delta as well.
    for suffix in ('release_hold_ratio', 'gt_normalized_delta'):
        key = 'release_hold/raw/' + suffix
        a, b = [row.get(key, row.get('metric/'+key)) for row in (initial, current)]
        if not all(isinstance(v, (int,float)) and math.isfinite(v) for v in (a,b)):
            failures.append(dict(metric=key, reason='missing_or_nonfinite_discrimination'))
        elif b < a-1e-12:
            failures.append(dict(metric=key, initial=a, current=b, reason='release_hold_discrimination_declined'))
    return {**result, 'passed': not failures, 'failures': failures}
