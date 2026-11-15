"""The instruction-outcome rules of the Inertia measurements (one code path for every analysis).

All on camera-compensated displacement (Reading.compensated) in the 448x256 work frame, over the WHOLE generated clip (the
generators often start late).  The thresholds are set from principle, not fitted to manual labels:
  moved            peak displacement >= max(16 px, 0.5 x object size), size = sqrt(first-frame mask area)
  left / right     the peak displacement within 45 deg of the horizontal side
  up / down        the same on the vertical axis
  relation         the A-B distance changes by >= max(16 px, 0.5 x mean size)
  noun             the named object moved, the other moved less than half as far (thresholds as 'moved')
  nothing / event  by the instruction text: 'stays / does not move / nothing moves / keeps holding it still' -> no motion
                   of the object; 'does not fall / without letting it drop / almost falls ... but does not' -> no fall;
                   the real-event side likewise: falling / dropping -> falls, rolling / sliding / poking -> moved
  role             first mover; a diagnostic only -> not scored
                   role_outcome (for the pre-contact role frames): the named
                   instrument moves first AND reaches the other object; a diagnostic, not scored
"""
import re

import numpy as np

RULE_VERSION = "1"
MOVE_MIN_PX, MOVE_SIZE_FRAC, AXIS_DEG = 16.0, 0.5, 45.0
FALL_MIN_PX = 16.0


def ang(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.nan if na < 1e-6 or nb < 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


def comp_at(r, k, t):
    ok = r.ok[k]
    if ok.sum() < 2 or t > r.times[ok][-1] + 1e-6:
        return None
    c = r.compensated()[k]
    return np.array([np.interp(t, r.times[ok], c[ok, i]) for i in (0, 1)])


def pos_series(r, k):
    return r.pos0[k][None, :] + r.compensated()[k], r.ok[k]


def obj_size(r, k):
    a = r.area[k][r.ok[k]] if r.ok[k].any() else np.zeros(0)
    return float(np.sqrt(a[0])) if len(a) else 0.0


def move_thr(r, k):
    return max(MOVE_MIN_PX, MOVE_SIZE_FRAC * obj_size(r, k))


def peak_vec(r, k):
    """camera-compensated displacement at its largest point over the clip (None if the object is not tracked)"""
    c, ok = r.compensated()[k], r.ok[k]
    if not ok.any():
        return None
    mag = np.where(ok, np.linalg.norm(c, axis=-1), -1)
    return c[int(np.argmax(mag))]


def peak(r, k):
    p = peak_vec(r, k)
    return 0.0 if p is None else float(np.linalg.norm(p))


def moved(r, k):
    return peak(r, k) >= move_thr(r, k)


def lr_outcome(gen, k):
    p = peak_vec(gen, k)
    if p is None or np.linalg.norm(p) < move_thr(gen, k):
        return "none"
    if ang(p, [1, 0]) <= AXIS_DEG:
        return "right"
    if ang(p, [-1, 0]) <= AXIS_DEG:
        return "left"
    return "other"


def ud_outcome(gen, k):
    """up / down / other / none: the peak displacement within 45 deg of the vertical axis (thresholds as 'moved')"""
    p = peak_vec(gen, k)
    if p is None or np.linalg.norm(p) < move_thr(gen, k):
        return "none"
    if ang(p, [0, -1]) <= AXIS_DEG:
        return "up"
    if ang(p, [0, 1]) <= AXIS_DEG:
        return "down"
    return "other"


def follows(gen, k, u):
    """the object moves along direction u (e.g. up = (0, -1)): peak displacement >= threshold within 45 deg of u"""
    p = peak_vec(gen, k)
    return bool(p is not None and np.linalg.norm(p) >= move_thr(gen, k) and ang(p, u) <= AXIS_DEG)


def relation_outcome(gen, a, b):
    pa, oa = pos_series(gen, a); pb, ob = pos_series(gen, b)
    ok = oa & ob
    if ok.sum() < 2:
        return None
    d = np.linalg.norm(pa - pb, axis=-1); d0 = d[np.where(ok)[0][0]]
    ch = (d - d0)[ok]
    v = ch[np.argmax(np.abs(ch))]
    thr = max(MOVE_MIN_PX, MOVE_SIZE_FRAC * 0.5 * (obj_size(gen, a) + obj_size(gen, b)))
    return "closer" if v <= -thr else ("apart" if v >= thr else "none")


def top_mover(gen, roles):
    best = (-1.0, -1)
    for j in range(gen.n_objects):
        if j < len(roles) and roles[j] == "hand":
            continue
        if gen.ok[j].any() and moved(gen, j):
            best = max(best, (peak(gen, j), j))
    return best[1] if best[1] >= 0 else None


def first_mover(gen, cands):
    """role (diagnostic): the candidate that starts moving first (reaches 4 px first among those that move)"""
    best = (np.inf, None)
    for j in cands:
        if j is None:
            continue
        c, ok = gen.compensated()[j], gen.ok[j]
        if not ok.any() or not moved(gen, j):
            continue
        mag = np.where(ok, np.linalg.norm(c, axis=-1), 0)
        t = gen.times[np.where(mag >= 4)[0][0]]
        best = min(best, (float(t), j))
    return best[1]


CONTACT_FRAC, ONSET_PX = 1.0, 4.0


def onset_t(gen, k):
    """first time object k is >= ONSET_PX from its start (camera-compensated), None if it never is"""
    c, ok = gen.compensated()[k], gen.ok[k]
    mag = np.where(ok, np.linalg.norm(c, axis=-1), 0.0)
    i = np.where(mag >= ONSET_PX)[0]
    return float(gen.times[i[0]]) if len(i) else None


def role_outcome(gen, instrument, patient):
    """Role rule v2 (diagnostic until validated on manual labels): 'A hand hits P with I' is done when the
    instrument I moves (>= the 'moved' threshold), I starts moving before P does (or P never moves), and I reaches P
    afterwards -- centroid distance <= CONTACT_FRAC x (r_I + r_P), r = sqrt(area / pi), at a frame where both are tracked.
    Returns 'follows', 'partial' (I moved first but never reached P), 'swapped' (P moved first: the other object was used),
    'none' (I did not move), or None (not measurable)."""
    if instrument is None or patient is None or not (gen.ok[instrument].any() and gen.ok[patient].any()):
        return None
    if not moved(gen, instrument):
        return "swapped" if moved(gen, patient) else "none"
    ti, tp = onset_t(gen, instrument), onset_t(gen, patient)
    if tp is not None and ti is not None and tp < ti:
        return "swapped"
    pi, oi = pos_series(gen, instrument); pp, op = pos_series(gen, patient)
    both = oi & op & (gen.times >= (ti or 0.0))
    if not both.any():
        return None
    ri = np.sqrt(np.maximum(gen.area[instrument], 1.0) / np.pi); rp = np.sqrt(np.maximum(gen.area[patient], 1.0) / np.pi)
    d = np.linalg.norm(pi - pp, axis=-1)
    return "follows" if bool((d[both] <= CONTACT_FRAC * (ri[both] + rp[both])).any()) else "partial"


def single_mover(gen, target, other):
    """the named object moved and the other moved less than half as far -> (ok, spread)"""
    if target is None:
        return False, False
    pt = peak(gen, target); po = peak(gen, other) if other is not None else 0.0
    mt = pt >= move_thr(gen, target)
    return mt and po < 0.5 * pt, mt and po >= 0.5 * pt


def falls(gen, k):
    c, ok = gen.compensated()[k], gen.ok[k]
    if not ok.any():
        return None
    y = c[:, 1]
    drop = max((float(np.max(y[ok & (gen.times > gen.times[i])] - y[i])) if (ok & (gen.times > gen.times[i])).any() else 0.0)
               for i in np.where(ok)[0])
    last = np.where(ok)[0][-1]
    lost_down = bool(last < len(gen.times) - 1 and y[last] >= FALL_MIN_PX)
    return bool(drop >= max(FALL_MIN_PX, MOVE_SIZE_FRAC * obj_size(gen, k)) or lost_down)


_NO_MOTION = re.compile(r"stays where it is|does not move|nothing moves|keeps holding it still|keep(s)? holding it", re.I)
_NO_FALL = re.compile(r"does not fall|without letting it drop|but does not|not fall down", re.I)
_FALL_EVENT = re.compile(r"\bfall|\bdrop", re.I)


def nothing_kind(text):
    """what a 'nothing happens' instruction forbids: 'motion' or 'fall'"""
    t = text or ""
    if _NO_FALL.search(t):
        return "fall"
    return "motion" if _NO_MOTION.search(t) else "fall"


def event_kind(text):
    """what the real-event instruction requires: 'fall' (falling / dropping) or 'motion' (rolling / sliding / poking ...)"""
    return "fall" if _FALL_EVENT.search(text or "") else "motion"


def event_happens(gen, k, text):
    if event_kind(text) == "fall":
        return falls(gen, k)
    return moved(gen, k) if gen.ok[k].any() else None


def nothing_holds(gen, k, text):
    if nothing_kind(text) == "fall":
        f = falls(gen, k)
        return None if f is None else not f
    return (not moved(gen, k)) if gen.ok[k].any() else None
