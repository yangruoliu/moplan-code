"""The one camera rule of the Entanglement (camera) measurements.

Chosen over a translation-only rule by agreement with manual labels of generated clips; the verdicts are stable when
FIT_LOST moves in 0.3-0.7 and WITNESS_PX in 6-12.

  fit verdict   per-frame similarity fit of the background grid (camera_deep.fit_camera): the camera moved if over the fitted
                frames the translation at the image centre reaches CAM_TRANS_PX, or |log scale| reaches CAM_ZOOM, or the
                rotation reaches CAM_ROT_DEG
  witness       the first-frame objects that are static in the real continuation (>= 2, hands excluded): the median of their
                raw (not camera-compensated) peak displacements -- when the whole scene slides, the camera moved
  verdict       fit says moved                                   -> moved
                fit unreliable on >= FIT_LOST of the frames:     witness >= WITNESS_PX -> moved, < -> still, no witness -> unscorable
                otherwise                                        -> still
A fit that fails is not evidence of a still camera: failed fits are often real camera motion (e.g. clips with almost
every frame unfit and a small partial-fit translation, but a clear witness shift).
Unscorable clips are reported per model, never counted as still.
"""
import math

CAM_TRANS_PX, CAM_ZOOM, CAM_ROT_DEG = 8.0, 0.10, 3.0
FIT_LOST, WITNESS_PX = 0.5, 8.0
# direction analyses (does the camera pan WITH the object?) use pans >= CAM_DIR_MIN_PX: below it the pan direction is not
# reliable (small pans follow the object at chance level, and small hand-held shakes of real continuations correlate with
# the object through the estimator), while real continuations almost never pan by CAM_DIR_MIN_PX or more.
CAM_DIR_MIN_PX = 16.0


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def fit_moved(trans, zoom, rot):
    t, z, r = _num(trans), _num(zoom), _num(rot)
    return bool((t is not None and t >= CAM_TRANS_PX) or (z is not None and z >= CAM_ZOOM) or (r is not None and r >= CAM_ROT_DEG))


def verdict(trans, zoom, rot, lost, witness):
    """'moved' / 'still' / 'unscorable' (see module docstring); lost = share of frames whose fit is unreliable"""
    if fit_moved(trans, zoom, rot):
        return "moved"
    lo = _num(lost)
    if lo is not None and lo >= FIT_LOST:
        w = _num(witness)
        if w is None:
            return "unscorable"
        return "moved" if w >= WITNESS_PX else "still"
    return "still"


def witness_from(disp, ok, static_idx):
    """median raw peak displacement of the given static objects (>= 2 tracked), else None.  disp [K, F, 2], ok [K, F]"""
    import numpy as np
    pk = [float(np.linalg.norm(disp[k][ok[k]], axis=-1).max()) for k in static_idx if k < len(disp) and ok[k].any()]
    return float(np.median(pk)) if len(pk) >= 2 else None
