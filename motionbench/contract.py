"""Constants of the MoProbe measurement. Change thresholds here only."""
from __future__ import annotations

import math

# ---- geometry ---------------------------------------------------------------
WORK_H, WORK_W = 256, 448                      # 16:9 letterboxed work frame; px below are work px
DIAG = math.hypot(WORK_W, WORK_H)              # 516.1 px

# ---- displacement thresholds (work px) --------------------------------------
MOVE_PX = 8.0      # a cell counts as "moving" at or above this
STILL_PX = 4.0     # a cell counts as "still" at or below this; 4–8 is an unscored band
ONSET_PX = 4.0     # onset = first crossing of this displacement
NONZERO_REL = 0.25 # generated motion is "nonzero" if |p| >= max(NONZERO_REL*|g|, NONZERO_ABS)
NONZERO_ABS = 4.0
MIN_OBJECT_AREA = 64.0   # px² in the work frame; smaller instances only count for vanish

# ---- time grid ----------------------------------------------------------------
ANCHORS_S = (0.2, 0.4, 0.6, 0.85, 1.2, 2.0, 3.0, 4.0)
MAX_INTERP_GAP_S = 0.5   # an anchor is unsupported if the bracketing valid frames are further apart
ACTIVE_STEP_S = 1.0 / 7.0  # activity is judged on displacement increments over this step
ACTIVE_STEP_PX = 2.0
FREEZE_AFTER_S = 0.4     # froze_early: no activity after onset + this while the reference is active
END_WINDOW_S = 0.3       # "still moving at the end" window

# ---- direction ------------------------------------------------------------------
COS30 = math.cos(math.radians(30.0))
COS60 = 0.5
SIGN_MIN_PX = 4.0        # axis sign agreement needs |g_axis| >= this and |p_axis| >= 1 px
SIGN_PRED_MIN_PX = 1.0

# ---- magnitude --------------------------------------------------------------------
LOG_MAG_CLIP = 3.0
SHRINK_BIAS = -1.0       # log2 ratio below this → shrink flag
OVERSHOOT_BIAS = 0.5     # log2 ratio above this → overshoot flag

# ---- identity / camera ---------------------------------------------------------------
VANISH_AREA_PX = 4.0
VANISH_MIN_S = 0.2              # the final gone run must span at least this long (a last-frame dropout is not a vanish)
SHAPE_LOG2_RATIO = 1.0          # a 2x area change relative to the real object's own area change ...
SHAPE_PERSIST_FRAC = 0.5        # ... on at least half of the supported anchors (hand occlusion is transient)
CAM_DRIFT_PX = 20.0
CAMERA_MOVING_REF_PX = 20.0
SNAP_BACK_END_PX = STILL_PX

ROLES = ("actor", "patient", "static", "hand", "unknown")

# ---- metric registry ------------------------------------------------------------------
# kind: R = needs a reference continuation, F = reference-free. level: clip | pair | track.
# better: +1 higher is better, -1 lower is better, 0 diagnostic.
METRICS = [
    # F1 wrong actor
    dict(name="actor_hit", failure="F1", kind="R", level="clip", primary=True, better=+1),
    dict(name="wrong_actor", failure="F1", kind="R", level="clip", primary=False, better=-1),
    dict(name="actor_missing", failure="F1", kind="R", level="clip", primary=False, better=-1),
    # F2 static leak
    dict(name="static_leak_worst", failure="F2", kind="R", level="clip", primary=True, better=-1),
    dict(name="static_leak_objects", failure="F2", kind="R", level="clip", primary=False, better=-1),
    dict(name="hand_drag", failure="F2", kind="R", level="clip", primary=False, better=-1),
    dict(name="bg_moved", failure="F2", kind="F", level="clip", primary=False, better=-1),
    # F3 near-zero / freeze
    dict(name="moved_any", failure="F3", kind="F", level="clip", primary=True, better=+1),
    dict(name="nearzero", failure="F3", kind="R", level="clip", primary=False, better=-1),
    dict(name="froze_early", failure="F3", kind="R", level="clip", primary=False, better=-1),
    # F4 direction
    dict(name="dir60", failure="F4", kind="R", level="clip", primary=True, better=+1),
    dict(name="dir30", failure="F4", kind="R", level="clip", primary=False, better=+1),
    dict(name="dir60_nz", failure="F4", kind="R", level="clip", primary=False, better=+1),
    dict(name="sign_x", failure="F4", kind="R", level="clip", primary=False, better=+1),
    dict(name="sign_y", failure="F4", kind="R", level="clip", primary=False, better=+1),
    dict(name="down_excess", failure="F4", kind="R", level="track", primary=False, better=-1),
    dict(name="down_bias", failure="F4", kind="R", level="track", primary=False, better=-1),
    # F5 instruction
    dict(name="cf_follows", failure="F5", kind="R", level="pair", primary=False, better=+1),
    dict(name="cf_response", failure="F5", kind="R", level="pair", primary=False, better=+1),
    dict(name="cf_response_free", failure="F5", kind="F", level="pair", primary=False, better=+1),
    dict(name="cf_actor_switch", failure="F5", kind="R", level="pair", primary=False, better=+1),
    dict(name="static_caption_false_motion", failure="F5", kind="F", level="clip", primary=True, better=-1),
    # F6 magnitude
    dict(name="log_mag_err", failure="F6", kind="R", level="clip", primary=True, better=-1),
    dict(name="log_mag_bias", failure="F6", kind="R", level="clip", primary=False, better=0),
    dict(name="shrink", failure="F6", kind="R", level="clip", primary=False, better=-1),
    dict(name="overshoot", failure="F6", kind="R", level="clip", primary=False, better=-1),
    # F7 timing
    dict(name="onset_err_s", failure="F7", kind="R", level="clip", primary=False, better=0),
    dict(name="onset_missing", failure="F7", kind="R", level="clip", primary=False, better=-1),
    dict(name="stop_err_s", failure="F7", kind="R", level="clip", primary=False, better=0),
    dict(name="never_stops", failure="F7", kind="R", level="clip", primary=False, better=-1),
    dict(name="contact_order_ok", failure="F7", kind="R", level="clip", primary=False, better=+1),
    # F8 identity
    dict(name="lost_rate", failure="F8", kind="R", level="clip", primary=False, better=-1),
    dict(name="vanish", failure="F8", kind="R", level="clip", primary=True, better=-1),
    dict(name="shape_change", failure="F8", kind="R", level="clip", primary=False, better=-1),
    dict(name="snap_back", failure="F8", kind="R", level="clip", primary=False, better=-1),
    # F9 camera
    dict(name="cam_drift_px", failure="F9", kind="R", level="clip", primary=True, better=-1),
    dict(name="cam_drift_flag", failure="F9", kind="R", level="clip", primary=False, better=-1),
    # composite
    dict(name="pass_1", failure="ladder", kind="R", level="clip", primary=False, better=+1),
    dict(name="pass_2", failure="ladder", kind="R", level="clip", primary=False, better=+1),
    dict(name="pass_3", failure="ladder", kind="R", level="clip", primary=False, better=+1),
    dict(name="pass_4", failure="ladder", kind="R", level="clip", primary=True, better=+1),
]
METRIC_BY_NAME = {m["name"]: m for m in METRICS}
PRIMARY_METRICS = [m["name"] for m in METRICS if m["primary"]]

# PisaBench real clips were recorded in slow motion at 120 fps (cell phones on tripods; PISA paper, arXiv 2503.09595)
# and are stored as 30 fps movie.mp4 files (360 frames = 12 s of file time = 3 s of real time).  Every reader of a PISA
# movie must pass fps_override=PISA_CAPTURE_FPS, otherwise the reference runs 4x slower than physical time.
PISA_CAPTURE_FPS = 120.0
