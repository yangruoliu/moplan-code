"""MoProbe measurement library: object-level motion measurement for image+text -> video generators.

Layout (the modules used by the MoProbe scripts in scripts/motionbench)
  contract.py    constants (single source of truth for thresholds)
  containers.py  Reading / Reference / Item containers (+ npz round trip)
  instrument.py  SAM 2 mask propagation + CoTracker point tracks -> Reading
  io.py          loading of built MoProbe items
  video.py       clip reading and work-frame geometry
  sam2auto.py    SAM 2 automatic masks (new-object scan)
  metrics/       camera_rule.py (camera verdict), instruction_rules.py (instruction outcomes)
"""
__version__ = "0.1.0"
