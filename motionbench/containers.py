"""Containers shared by the instrument, the reference builders and the metrics."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import contract as C


@dataclass
class Reading:
    """What the instrument measures on ONE clip (generated or real), in work px.

    times   [F]      seconds from the conditioning frame (times[0] == 0)
    pos0    [K,2]    centroid of each instance in frame 0 (x, y)
    disp    [K,F,2]  centroid displacement vs frame 0 (NOT camera-compensated)
    ok      [K,F]    reading valid (mask found, area ratio within gate)
    area    [K,F]    mask area in px² (0 when not ok)
    camera  [F,2]    camera translation estimate (median of the free background grid)
    bg_disp [P,F,2]  free background grid displacement (optional)
    bg_ok   [P,F]
    """
    times: np.ndarray
    pos0: np.ndarray
    disp: np.ndarray
    ok: np.ndarray
    area: np.ndarray
    camera: np.ndarray
    bg_disp: Optional[np.ndarray] = None
    bg_ok: Optional[np.ndarray] = None
    fps: float = float("nan")
    meta: Dict = field(default_factory=dict)
    pt_disp: Optional[np.ndarray] = None   # [K,F,2] median displacement of each object's tracked points (not compensated)
    pt_ok: Optional[np.ndarray] = None     # [K,F]   >= 3 of its points visible at frame 0 and f

    # -- helpers ---------------------------------------------------------------------
    @property
    def n_objects(self) -> int:
        return int(self.disp.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.times[-1])

    def validate(self) -> "Reading":
        t = np.asarray(self.times, dtype=np.float64)
        if t.ndim != 1 or t.size < 2 or abs(float(t[0])) > 1e-6 or np.any(np.diff(t) <= 0):
            raise ValueError("times must be 1-D, start at 0 and increase")
        K, F = self.disp.shape[:2]
        if self.disp.shape != (K, F, 2) or self.ok.shape != (K, F) or self.area.shape != (K, F):
            raise ValueError("disp/ok/area shapes disagree")
        if self.pos0.shape != (K, 2) or self.camera.shape != (F, 2) or F != t.size:
            raise ValueError("pos0/camera shapes disagree with times")
        if self.bg_disp is not None:
            if self.bg_ok is None or self.bg_disp.shape[1:] != (F, 2) or self.bg_ok.shape != self.bg_disp.shape[:2]:
                raise ValueError("bg_disp/bg_ok shapes disagree")
        if self.pt_disp is not None:
            if self.pt_ok is None or self.pt_disp.shape != (K, F, 2) or self.pt_ok.shape != (K, F):
                raise ValueError("pt_disp/pt_ok shapes disagree")
            self.pt_ok = self.pt_ok.astype(bool)
        self.times = t
        self.ok = self.ok.astype(bool)
        return self

    def compensated(self) -> np.ndarray:
        """Camera-compensated displacement [K,F,2]."""
        return self.disp - self.camera[None, :, :]

    def pt_compensated(self) -> Optional[np.ndarray]:
        return None if self.pt_disp is None else self.pt_disp - self.camera[None, :, :]

    # -- npz round trip -------------------------------------------------------------------
    def to_npz(self, path) -> None:
        d = dict(times=self.times, pos0=self.pos0, disp=self.disp, ok=self.ok, area=self.area,
                 camera=self.camera, fps=np.float64(self.fps))
        if self.bg_disp is not None:
            d["bg_disp"] = self.bg_disp
            d["bg_ok"] = self.bg_ok
        if self.pt_disp is not None:
            d["pt_disp"] = self.pt_disp
            d["pt_ok"] = self.pt_ok
        np.savez_compressed(path, **d)

    @classmethod
    def from_npz(cls, path) -> "Reading":
        z = np.load(path, allow_pickle=False)
        return cls(times=z["times"], pos0=z["pos0"], disp=z["disp"], ok=z["ok"], area=z["area"],
                   camera=z["camera"], bg_disp=z["bg_disp"] if "bg_disp" in z else None,
                   bg_ok=z["bg_ok"] if "bg_ok" in z else None, fps=float(z["fps"]),
                   pt_disp=z["pt_disp"] if "pt_disp" in z else None, pt_ok=z["pt_ok"] if "pt_ok" in z else None).validate()


@dataclass
class Reference:
    """The real continuation measured with the same instrument, plus object-level labels."""
    reading: Reading
    moving: np.ndarray    # [K] bool, object moved in the real clip (relabel_v2 rule)
    static: np.ndarray    # [K] bool, object stayed still in the real clip
    primary: int = -1     # index of the object with the largest real displacement, -1 if none

    def validate(self) -> "Reference":
        self.reading.validate()
        K = self.reading.n_objects
        self.moving = np.asarray(self.moving, dtype=bool)
        self.static = np.asarray(self.static, dtype=bool)
        if self.moving.shape != (K,) or self.static.shape != (K,):
            raise ValueError("moving/static must be [K]")
        if np.any(self.moving & self.static):
            raise ValueError("an object cannot be both moving and static")
        return self

    def to_npz(self, path) -> None:
        d = dict(times=self.reading.times, pos0=self.reading.pos0, disp=self.reading.disp,
                 ok=self.reading.ok, area=self.reading.area, camera=self.reading.camera,
                 fps=np.float64(self.reading.fps), moving=self.moving, static=self.static,
                 primary=np.int64(self.primary))
        if self.reading.bg_disp is not None:
            d["bg_disp"] = self.reading.bg_disp
            d["bg_ok"] = self.reading.bg_ok
        if self.reading.pt_disp is not None:
            d["pt_disp"] = self.reading.pt_disp
            d["pt_ok"] = self.reading.pt_ok
        np.savez_compressed(path, **d)

    @classmethod
    def from_npz(cls, path) -> "Reference":
        z = np.load(path, allow_pickle=False)
        r = Reading(times=z["times"], pos0=z["pos0"], disp=z["disp"], ok=z["ok"], area=z["area"],
                    camera=z["camera"], bg_disp=z["bg_disp"] if "bg_disp" in z else None,
                    bg_ok=z["bg_ok"] if "bg_ok" in z else None, fps=float(z["fps"]),
                    pt_disp=z["pt_disp"] if "pt_disp" in z else None, pt_ok=z["pt_ok"] if "pt_ok" in z else None)
        return cls(reading=r, moving=z["moving"], static=z["static"], primary=int(z["primary"])).validate()


@dataclass
class Item:
    """One benchmark item = (first frame, instruction) with object roles."""
    sid: str
    source: str
    group_id: str
    roles: List[str]                       # per instance, one of contract.ROLES
    instruction: str = ""
    event_tags: List[str] = field(default_factory=list)
    rigid_support: bool = True
    static_caption: bool = False           # instruction says nothing happens / hold still
    hidden_mover: bool = False
    named: Optional[List[bool]] = None     # instance named in the instruction
    hand_score: Optional[List[float]] = None
    counterfactual_sibling: Optional[str] = None
    instr_dir: Optional[Sequence[float]] = None   # unit vector of the instructed direction (x right, y down)
    actor_patient: Optional[Sequence[int]] = None # (actor_idx, patient_idx) for two-object events
    trusted: Optional[List[bool]] = None   # annotated instance (actor/patient/hand or matched to a box); others = SAM2 regions
    meta: Dict = field(default_factory=dict)

    def validate(self, n_objects: Optional[int] = None) -> "Item":
        for r in self.roles:
            if r not in C.ROLES:
                raise ValueError(f"unknown role {r!r}")
        if n_objects is not None and len(self.roles) != n_objects:
            raise ValueError("roles length must equal the number of instances")
        return self

    @property
    def actor_idx(self) -> List[int]:
        return [i for i, r in enumerate(self.roles) if r == "actor"]

    @property
    def hand_idx(self) -> List[int]:
        return [i for i, r in enumerate(self.roles) if r == "hand"]

    def to_dict(self) -> Dict:
        return asdict(self)
