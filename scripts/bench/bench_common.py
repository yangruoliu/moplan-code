"""Shared pieces of the benchmark adapters (RigidBench, Morpheus, MoProbe, ...): the caption rule, the input canvases
and the manifest writer.

Manifest contract (what cache_sam2_masks.py / infer_stage1_physicsiq.py read from the csv, cache_stage2_cond_v2.py and
infer_stage2_cond_v2.py from the jsonl):  sample_id, scenario, video_path ("" = deployment, no future video),
first_frame_path, caption, fps, frames, width, height.  fps / frames are the GENERATION target (our native 7 fps and
the 4k+1 clip length passed to cache_stage2_cond_v2.py --frames); width / height are the native size of the benchmark's
conditioning image, which export_videos.py stretches the 448x256 work frame back to.  The jsonl carries extra
benchmark fields (bench, bench_id, ...) that the csv omits.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CSV_FIELDS = ["sample_id", "scenario", "video_path", "first_frame_path", "caption", "fps", "frames", "width", "height"]
OUR_FPS = 7.0
WORK_W, WORK_H = 448, 256                  # the fixed Stage-2 work frame (cache_stage2_cond_v2.py: PIL LANCZOS stretch)
WORK_ASPECT = WORK_W / WORK_H              # 7:4
STATIC_SUFFIX = "Static shot with no camera movement."
# a benchmark prompt that itself says the camera does not move (then, and only then, our training phrase is appended)
STATIC_RE = re.compile(
    r"static shot|static camera|fixed camera|fixed tripod|locked[- ]off|stationary (?:camera|framing)|"
    r"no camera (?:movement|motion)|camera (?:is )?(?:static|fixed|still)", re.IGNORECASE)


def says_static(prompt: str) -> bool:
    return bool(STATIC_RE.search(prompt))


def caption_for(prompt: str) -> str:
    """The benchmark prompt, plus our training-time camera phrase iff the prompt already states a static camera."""
    text = " ".join(prompt.split())
    if not says_static(text) or text.endswith(STATIC_SUFFIX):
        return text
    if text and text[-1] not in ".!?":
        text += "."
    return f"{text} {STATIC_SUFFIX}"


def copy_first_frame(source: Path, folder: Path, sample_id: str, link: bool = False) -> tuple[Path, int, int]:
    """Byte-exact copy of the conditioning image (so caches never depend on the benchmark checkout) and its size.
    link=True: a hard link instead of a copy when source and target share a file system (same bytes, no extra space)."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{sample_id}{source.suffix.lower()}"
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        target.unlink(missing_ok=True)
        try:
            if not link:
                raise OSError
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    with Image.open(target) as image:
        image.verify()
    with Image.open(target) as image:
        width, height = image.size
    return target.resolve(), width, height


def stretch_factor(width: int, height: int) -> float:
    """Anisotropic distortion of a plain stretch of a width x height image to the 7:4 work frame (1 = none)."""
    aspect = width / height
    return max(aspect / WORK_ASPECT, WORK_ASPECT / aspect)


def pillarbox_first_frame(source: Path, folder: Path, sample_id: str) -> tuple[Path, int, int, list[int]]:
    """Paste the conditioning image, unscaled, centred on a black 7:4 canvas (PNG), so that the chain's plain stretch
    to 448x256 becomes a uniform scale: the content keeps its aspect (pillarbox for portrait / square, letterbox for
    very wide images).  Returns (canvas path, canvas width, canvas height, content box [x0, y0, w, h] in canvas pixels);
    an exporter can crop the box back out of the generated work frames."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{sample_id}_pad.png"
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width / height < WORK_ASPECT:
            canvas_w, canvas_h = round(height * WORK_ASPECT), height
        else:
            canvas_w, canvas_h = width, round(width / WORK_ASPECT)
        box = [(canvas_w - width) // 2, (canvas_h - height) // 2, width, height]
        if not target.is_file():
            canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
            canvas.paste(image, (box[0], box[1]))
            temporary = target.with_name(target.stem + ".tmp.png")
            canvas.save(temporary)
            temporary.replace(target)
    with Image.open(target) as image:
        if image.size != (canvas_w, canvas_h):
            raise SystemExit(f"{target}: {image.size} != canvas {(canvas_w, canvas_h)}")
    return target.resolve(), canvas_w, canvas_h, box


def letterbox_16x9_array(img: np.ndarray, height: int = 720) -> np.ndarray:
    """The MoProbe input canvas: the native frame (BGR or RGB uint8 as read) fitted into a black 16:9 canvas of the
    given height (1280x720 by default), centred; INTER_AREA when shrinking, INTER_CUBIC when enlarging."""
    h, w = img.shape[:2]
    width = int(round(height * 16 / 9))
    scale = min(width / w, height / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    canvas = np.zeros((height, width, 3), np.uint8)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return canvas


def letterbox_16x9(src: Path, dst: Path, height: int = 720) -> None:
    """Every MoProbe generator received the same 16:9 input: cv2.imread(src) letterboxed into 1280x720, written as PNG."""
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"cannot decode image: {src}")
    canvas = letterbox_16x9_array(img, height)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), canvas):
        raise OSError(f"cannot write {dst}")


def work_geometry(source: Path, folder: Path, sample_id: str, link: bool, max_stretch: float) -> dict:
    """The conditioning image for the chain + the geometry fields every adapter writes:
    stretch_factor <= max_stretch -> byte-exact copy, plain stretch to the work frame (geometry "stretch");
    otherwise -> pillarbox_first_frame (geometry "pillarbox").  width / height = the image the chain reads (canvas for
    pillarbox), native_width / native_height = the benchmark image, content_box = where it sits in that image."""
    with Image.open(source) as image:
        native_w, native_h = image.size
    factor = stretch_factor(native_w, native_h)
    if factor <= max_stretch:
        first, width, height = copy_first_frame(source, folder, sample_id, link=link)
        box, geometry = [0, 0, width, height], "stretch"
    else:
        first, width, height, box = pillarbox_first_frame(source, folder, sample_id)
        geometry = "pillarbox"
    return {"first_frame_path": str(first), "width": width, "height": height, "native_width": native_w, "native_height": native_h,
            "work_geometry": geometry, "content_box": box, "stretch_factor": round(factor, 4),
            "work_content_px": [round(box[2] * WORK_W / width, 2), round(box[3] * WORK_H / height, 2)]}


def wan_max_area_size(width: int, height: int, max_area: int = 832 * 480) -> tuple[int, int]:
    """Official Wan2.2 WanI2V size rule (lat = round(sqrt(A * ar) // 8 // 2 * 2); size = lat * 8): the 480P-area output
    size for an input aspect."""
    aspect = height / width
    out_h = round((max_area * aspect) ** 0.5 // 8 // 2 * 2) * 8
    out_w = round((max_area / aspect) ** 0.5 // 8 // 2 * 2) * 8
    return out_w, out_h


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_against_tree(root: Path, tree_json: Path | None, paths: list[str]) -> dict:
    """Check files under root against a pinned listing: a HuggingFace tree API dump ([{path, oid, lfs: {oid}}]) or a
    `git ls-tree -r` dump turned into [{path, oid}].  LFS files compare sha256, the others the git blob sha1."""
    if tree_json is None:
        return {"verified": False, "reason": "no --tree given"}
    tree = {e["path"]: e for e in json.loads(Path(tree_json).read_text()) if e.get("type", "file") == "file"}
    bad = []
    for rel in paths:
        entry, path = tree.get(rel), root / rel
        if entry is None or not path.is_file():
            bad.append(f"{rel}: {'not in tree' if entry is None else 'missing'}"); continue
        lfs = entry.get("lfs") or {}
        if lfs.get("oid"):
            ok = hashlib.sha256(path.read_bytes()).hexdigest() == lfs["oid"]
        else:
            ok = git_blob_sha1(path) == entry["oid"]
        if not ok:
            bad.append(f"{rel}: content differs")
    if bad:
        raise SystemExit(f"{len(bad)} of {len(paths)} files differ from {tree_json}: {bad[:5]}")
    return {"verified": True, "files": len(paths), "tree": str(tree_json)}


def write_manifest(rows: list[dict], out_dir: Path) -> None:
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate sample ids")
    for row in rows:
        missing = [key for key in CSV_FIELDS if key not in row]
        if missing or not Path(row["first_frame_path"]).is_file() or not row["caption"].strip():
            raise SystemExit(f"bad row {row.get('sample_id')}: missing {missing} or image / caption")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "ids_all.txt").write_text("".join(f"{sid}\n" for sid in ids))


def check_manifest(out_dir: Path, expected: int) -> dict:
    """Re-read both files the way the chain reads them and check that they agree."""
    jsonl = [json.loads(line) for line in (out_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    with (out_dir / "manifest.csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    problems = []
    if len(jsonl) != expected or len(table) != expected:
        problems.append(f"rows: jsonl {len(jsonl)} csv {len(table)}, expected {expected}")
    for a, b in zip(jsonl, table):
        for key in CSV_FIELDS:
            if str(a[key]) != b[key]:
                problems.append(f"{a['sample_id']}: {key} differs between jsonl and csv")
        if not Path(b["first_frame_path"]).is_file():
            problems.append(f"{b['sample_id']}: no image {b['first_frame_path']}")
        frames = int(b["frames"])
        if (frames - 1) % 4:
            problems.append(f"{b['sample_id']}: {frames} frames is not 4k+1")
    sizes = sorted({(int(r["width"]), int(r["height"])) for r in table})
    static = sum(r["caption"].endswith(STATIC_SUFFIX) for r in table)
    return {"rows": len(table), "sizes": sizes, "static_suffix_appended": static, "problems": problems[:20], "n_problems": len(problems)}
