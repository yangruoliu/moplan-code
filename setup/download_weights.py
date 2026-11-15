#!/usr/bin/env python3
"""Download and verify the model weights used by MoPlan.

Components (sources, revisions and checksums are pinned in setup/checksums.json):

  wan        Wan-AI/Wan2.2-I2V-A14B-Diffusers  -> $WAN_MODEL_DIR    (about 126 GB)
  moplan     yrlyrl/MoPlan                      -> $MOPLAN_WEIGHTS   (about 2.4 GB)
  sam2       SAM 2.1 Hiera-L checkpoint         -> $SAM2_CHECKPOINT  (about 0.9 GB)
  cotracker  CoTracker3 scaled_offline.pth      -> $COTRACKER_CKPT   (about 0.1 GB)

An unset variable falls back to a default path under <repo>/weights/ (see the
"default_path" entries of setup/checksums.json); the --*-dir / --*-checkpoint
flags override both.

Examples:
  python setup/download_weights.py --offline       # list targets and local state; no network
  python setup/download_weights.py --dry-run       # also query remote metadata; download nothing
  python setup/download_weights.py                 # download everything
  python setup/download_weights.py --only wan,sam2
  python setup/download_weights.py --verify-only   # hash local files; download nothing

Downloads resume after an interruption: run the same command again. Every file
is checked against its recorded size and sha256. Verified files are remembered
in <target>/.moplan_verified.json (keyed by size and mtime), so a rerun does not
hash them again; --rehash ignores that record.

Exit status: 0 on success, 1 if a file is missing or fails verification (or a
download fails), 2 on usage or checksum-file errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKSUMS = Path(__file__).resolve().with_name("checksums.json")
STAMP_NAME = ".moplan_verified.json"
COMPONENTS = ("wan", "moplan", "sam2", "cotracker")
PATH_FLAGS = {
    "wan": "wan_dir",
    "moplan": "moplan_dir",
    "sam2": "sam2_checkpoint",
    "cotracker": "cotracker_checkpoint",
}
# Repository files that are never needed at run time.
SKIP_REMOTE = {".gitattributes"}
CHUNK = 8 << 20


def log(msg: str = "") -> None:
    print(msg, flush=True)


def human(n: int | None) -> str:
    if n is None:
        return "size unknown"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1000
    return f"{n} B"


# ----------------------------------------------------------------------------- checksum file


def load_checksums(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: cannot read checksum file {path}: {exc}") from exc
    comps = data.get("components")
    if data.get("format") != 1 or not isinstance(comps, dict):
        raise SystemExit(f"error: {path}: expected format 1 with a 'components' mapping")
    for name, spec in comps.items():
        for key in ("env", "default_path", "layout", "download", "repo_id", "files"):
            if key not in spec:
                raise SystemExit(f"error: {path}: component '{name}' lacks '{key}'")
        if spec["layout"] not in ("dir", "file"):
            raise SystemExit(f"error: {path}: component '{name}': layout must be 'dir' or 'file'")
        if spec["download"] not in ("listed", "all"):
            raise SystemExit(f"error: {path}: component '{name}': download must be 'listed' or 'all'")
        if spec["layout"] == "file" and len(spec["files"]) != 1:
            raise SystemExit(f"error: {path}: component '{name}': layout 'file' needs exactly one file")
        for rel, entry in spec["files"].items():
            sha = entry.get("sha256")
            if sha is not None and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
                raise SystemExit(f"error: {path}: {name}/{rel}: sha256 must be 64 lowercase hex digits or null")
    return data


# ----------------------------------------------------------------------------- targets


class Target:
    """One component: its pinned source and where its files go on disk."""

    def __init__(self, name: str, spec: dict, path: Path, origin: str):
        self.name = name
        self.spec = spec
        self.path = path
        self.origin = origin

    @property
    def is_file(self) -> bool:
        return self.spec["layout"] == "file"

    @property
    def root(self) -> Path:
        return self.path.parent if self.is_file else self.path

    @property
    def revision(self) -> str | None:
        return self.spec.get("revision")

    @property
    def files(self) -> dict:
        return self.spec["files"]

    def local_path(self, rel: str) -> Path:
        return self.path if self.is_file else self.path / rel

    def stamp_key(self, rel: str) -> str:
        return self.path.name if self.is_file else rel


def resolve_target(name: str, spec: dict, args: argparse.Namespace) -> Target:
    flag = getattr(args, PATH_FLAGS[name])
    env = os.environ.get(spec["env"])
    if flag:
        path, origin = Path(flag), f"--{PATH_FLAGS[name].replace('_', '-')}"
    elif env:
        path, origin = Path(env), f"${spec['env']}"
    else:
        path, origin = REPO_ROOT / spec["default_path"], f"default (${spec['env']} unset)"
    return Target(name, spec, path.expanduser().absolute(), origin)


# ----------------------------------------------------------------------------- verification


def read_stamps(root: Path) -> dict:
    try:
        data = json.loads((root / STAMP_NAME).read_text())
        return data.get("files", {}) if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_stamps(root: Path, stamps: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (STAMP_NAME + ".tmp")
    tmp.write_text(json.dumps({"format": 1, "files": stamps}, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, root / STAMP_NAME)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp_matches(stamps: dict, key: str, path: Path, sha: str | None) -> bool:
    rec = stamps.get(key)
    if not rec or sha is None or rec.get("sha256") != sha:
        return False
    st = path.stat()
    return rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns


def local_state(target: Target, rel: str, entry: dict, stamps: dict) -> str:
    """Cheap status without hashing: missing / size-mismatch / verified / present."""
    path = target.local_path(rel)
    if not path.is_file():
        return "missing"
    size = entry.get("size")
    if size is not None and path.stat().st_size != size:
        return "size-mismatch"
    if stamp_matches(stamps, target.stamp_key(rel), path, entry.get("sha256")):
        return "verified"
    return "present"


def verify(target: Target, rel: str, entry: dict, stamps: dict, rehash: bool) -> tuple[bool | None, str]:
    """Return (True, msg) if verified, (None, msg) if there is no recorded hash, (False, msg) on failure."""
    path = target.local_path(rel)
    if not path.is_file():
        return False, "missing"
    size, sha = entry.get("size"), entry.get("sha256")
    actual_size = path.stat().st_size
    if size is not None and actual_size != size:
        return False, f"size {actual_size} != expected {size}"
    if sha is None:
        return None, "present, no recorded sha256"
    key = target.stamp_key(rel)
    if not rehash and stamp_matches(stamps, key, path, sha):
        return True, "verified (cached)"
    if actual_size > (1 << 30):
        log(f"    hashing {rel} ({human(actual_size)}) ...")
    got = sha256_file(path)
    if got != sha:
        stamps.pop(key, None)
        return False, f"sha256 {got} != expected {sha}"
    st = path.stat()
    stamps[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
    return True, "verified"


def free_bytes(path: Path) -> int | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


# ----------------------------------------------------------------------------- Hugging Face access


def hub():
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit("error: huggingface_hub is not installed (pip install -r requirements.txt)") from exc
    return huggingface_hub


def remote_listing(target: Target) -> dict:
    """{path: {"size": int, "sha256": str | None}} for every file of the pinned revision."""
    api = hub().HfApi()
    out = {}
    for item in api.list_repo_tree(
        target.spec["repo_id"],
        revision=target.revision,
        repo_type=target.spec.get("repo_type", "model"),
        recursive=True,
    ):
        size = getattr(item, "size", None)
        if size is None:  # folder entry
            continue
        lfs = getattr(item, "lfs", None)
        if isinstance(lfs, dict):
            sha = lfs.get("sha256") or lfs.get("oid")
        else:
            sha = getattr(lfs, "sha256", None)
        out[item.path] = {"size": size, "sha256": sha}
    return out


def hf_download(target: Target, rel: str) -> Path:
    """Download one file into the target (resumable) and return its final path."""
    target.root.mkdir(parents=True, exist_ok=True)
    got = Path(
        hub().hf_hub_download(
            target.spec["repo_id"],
            filename=rel,
            revision=target.revision,
            repo_type=target.spec.get("repo_type", "model"),
            local_dir=str(target.root),
        )
    )
    final = target.local_path(rel)
    if got.resolve() != final.resolve():
        os.replace(got, final)
    return final


def discard(target: Target, rel: str) -> None:
    """Remove a bad local copy and its download metadata so the next attempt starts clean."""
    path = target.local_path(rel)
    if path.is_file():
        path.unlink()
    meta = target.root / ".cache" / "huggingface" / "download" / (rel + ".metadata")
    if meta.is_file():
        meta.unlink()


# ----------------------------------------------------------------------------- per-component actions


def describe(target: Target) -> None:
    spec = target.spec
    rev = target.revision or "main (not pinned)"
    log(f"[{target.name}] {spec.get('description', '')}")
    log(f"  source : https://huggingface.co/{spec['repo_id']} @ {rev}")
    log(f"  target : {target.path}  [{target.origin}]")
    if spec.get("license"):
        log(f"  licence: {spec['license']}")


def plan_files(target: Target, remote: dict | None) -> dict:
    """Files to fetch: the recorded list, or for download='all' every remote file."""
    if target.spec["download"] == "listed" or remote is None:
        return dict(target.files)
    plan = {}
    for rel, meta in sorted(remote.items()):
        if rel in SKIP_REMOTE:
            continue
        rec = target.files.get(rel, {})
        size = rec.get("size") if rec.get("size") is not None else meta["size"]
        plan[rel] = {"size": size, "sha256": rec.get("sha256")}
    return plan


def check_remote(target: Target, remote: dict) -> list[str]:
    """Differences between the recorded checksums and the remote metadata."""
    problems = []
    for rel, entry in sorted(target.files.items()):
        meta = remote.get(rel)
        if meta is None:
            problems.append(f"{rel}: recorded but not in the remote revision")
            continue
        if entry.get("size") is not None and meta["size"] != entry["size"]:
            problems.append(f"{rel}: remote size {meta['size']} != recorded {entry['size']}")
        if entry.get("sha256") and meta.get("sha256") and meta["sha256"] != entry["sha256"]:
            problems.append(f"{rel}: remote sha256 {meta['sha256']} != recorded {entry['sha256']}")
    return problems


def run_listing(target: Target, online: bool, verbose: bool) -> int:
    describe(target)
    remote, status = None, 0
    if online:
        try:
            remote = remote_listing(target)
        except Exception as exc:  # network, auth or a repository that is not published yet
            log(f"  remote : not reachable ({type(exc).__name__}); showing the recorded file list")
        else:
            problems = check_remote(target, remote)
            log(f"  remote : {len(remote)} files at this revision")
            for msg in problems:
                log(f"  MISMATCH {msg}")
            status = 1 if problems else 0
    plan = plan_files(target, remote)
    stamps = read_stamps(target.root)
    counts: dict[str, int] = {}
    fetch_bytes, fetch_unknown, total, total_unknown = 0, 0, 0, 0
    for rel, entry in sorted(plan.items()):
        state = local_state(target, rel, entry, stamps)
        counts[state] = counts.get(state, 0) + 1
        total += entry.get("size") or 0
        total_unknown += entry.get("size") is None
        if state in ("missing", "size-mismatch"):
            if entry.get("size") is None:
                fetch_unknown += 1
            else:
                fetch_bytes += entry["size"]
        if verbose or state in ("missing", "size-mismatch"):
            note = "" if entry.get("sha256") else "  (no recorded sha256)"
            log(f"    {state:13s} {rel}  {human(entry.get('size'))}{note}")
    summary = ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    size_txt = human(total) + (f" + {total_unknown} of unknown size" if total_unknown else "")
    log(f"  files  : {len(plan)} ({size_txt}): {summary}")
    extra = f" + {fetch_unknown} of unknown size" if fetch_unknown else ""
    free = free_bytes(target.root)
    free_txt = human(free) if free is not None else "unknown"
    log(f"  fetch  : {human(fetch_bytes)}{extra}; free space at target: {free_txt}")
    log()
    return status


def run_verify(target: Target, rehash: bool) -> int:
    describe(target)
    stamps = read_stamps(target.root)
    counts = {"verified": 0, "unverified": 0, "placeholder missing": 0, "failed": 0}
    for rel, entry in sorted(target.files.items()):
        ok, msg = verify(target, rel, entry, stamps, rehash)
        if ok is True:
            counts["verified"] += 1
        elif ok is None:
            counts["unverified"] += 1
            log(f"    note  {rel}: {msg}")
        elif entry.get("sha256") is None and msg == "missing":
            counts["placeholder missing"] += 1
            log(f"    note  {rel}: missing (entry has no recorded sha256)")
        else:
            counts["failed"] += 1
            log(f"    FAIL  {rel}: {msg}")
    if target.root.is_dir():
        write_stamps(target.root, stamps)
    log("  result : " + ", ".join(f"{n} {k}" for k, n in counts.items() if n or k == "verified"))
    log()
    return 1 if counts["failed"] else 0


def run_download(target: Target, rehash: bool, ignore_space: bool) -> int:
    describe(target)
    remote = None
    if target.spec["download"] == "all":
        try:
            remote = remote_listing(target)
        except Exception as exc:
            log(f"  error  : cannot list {target.spec['repo_id']}: {type(exc).__name__}: {exc}")
            log()
            return 1
        for msg in check_remote(target, remote):
            log(f"  warning: {msg}")
    plan = plan_files(target, remote)
    stamps = read_stamps(target.root)

    need = sum((e.get("size") or 0) for r, e in plan.items()
               if local_state(target, r, e, stamps) in ("missing", "size-mismatch"))
    free = free_bytes(target.root)
    if free is not None and need > free:
        msg = f"needs {human(need)} but only {human(free)} is free at {target.root}"
        if not ignore_space:
            log(f"  error  : {msg} (use --ignore-space to try anyway)")
            log()
            return 1
        log(f"  warning: {msg}")

    failures = 0
    for rel, entry in sorted(plan.items()):
        path = target.local_path(rel)
        action = "get  "
        if path.is_file():
            ok, msg = verify(target, rel, entry, stamps, rehash)
            if ok is True:
                continue
            if ok is False:
                log(f"    replacing {rel}: {msg}")
                discard(target, rel)
            else:
                action = "check"  # no recorded hash: let the hub client compare with the remote copy
        log(f"    {action} {rel}  {human(entry.get('size'))}")
        try:
            hf_download(target, rel)
        except Exception as exc:
            failures += 1
            log(f"    FAIL  {rel}: download failed: {type(exc).__name__}: {exc}")
            continue
        ok, msg = verify(target, rel, entry, stamps, rehash=True)
        if ok is False:
            failures += 1
            log(f"    FAIL  {rel}: {msg}; the file was removed, run again to retry")
            discard(target, rel)
        elif ok is None:
            log(f"    note  {rel}: {msg}")
        write_stamps(target.root, stamps)
    if target.root.is_dir():
        write_stamps(target.root, stamps)
    for rel, entry in sorted(target.files.items()):
        if rel not in plan and entry.get("sha256") is not None:
            failures += 1
            log(f"    FAIL  {rel}: recorded but not available from the source")
    log(f"  result : {len(plan) - failures} of {len(plan)} files OK")
    log()
    return 1 if failures else 0


# ----------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and verify the MoPlan model weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:" + __doc__.split("Examples:", 1)[1].split("Downloads resume", 1)[0].rstrip(),
    )
    p.add_argument("--only", action="append", default=[],
                   help=f"comma-separated subset of {','.join(COMPONENTS)} (repeatable)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="list what would be fetched, using remote metadata only")
    mode.add_argument("--offline", action="store_true",
                      help="list targets and local state from the checksum file, without network access")
    mode.add_argument("--verify-only", action="store_true",
                      help="verify local files against the checksum file; download nothing")
    p.add_argument("--wan-dir", help="Wan2.2-I2V-A14B-Diffusers directory (default: $WAN_MODEL_DIR)")
    p.add_argument("--moplan-dir", help="MoPlan weights directory (default: $MOPLAN_WEIGHTS)")
    p.add_argument("--sam2-checkpoint", help="SAM 2.1 Hiera-L .pt path (default: $SAM2_CHECKPOINT)")
    p.add_argument("--cotracker-checkpoint", help="CoTracker3 scaled_offline.pth path (default: $COTRACKER_CKPT)")
    p.add_argument("--checksums", default=str(DEFAULT_CHECKSUMS), help="checksum file (default: setup/checksums.json)")
    p.add_argument("--rehash", action="store_true", help="ignore the record of verified files and hash again")
    p.add_argument("--ignore-space", action="store_true", help="download even if free space looks too small")
    p.add_argument("-v", "--verbose", action="store_true", help="list every file in the listing modes")
    return p.parse_args(argv)


def selected(args: argparse.Namespace) -> list[str]:
    names = [n.strip() for chunk in args.only for n in chunk.split(",") if n.strip()]
    unknown = sorted(set(names) - set(COMPONENTS))
    if unknown:
        raise SystemExit(f"error: unknown component(s) {', '.join(unknown)}; choose from {', '.join(COMPONENTS)}")
    return [n for n in COMPONENTS if n in names] if names else list(COMPONENTS)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        names = selected(args)
        data = load_checksums(Path(args.checksums))
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    missing = [n for n in names if n not in data["components"]]
    if missing:
        print(f"error: {args.checksums} has no entry for {', '.join(missing)}", file=sys.stderr)
        return 2

    if not args.offline and not args.verify_only and os.environ.get("HF_HUB_OFFLINE", "0") not in ("", "0"):
        # huggingface_hub reads this variable when it is first imported.
        log("note: ignoring HF_HUB_OFFLINE for this script (use --offline to stay offline)")
        os.environ["HF_HUB_OFFLINE"] = "0"

    targets = [resolve_target(n, data["components"][n], args) for n in names]
    status = 0
    for target in targets:
        if args.offline or args.dry_run:
            status |= run_listing(target, online=args.dry_run, verbose=args.verbose)
        elif args.verify_only:
            status |= run_verify(target, args.rehash)
        else:
            status |= run_download(target, args.rehash, args.ignore_space)

    log("Environment for setup/env.sh:")
    for target in targets:
        log(f"  export {target.spec['env']}={shlex.quote(str(target.path))}")
    return status


if __name__ == "__main__":
    sys.exit(main())
