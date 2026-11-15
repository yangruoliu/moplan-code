"""Cross-component consistency checks of the release (CPU, no weights): weight manifests, config paths, launcher
flags against the argparse definitions of the scripts they call, the showcase commands against generate.py, and the
showcase helper (case numbers, the README case table, verify on references that are evaluation exports)."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import shlex
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checksums_match_weights_manifest():
    checksums = json.loads((REPO / "setup" / "checksums.json").read_text())["components"]["moplan"]
    manifest = json.loads((REPO / "tools" / "weights_manifest.json").read_text())
    assert checksums["repo_id"] == manifest["repo_id"] == "yrlyrl/MoPlan"
    recorded = {k: (v["size"], v["sha256"]) for k, v in checksums["files"].items()}
    released = {e["target"]: (e["size"], e["sha256"]) for e in manifest["files"]}
    assert recorded == released


def test_check_env_markers_are_released_files():
    check_env = _load(REPO / "setup" / "check_env.py", "check_env_for_test")
    released = {e["target"] for e in json.loads((REPO / "tools" / "weights_manifest.json").read_text())["files"]
                if e.get("required", True)}
    assert set(check_env.WEIGHT_MARKERS["MOPLAN_WEIGHTS"]) == released


@pytest.mark.parametrize("rel", ["configs/stage1/train.yaml", "configs/stage2/inference.yaml", "configs/stage2/sft.yaml",
                                 "configs/stage2/raft.yaml", "configs/stage2/sft_overrides.json",
                                 "configs/stage2/raft_overrides.json"])
def test_shipped_configs_exist(rel):
    assert (REPO / rel).is_file()


def test_every_referenced_repo_config_exists():
    pattern = re.compile(r"configs/(?:stage1|stage2)/[A-Za-z0-9_.-]+\.(?:yaml|json)")
    missing = set()
    for path in list(REPO.rglob("*.py")) + list((REPO / "launch").glob("*.sh")) + list((REPO / "docs").glob("*.md")):
        if "__pycache__" in path.parts:
            continue
        for rel in pattern.findall(path.read_text(errors="replace")):
            if not (REPO / rel).is_file():
                missing.add((str(path.relative_to(REPO)), rel))
    assert not missing, sorted(missing)


def _argparse_flags(script: Path) -> set[str]:
    flags = set()
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    flags.add(arg.value)
    text = script.read_text()
    for group in re.findall(r"for name in \(([^)]*)\):\s*\n\s*\w+\.add_argument\('--' \+ name", text):
        flags.update("--" + n for n in re.findall(r"'([\w-]+)'", group))
    return flags


def _logical_lines(text: str):
    buffer = ""
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        yield buffer + stripped
        buffer = ""


CALL = re.compile(r"((?:scripts|setup)/[\w/.-]+\.py|generate\.py)")


def test_launcher_flags_are_defined():
    """Every --flag a launcher passes on the line that calls a repo script exists in that script's argparse."""
    problems = []
    for launcher in sorted((REPO / "launch").glob("*.sh")) + sorted((REPO / "setup").glob("*.sh")):
        for line in _logical_lines(launcher.read_text()):
            if line.lstrip().startswith("#"):
                continue
            for match in CALL.finditer(line):
                script = REPO / match.group(1)
                if not script.is_file():
                    problems.append((launcher.name, match.group(1), "missing script"))
                    continue
                rest = re.split(r"\s(?:\||;|&&|>|2>)\s|" + CALL.pattern, line[match.end():])[0]
                defined = _argparse_flags(script)
                for flag in sorted(set(re.findall(r"(?<![\w-])(--[a-z][\w-]*)", rest)) - defined - {"--help"}):
                    problems.append((launcher.name, match.group(1), flag))
    assert not problems, problems


def test_showcase_commands_parse_with_generate():
    generate = _load(REPO / "generate.py", "generate_for_test")
    count = 0
    for meta_path in sorted((REPO / "assets" / "showcase").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        for row in meta["layout"]["rows"]:
            for panel in row["panels"]:
                command = panel.get("command")
                if not command:
                    continue
                argv = shlex.split(command)
                assert argv[:2] == ["python", "generate.py"], command
                args = generate.parse_args(argv[2:])
                assert Path(args.image).as_posix().startswith("assets/showcase/") and (REPO / args.image).is_file()
                assert args.output.endswith(f"/{panel['key']}.mp4")
                assert args.protocol == meta["protocol"]["name"]
                count += 1
    assert count > 0


def test_added_data_helpers_import():
    for rel in ("scripts/motionbench/ssv2_select.py", "scripts/motionbench/ssv2_instructions.py",
                "scripts/control/rerender_real.py", "scripts/data_real/p101_splits.py"):
        _load(REPO / rel, "helper_" + Path(rel).stem)
    from scripts.stage1_real import ssv2_train_select  # noqa: F401  (imports ssv2_select / ssv2_instructions)


def _showcase():
    return _load(REPO / "assets" / "showcase" / "showcase.py", "showcase_for_test")


def test_showcase_commands_unknown_case_is_an_error(capsys):
    showcase = _showcase()
    numbers = [c["number"] for c in showcase.load_cases()]
    assert numbers == list(range(1, len(numbers) + 1))
    assert showcase.main(["commands", "--case", str(len(numbers) + 1)]) == 2
    assert "there is no case" in capsys.readouterr().err
    assert showcase.main(["commands", "--case", "1"]) == 0
    assert capsys.readouterr().out.count("python generate.py") == 4


def test_showcase_verify_does_not_fail_on_export_references(tmp_path, capsys):
    showcase = _showcase()
    unverifiable = [(c["folder"], p["key"]) for c in showcase.load_cases() for _r, p in showcase.generated_panels(c)
                    if p["reference"].get("verifiable") is False]
    assert unverifiable, "expected the RigidBench panels to be marked as not verifiable"
    for folder, key in unverifiable:
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / f"{key}.npy").write_bytes(b"not the reference")
    assert showcase.main(["verify", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "DIFFERS" not in out and out.count("NOT-VERIFIABLE") == len(unverifiable)


def test_showcase_readme_table_and_case_numbers_are_current():
    showcase = _showcase()
    cases = showcase.load_cases()
    readme = (REPO / "assets" / "showcase" / "README.md").read_text()
    a = readme.index(showcase.TABLE_START) + len(showcase.TABLE_START)
    assert readme[a:readme.index(showcase.TABLE_END)].strip() == showcase.readme_table(cases)
    top = max(c["number"] for c in cases)
    for path in (REPO / "assets" / "showcase" / "README.md", REPO / "LICENSE_NOTICE.md", REPO / "README.md"):
        for m in re.finditer(r"\b[Cc]ases? ((?:\d+(?:\s*[–-]\s*\d+)?(?:,\s*|\s+and\s+)?)+)", path.read_text()):
            numbers = [int(n) for n in re.findall(r"\d+", m.group(1))]
            assert max(numbers) <= top, (path.name, m.group(0))
