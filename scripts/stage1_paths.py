"""Filesystem placeholders of the Stage-1 configs.

``configs/stage1/train.yaml`` writes every location as one of

* ``${MOPLAN_DATA}``     built Stage-1 data root (``--data-root``),
* ``${WAN_MODEL_DIR}``   Wan2.2-I2V-A14B-Diffusers directory (``--wan-model-dir``),
* ``${MOPLAN_WORK_DIR}`` run output root (``--work-dir``).

Each value comes from the command-line option when given, otherwise from the
environment variable of the same name. Only path strings change; no other
config value is touched.
"""
from __future__ import annotations

import argparse
import os
import re

# placeholder -> (argparse attribute, environment variable)
PLACEHOLDER_SOURCES = {
    "MOPLAN_DATA": ("data_root", "MOPLAN_DATA"),
    "WAN_MODEL_DIR": ("wan_model_dir", "WAN_MODEL_DIR"),
    "MOPLAN_WORK_DIR": ("work_dir", "MOPLAN_WORK_DIR"),
}
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def add_placeholder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="",
                        help="fills ${MOPLAN_DATA} in the config (default: $MOPLAN_DATA)")
    parser.add_argument("--wan-model-dir", default="",
                        help="Wan2.2-I2V-A14B-Diffusers directory; fills ${WAN_MODEL_DIR} "
                             "(default: $WAN_MODEL_DIR)")
    parser.add_argument("--work-dir", default="",
                        help="run output root; fills ${MOPLAN_WORK_DIR} (default: $MOPLAN_WORK_DIR)")


def placeholder_values(args: argparse.Namespace | None = None,
                       environ: dict[str, str] | None = None) -> dict[str, str]:
    """Collect placeholder values; a command-line value wins over the environment."""
    environ = os.environ if environ is None else environ
    values = {}
    for name, (attribute, variable) in PLACEHOLDER_SOURCES.items():
        value = getattr(args, attribute, "") if args is not None else ""
        value = value or environ.get(variable, "")
        if value:
            values[name] = os.path.abspath(os.path.expanduser(value))
    return values


def resolve_placeholders(config, values: dict[str, str], *, strict: bool = True):
    """Return a copy of ``config`` with every ``${NAME}`` in a string replaced.

    Only the names in ``PLACEHOLDER_SOURCES`` are accepted. With ``strict`` a
    placeholder without a value is an error, so a training run never starts
    with an unresolved path; data tools pass ``strict=False`` and leave such
    strings unchanged (a path they actually open then fails visibly). Configs
    without placeholders come back unchanged.
    """
    def substitute(match: re.Match) -> str:
        name = match.group(1)
        if name not in PLACEHOLDER_SOURCES:
            raise ValueError(f"unknown config placeholder ${{{name}}}")
        if name not in values:
            if not strict:
                return match.group(0)
            attribute, variable = PLACEHOLDER_SOURCES[name]
            raise ValueError(
                f"config uses ${{{name}}} but it is not set; pass "
                f"--{attribute.replace('_', '-')} or set {variable}"
            )
        return values[name]

    if isinstance(config, dict):
        return {key: resolve_placeholders(value, values, strict=strict) for key, value in config.items()}
    if isinstance(config, list):
        return [resolve_placeholders(value, values, strict=strict) for value in config]
    if isinstance(config, str) and "${" in config:
        return _PLACEHOLDER.sub(substitute, config)
    return config
