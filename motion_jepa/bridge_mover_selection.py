"""Task-object selection contract for Bridge first-frame SAM proposals."""
from __future__ import annotations

import json
import re
from typing import Any

SCHEMA = "bridge_task_object_selection_v1"
VERIFICATION_SCHEMA = "bridge_task_object_verification_v1"
CONDITION_SOURCE = "first_frame_qwen_vlm_selected_sam2_instance_masks"


def render_selection_prompt(caption: str, marks: list[int]) -> str:
    mark_text = ", ".join(str(value) for value in marks)
    return "\n".join([
        "The image is the first frame of a robot manipulation video. Colored outlines and numbers are SAM object proposals.",
        f"Task instruction: {caption.strip()}",
        f"Available marks: [{mark_text}]",
        "Select only the physical task object or objects whose motion should be predicted: the object directly pushed, pulled, lifted, released, rotated, inserted, poured, or placed by the instruction.",
        "A single physical object may be split across several SAM proposals. Select EVERY mark that covers any visible part of the same named task object (for example both a spoon handle and spoon bowl). Do not select just the most salient fragment.",
        "Exclude robot arms, grippers, hands, tools used only as actuators, the table, background, and stationary receptacles or supports. Include a receptacle only when the instruction explicitly moves that receptacle itself.",
        "Never select a rack, bowl, pot, or other container merely because it contains the named target. If no proposal isolates the named object without substantial pixels from another object or support, set needs_box_prompt to true.",
        "Do not predict the future trajectory, collision outcome, or other consequence. Identify object identity only from the first frame and instruction.",
        "If the instruction does not uniquely identify what object is acted on, return an empty mover_marks list and set instruction_sufficient to false. Do not guess from which object looks movable.",
        "If the instruction explicitly says nothing moves, set no_task_motion to true. If the task object is clear but no outlined proposal cleanly covers it, set needs_box_prompt to true.",
        'Return JSON only: {"mover_marks":[1],"actuator_marks":[2],"instruction_sufficient":true,"no_task_motion":false,"needs_box_prompt":false,"confidence":0.0,"reason":"short object-identity reason"}',
    ])


def render_verification_prompt(
    caption: str, marks: list[int], selection: dict[str, Any]
) -> str:
    """Ask an independent pass to fail closed on semantics and mask coverage."""
    mark_text = ", ".join(str(value) for value in marks)
    candidate = json.dumps(selection, ensure_ascii=False, sort_keys=True)
    return "\n".join([
        "Audit a proposed task-object label for the first frame of a robot manipulation video. Colored outlines and numbers are SAM proposals.",
        f"Task instruction: {caption.strip()}",
        f"Available marks: [{mark_text}]",
        f"Proposed label: {candidate}",
        "Accept only if the instruction is complete enough to determine both the acted-on object and the commanded action/direction or destination. Reject fragments such as 'moved to the side' or an unfinished destination.",
        "The selected marks must contain every visible proposal fragment of the named task object and must not substantially contain a robot, gripper, rack, receptacle, support, or a different object.",
        "A container that merely contains the named object is not a valid substitute for that object. If the task is clear but the proposals do not cleanly and completely cover it, set needs_box_prompt true.",
        "Judge object identity and proposal coverage only. Do not infer the future physical outcome.",
        'Return JSON only: {"accept":true,"instruction_sufficient":true,"selection_complete":true,"selection_pure":true,"needs_box_prompt":false,"confidence":0.0,"reason":"short audit reason"}',
    ])


def _json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("selection response has no JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("selection response JSON must be an object")
    return payload


def parse_selection(text: str, *, available_marks: list[int]) -> dict[str, Any]:
    payload = _json_object(text)
    allowed = set(int(value) for value in available_marks)

    def marks(key: str) -> list[int]:
        values = payload.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list")
        result = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError(f"{key} contains a boolean")
            number = int(value)
            if number not in allowed:
                raise ValueError(f"{key} contains unavailable mark {number}")
            if number not in result:
                result.append(number)
        return result

    mover = marks("mover_marks")
    actuator = marks("actuator_marks")
    if set(mover) & set(actuator):
        raise ValueError("a mark cannot be both mover and actuator")
    needs_box = payload.get("needs_box_prompt", not mover)
    if not isinstance(needs_box, bool):
        raise ValueError("needs_box_prompt must be boolean")
    instruction_sufficient = payload.get("instruction_sufficient", True)
    no_task_motion = payload.get("no_task_motion", False)
    if not isinstance(instruction_sufficient, bool):
        raise ValueError("instruction_sufficient must be boolean")
    if not isinstance(no_task_motion, bool):
        raise ValueError("no_task_motion must be boolean")
    if (not instruction_sufficient or no_task_motion) and mover:
        raise ValueError("insufficient or no-motion instructions cannot select mover marks")
    confidence = float(payload.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0,1]")
    reason = str(payload.get("reason", "")).strip()
    return {
        "schema": SCHEMA,
        "mover_marks": mover,
        "actuator_marks": actuator,
        "instruction_sufficient": instruction_sufficient,
        "no_task_motion": no_task_motion,
        "needs_box_prompt": needs_box,
        "confidence": confidence,
        "reason": reason,
    }


def parse_verification(text: str) -> dict[str, Any]:
    payload = _json_object(text)
    keys = (
        "accept", "instruction_sufficient", "selection_complete",
        "selection_pure", "needs_box_prompt",
    )
    values = {}
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be boolean")
        values[key] = value
    confidence = float(payload.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0,1]")
    logically_acceptable = (
        values["instruction_sufficient"]
        and values["selection_complete"]
        and values["selection_pure"]
        and not values["needs_box_prompt"]
    )
    if values["accept"] != logically_acceptable:
        raise ValueError("accept is inconsistent with verification fields")
    return {
        "schema": VERIFICATION_SCHEMA,
        **values,
        "confidence": confidence,
        "reason": str(payload.get("reason", "")).strip(),
    }


def select_instance_labels(labels: "torch.Tensor", mover_marks: list[int]) -> "torch.Tensor":
    """Keep selected proposal IDs and remap them densely from one."""
    import torch

    if labels.ndim != 2:
        raise ValueError("instance labels must be [H,W]")
    output = torch.zeros_like(labels, dtype=torch.uint8)
    for new_identifier, source_identifier in enumerate(mover_marks, 1):
        selected = labels == int(source_identifier)
        if not selected.any():
            raise ValueError(f"selected mark {source_identifier} has no pixels")
        output[selected] = new_identifier
    return output
