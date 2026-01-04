import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import yaml

from task_resolver import TaskInputs

TaskOutputHandler = Callable[[TaskInputs, str], Optional[Path]]


def _extract_yaml_block(text: str) -> str:
    if "```yaml" in text:
        match = re.search(r"```yaml\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1)
    if "```" in text:
        match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1)
    return text


def _ensure_results_dir(folder_path: str) -> Path:
    results_dir = Path(folder_path) / "spotter-results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_spotter_output(inputs: TaskInputs, response_text: str) -> Optional[Path]:
    folder_path = inputs.folder_path
    if not folder_path:
        print("Warning: No folder_path for spotter output; skip saving.")
        return None

    yaml_content = _extract_yaml_block(response_text)
    results_dir = _ensure_results_dir(folder_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"{timestamp}.yaml"
    output_file.write_text(yaml_content, encoding="utf-8")
    print(f"Saved spotter results to {output_file}")
    return output_file


def _normalize_dig_hard_items(raw_yaml: str) -> List[Dict[str, Any]]:
    try:
        data = yaml.safe_load(raw_yaml)
    except Exception as e:
        print(f"Warning: Failed to parse dig-hard-samples YAML: {e}")
        return []

    if not isinstance(data, list):
        print("Warning: dig-hard-samples output is not a list; skip normalization.")
        return []

    normalized: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        frame = item.get("frame")
        if frame is None:
            continue
        try:
            frame_int = int(frame)
        except Exception:
            continue
        normalized_item = dict(item)
        normalized_item["frame"] = frame_int
        normalized.append(normalized_item)

    return normalized


def save_dig_hard_output(inputs: TaskInputs, response_text: str) -> Optional[Path]:
    folder_path = inputs.folder_path
    if not folder_path:
        print("Warning: No folder_path for dig-hard-samples output; skip saving.")
        return None

    raw_yaml = _extract_yaml_block(response_text)
    normalized_items = _normalize_dig_hard_items(raw_yaml)

    results_dir = _ensure_results_dir(folder_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"digger_results_{timestamp}.yaml"

    if normalized_items:
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(normalized_items, f, allow_unicode=True)
        unique_frames: Set[int] = {item["frame"] for item in normalized_items if "frame" in item}
        print(f"Saved dig-hard-samples results to {output_file} ({len(unique_frames)} unique frames)")
    else:
        output_file.write_text(raw_yaml, encoding="utf-8")
        print(f"Saved raw dig-hard-samples output to {output_file} (could not normalize)")

    return output_file


TASK_OUTPUT_HANDLERS: Dict[str, TaskOutputHandler] = {
    "spotter": save_spotter_output,
    "dig-hard-samples": save_dig_hard_output,
}


def get_output_handler(task_name: str) -> Optional[TaskOutputHandler]:
    return TASK_OUTPUT_HANDLERS.get(task_name)



