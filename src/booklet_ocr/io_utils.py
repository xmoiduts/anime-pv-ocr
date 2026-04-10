import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import yaml

YAML_FENCE_RE = re.compile(r"```(?:yaml|yml)\s*\n(.*?)\n```", re.DOTALL)
LRC_FENCE_RE = re.compile(r"```(?:lrc|LRC)\s*\n(.*?)\n```", re.DOTALL)
GENERIC_FENCE_RE = re.compile(r"```(?:[^\n`]*)\n(.*?)\n```", re.DOTALL)


@dataclass(frozen=True)
class SavedOutputs:
    output_dir: Path
    timestamp_prefix: str
    raw_log_path: Path
    yaml_path: Path
    lrc_path: Path


def _normalize_block(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def build_raw_log_text(raw_response: str, thought_text: str = "") -> str:
    if not thought_text or not thought_text.strip():
        return (raw_response or "").rstrip() + "\n"

    sections = [
        "===== Gemini Thought Summary =====",
        thought_text.rstrip(),
        "",
        "===== Gemini Response =====",
        (raw_response or "").rstrip(),
    ]
    return "\n".join(sections).rstrip() + "\n"


def _extract_yaml_from_generic_fences(response_text: str) -> Optional[str]:
    for match in GENERIC_FENCE_RE.finditer(response_text):
        candidate = _normalize_block(match.group(1))
        if not candidate.strip():
            continue
        try:
            parsed = yaml.safe_load(candidate)
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return candidate
    return None


def extract_yaml_text(response_text: str) -> str:
    if not response_text or not response_text.strip():
        raise ValueError("Gemini response is empty.")

    match = YAML_FENCE_RE.search(response_text)
    if match:
        normalized = _normalize_block(match.group(1))
        if normalized.strip():
            return normalized

    fallback = _extract_yaml_from_generic_fences(response_text)
    if fallback:
        return fallback

    raise ValueError("Could not extract a YAML block from the Gemini response.")


def extract_lrc_text(response_text: str) -> str:
    if not response_text or not response_text.strip():
        raise ValueError("Gemini response is empty.")

    match = LRC_FENCE_RE.search(response_text)
    if match:
        normalized = _normalize_block(match.group(1))
        if normalized.strip():
            return normalized

    generic_blocks = []
    for match in GENERIC_FENCE_RE.finditer(response_text):
        candidate = _normalize_block(match.group(1))
        if candidate.strip():
            generic_blocks.append(candidate)
    if len(generic_blocks) >= 2:
        return generic_blocks[1]

    raise ValueError("Could not extract an LRC block from the Gemini response.")


def create_output_dir(
    project_root: Path,
    output_root: str,
    now: Optional[datetime] = None,
    create: bool = True,
) -> tuple[Path, str]:
    timestamp = now or datetime.now()
    day_dir = (project_root / output_root / timestamp.strftime("%Y%m%d")).resolve()
    if create:
        day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir, timestamp.strftime("%Y%m%d-%H%M%S")


def save_outputs(
    output_dir: Path,
    timestamp_prefix: str,
    raw_response: str,
    raw_log_suffix: str,
    yaml_suffix: str,
    lrc_suffix: str,
    thought_text: str = "",
) -> SavedOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_log_path = output_dir / f"{timestamp_prefix}.{raw_log_suffix}"
    raw_log_path.write_text(build_raw_log_text(raw_response, thought_text), encoding="utf-8")

    try:
        yaml_text = extract_yaml_text(raw_response)
        lrc_text = extract_lrc_text(raw_response)
    except Exception as exc:
        raise ValueError(
            f"Saved raw response to {raw_log_path}, but failed to extract structured outputs: {exc}"
        ) from exc

    yaml_path = output_dir / f"{timestamp_prefix}.{yaml_suffix}"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    lrc_path = output_dir / f"{timestamp_prefix}.{lrc_suffix}"
    lrc_path.write_text(lrc_text, encoding="utf-8")

    return SavedOutputs(
        output_dir=output_dir,
        timestamp_prefix=timestamp_prefix,
        raw_log_path=raw_log_path,
        yaml_path=yaml_path,
        lrc_path=lrc_path,
    )


def build_song_title_placeholder_name(song_titles: Sequence[str], max_title_length: int = 24) -> str:
    """
    Reserved for a future schema-stable song list placeholder file.

    Note: Windows filenames cannot contain '<' or '>', so the eventual
    implementation must sanitize the requested placeholder naming shape.
    """

    truncated = [title.strip()[:max_title_length] for title in song_titles if title and title.strip()]
    if not truncated:
        return "song-titles.pending.txt"
    return " + ".join(truncated) + ".txt"
