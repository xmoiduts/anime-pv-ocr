from dataclasses import dataclass
from pathlib import Path

from music_transcriber.lrc_parser import extract_lrc_text


@dataclass(frozen=True)
class SavedOutputs:
    raw_log_path: Path
    lrc_path: Path


def save_outputs(output_dir: Path, raw_response: str, raw_log_filename: str, lrc_filename: str) -> SavedOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_log_path = output_dir / raw_log_filename
    raw_log_path.write_text(raw_response or "", encoding="utf-8")

    try:
        lrc_text = extract_lrc_text(raw_response)
    except Exception as exc:
        raise ValueError(
            f"Saved raw response to {raw_log_path}, but failed to extract LRC: {exc}"
        ) from exc

    lrc_path = output_dir / lrc_filename
    lrc_path.write_text(lrc_text, encoding="utf-8")

    return SavedOutputs(raw_log_path=raw_log_path, lrc_path=lrc_path)
