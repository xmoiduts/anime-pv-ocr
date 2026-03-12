from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from media_utils import get_expected_output_dir
from music_transcriber.config import LocalConfig


@dataclass(frozen=True)
class MediaJob:
    folder_name: str
    folder_path: Path
    media_paths: List[Path]
    output_dir: Path
    sort_mtime: float
    newest_media_path: Path


def _iter_supported_files(folder: Path, extensions: Iterable[str]) -> List[Path]:
    normalized_exts = {ext.lower() for ext in extensions}
    media_files = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_exts
    ]
    media_files.sort(key=lambda item: item.relative_to(folder).as_posix().lower())
    return media_files


def list_latest_media_jobs(
    media_source_folder: Path,
    latest_count: int,
    local_config: LocalConfig,
    project_root: Path,
) -> List[MediaJob]:
    if latest_count < 1:
        raise ValueError("--latest-medias must be at least 1.")
    if not media_source_folder.exists():
        raise FileNotFoundError(f"Media source folder does not exist: {media_source_folder}")
    if not media_source_folder.is_dir():
        raise NotADirectoryError(f"Media source folder is not a directory: {media_source_folder}")

    output_root = (project_root / local_config.output_root).resolve()
    jobs: List[MediaJob] = []
    candidates = [path for path in media_source_folder.iterdir() if path.is_dir()]
    ranked_candidates = []
    for folder in candidates:
        media_paths = _iter_supported_files(folder, local_config.supported_media_extensions)
        if not media_paths:
            continue
        newest_media_path = max(media_paths, key=lambda item: item.stat().st_mtime)
        sort_mtime = newest_media_path.stat().st_mtime
        output_dir = Path(get_expected_output_dir(folder.name, str(output_root)))
        ranked_candidates.append(
            MediaJob(
                folder_name=folder.name,
                folder_path=folder,
                media_paths=media_paths,
                output_dir=output_dir,
                sort_mtime=sort_mtime,
                newest_media_path=newest_media_path,
            )
        )

    ranked_candidates.sort(
        key=lambda job: (-job.sort_mtime, job.folder_name.lower())
    )
    for job in ranked_candidates:
        jobs.append(job)
        if len(jobs) >= latest_count:
            break

    return jobs
