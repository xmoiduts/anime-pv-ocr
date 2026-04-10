from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image

from booklet_ocr.config import LocalConfig


@dataclass(frozen=True)
class InputListing:
    input_source_folder: Path
    media_paths: List[Path]
    pixel_counts: Dict[str, int]
    per_file_media_resolution: Dict[str, str]


def _iter_supported_files(folder: Path, extensions: Iterable[str]) -> List[Path]:
    normalized_exts = {ext.lower() for ext in extensions}
    media_files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in normalized_exts
    ]
    media_files.sort(key=lambda item: item.name.lower())
    return media_files


def get_image_pixel_count(path: Path) -> int:
    with Image.open(path) as image:
        width, height = image.size
    return int(width) * int(height)


def choose_media_resolution(pixel_count: int, local_config: LocalConfig) -> str:
    if pixel_count >= local_config.pixel_threshold_for_ultra_high:
        return local_config.large_image_media_resolution
    return local_config.small_image_media_resolution


def build_input_listing(input_source_folder: Path, local_config: LocalConfig) -> InputListing:
    if not input_source_folder.exists():
        raise FileNotFoundError(f"Input source folder does not exist: {input_source_folder}")
    if not input_source_folder.is_dir():
        raise NotADirectoryError(f"Input source folder is not a directory: {input_source_folder}")

    media_paths = _iter_supported_files(input_source_folder, local_config.supported_media_extensions)
    pixel_counts: Dict[str, int] = {}
    per_file_media_resolution: Dict[str, str] = {}
    for path in media_paths:
        pixel_count = get_image_pixel_count(path)
        pixel_counts[str(path)] = pixel_count
        per_file_media_resolution[str(path)] = choose_media_resolution(pixel_count, local_config)

    return InputListing(
        input_source_folder=input_source_folder,
        media_paths=media_paths,
        pixel_counts=pixel_counts,
        per_file_media_resolution=per_file_media_resolution,
    )
