"""
Glue workflow: Downloads → pending-OCR → booklet_ocr.main → finishing-OCR/Done.

Reads paths from its sibling config.yaml.  Run from anywhere — project root is
auto-detected from this file's location.

Usage:
    python ocr_workflow.py
    python ocr_workflow.py --minutes 5
    python ocr_workflow.py --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from booklet_ocr.config import (
    DEFAULT_LOCAL_CONFIG,
    PROJECT_ROOT,
    SRC_DIR,
    LocalConfig,
    load_local_config,
    resolve_project_path,
)

DOWNLOADS_DIR = Path.home() / "Downloads"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def download_activity_timestamp(path: Path) -> float:
    stat = path.stat()
    return max(stat.st_mtime, stat.st_ctime)


def list_download_files() -> list[tuple[Path, float]]:
    files: list[tuple[Path, float]] = []
    for path in DOWNLOADS_DIR.iterdir():
        if not path.is_file():
            continue
        try:
            activity_ts = download_activity_timestamp(path)
        except OSError:
            continue
        files.append((path, activity_ts))
    files.sort(key=lambda item: item[1], reverse=True)
    return files


def scan_recent_images(minutes: int) -> list[Path]:
    files = list_download_files()
    cutoff_ts = (datetime.now() - timedelta(minutes=minutes)).timestamp()
    images = [path for path, activity_ts in files if is_supported_image(path) and activity_ts >= cutoff_ts]
    if images or not files:
        return images

    latest_path, latest_ts = files[0]
    if not is_supported_image(latest_path):
        return []

    anchored_cutoff_ts = latest_ts - timedelta(minutes=minutes).total_seconds()
    return [
        path
        for path, activity_ts in files
        if is_supported_image(path) and anchored_cutoff_ts <= activity_ts <= latest_ts
    ]


def scan_pending_images(pending_ocr: Path, config: LocalConfig) -> list[Path]:
    if not pending_ocr.exists() or not pending_ocr.is_dir():
        return []

    supported_exts = {ext.lower() for ext in config.supported_media_extensions}
    images = [
        path
        for path in pending_ocr.iterdir()
        if path.is_file() and path.suffix.lower() in supported_exts
    ]
    images.sort(key=lambda path: path.name.lower())
    return images


def format_download_activity_time(path: Path) -> str:
    return datetime.fromtimestamp(download_activity_timestamp(path)).strftime("%Y-%m-%d %H:%M:%S")


def format_modified_time(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def display_images(images: list[Path]) -> None:
    print(f"\nFound {len(images)} image(s) in Downloads within the time window:\n")
    for i, path in enumerate(images, 1):
        size_str = format_size(path.stat().st_size)
        print(f"  {i:2d}. {path.name}")
        print(f"      Size: {size_str}  |  Detected: {format_download_activity_time(path)}  |  Modified: {format_modified_time(path)}")
    print()


def confirm_files(images: list[Path]) -> list[Path]:
    display_images(images)
    while True:
        choice = input("[A]ccept all  [S]elect individually  [Q]uit: ").strip().lower()
        if choice in ("a", ""):
            return list(images)
        elif choice == "s":
            selected: list[Path] = []
            for path in images:
                ans = input(f"  Include '{path.name}'? [Y/n]: ").strip().lower()
                if ans in ("y", "yes", ""):
                    selected.append(path)
            return selected
        elif choice == "q":
            print("Aborted.")
            return []
        else:
            print("Invalid choice. Enter A, S, or Q.")


def move_files(paths: list[Path], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for src in paths:
        dst = dest / src.name
        shutil.move(str(src), str(dst))
        print(f"  Moved: {src.name}")
        moved.append(dst)
    return moved


def run_booklet_ocr() -> int:
    print("\n── Running booklet OCR ──\n")
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    result = subprocess.run(
        [sys.executable, "-m", "booklet_ocr.main"],
        cwd=str(SRC_DIR),
        env=env,
    )
    return result.returncode


def post_ocr_move(processed_paths: list[Path], finishing_ocr: Path, done_dir: Path) -> None:
    label_f = finishing_ocr.name if finishing_ocr != done_dir else "finishing-OCR"
    label_d = done_dir.name
    print()
    while True:
        choice = input(
            f"Move processed images to [F]{label_f}, [D]{label_d}, or [L]eave in pending-OCR? (F/D/L): "
        ).strip().lower()
        if choice == "f":
            dest = finishing_ocr
            break
        elif choice == "d":
            dest = done_dir
            break
        elif choice == "l":
            print(f"  → Left {len(processed_paths)} file(s) in pending-OCR.")
            return
        else:
            print("Invalid choice. Enter F, D, or L.")
    move_files(processed_paths, dest)
    print(f"  → Moved {len(processed_paths)} file(s) to {dest.name}")


def parse_args(config: LocalConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR workflow glue: Downloads → pending-OCR → OCR → finishing/Done")
    parser.add_argument(
        "-m", "--minutes",
        type=int,
        default=config.default_lookback_minutes,
        help=f"Lookback window in minutes (default: {config.default_lookback_minutes})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without moving or running OCR")
    return parser.parse_args()


def main() -> int:
    config, _config_path = load_local_config()
    args = parse_args(config)

    # Resolve folders from config (they are relative to PROJECT_ROOT).
    pending_ocr = resolve_project_path(config.input_source_folder)
    finishing_ocr = resolve_project_path(config.finishing_ocr_folder) if config.finishing_ocr_folder else pending_ocr
    done_dir = resolve_project_path(config.done_folder) if config.done_folder else pending_ocr.parent / "Done"

    print("=" * 60)
    print("  Booklet OCR Workflow")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Config       : {DEFAULT_LOCAL_CONFIG}")
    print(f"  Lookback     : {args.minutes} minute(s)")
    print(f"  pending-OCR  : {pending_ocr}")
    print(f"  finishing-OCR: {finishing_ocr}")
    print(f"  Done         : {done_dir}")
    if args.dry_run:
        print("  *** DRY RUN — no files will be moved, no OCR will run ***")
    print("=" * 60)

    images = scan_recent_images(args.minutes)
    if not images:
        print("No recent image files found in Downloads.")
        pending_images = scan_pending_images(pending_ocr, config)
        if not pending_images:
            print("No existing image files found in pending-OCR.")
            return 0

        print(f"Found {len(pending_images)} existing image file(s) in pending-OCR.")
        print("Skipping Downloads selection and pre-moving; running booklet OCR directly.")
        if args.dry_run:
            print("[Dry run] Would run booklet OCR directly on existing pending-OCR files.")
            return 0

        rc = run_booklet_ocr()
        if rc != 0:
            print(f"\nbooklet_ocr.main exited with code {rc}. Files remain in pending-OCR.")
            return rc

        post_ocr_move(pending_images, finishing_ocr, done_dir)

        print("\n" + "=" * 60)
        print("  Workflow complete.")
        print("=" * 60)
        return 0

    if args.dry_run:
        display_images(images)
        print("[Dry run] Would ask for confirmation and move selected files to pending-OCR.")
        return 0

    selected = confirm_files(images)
    if not selected:
        return 0

    print(f"\nMoving {len(selected)} file(s) to pending-OCR...")
    moved_to_pending = move_files(selected, pending_ocr)

    rc = run_booklet_ocr()
    if rc != 0:
        print(f"\nbooklet_ocr.main exited with code {rc}. Files remain in pending-OCR.")
        return rc

    post_ocr_move(moved_to_pending, finishing_ocr, done_dir)

    print("\n" + "=" * 60)
    print("  Workflow complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
