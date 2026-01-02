import hashlib
import os
import re
from typing import List, Optional, Tuple


def get_expected_output_dir(filename: str, base_dir: str) -> str:
    """Compute deterministic output folder name from media filename."""
    basename = os.path.basename(filename)
    name_part = basename[:10]
    hash_part = hashlib.md5(basename.encode("utf-8")).hexdigest()[:8]
    name_part = "".join([c for c in name_part if c.isalnum() or c in (" ", ".", "_", "-")]).strip()
    return os.path.join(base_dir, f"{name_part}{hash_part}")


def find_target_folder(substring: str, outputs_dir: str, medias_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Locate the output folder and matching media file by fuzzy substring."""
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)

    media_matches = []
    if os.path.exists(medias_dir):
        files = [f for f in os.listdir(medias_dir) if os.path.isfile(os.path.join(medias_dir, f))]
        for f in files:
            if substring.lower() in f.lower():
                full_path = os.path.join(medias_dir, f)
                expected_dir = get_expected_output_dir(f, outputs_dir)
                media_matches.append((expected_dir, full_path))

    folder_matches = []
    if os.path.exists(outputs_dir):
        folders = [f for f in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, f))]
        for f in folders:
            if substring.lower() in f.lower():
                folder_matches.append((os.path.join(outputs_dir, f), None))

    all_matches_dict = {}
    for out_dir, media_path in media_matches + folder_matches:
        if out_dir not in all_matches_dict or (all_matches_dict[out_dir] is None and media_path is not None):
            all_matches_dict[out_dir] = media_path

    if not all_matches_dict:
        print(f"No folders or media files matching '{substring}' found.")
        return None, None

    unique_dirs = list(all_matches_dict.keys())

    if len(unique_dirs) == 1:
        target_dir = unique_dirs[0]
        return target_dir, all_matches_dict[target_dir]

    print(f"Multiple matches found for '{substring}':")
    for idx, folder in enumerate(unique_dirs):
        media_info = f" (Media: {os.path.basename(all_matches_dict[folder])})" if all_matches_dict[folder] else ""
        print(f"[{idx}] {os.path.basename(folder)}{media_info}")

    while True:
        try:
            choice = input("Enter the index of the folder to use: ")
            idx = int(choice)
            if 0 <= idx < len(unique_dirs):
                target_dir = unique_dirs[idx]
                return target_dir, all_matches_dict[target_dir]
            else:
                print("Invalid index.")
        except ValueError:
            print("Please enter a number.")


def parse_range(range_str: Optional[str]) -> Tuple[int, float]:
    if not range_str:
        return 1, float("inf")

    match = re.match(r"^(\d+)-(\d*)$", range_str)
    if not match:
        print(f"Invalid range format: {range_str}. Use 'start-end' or 'start-'.")
        return 1, float("inf")

    start = int(match.group(1))
    end_str = match.group(2)
    end = int(end_str) if end_str else float("inf")

    return start, end


def get_grid_files(folder_path: str, start_frame: int, end_frame: float, rows: int, cols: int) -> List[str]:
    preferred_dir_name = f"{cols}x{rows}_grids"
    preferred_path = os.path.join(folder_path, preferred_dir_name)
    cells_count = rows * cols

    if os.path.isdir(preferred_path):
        files = [f for f in os.listdir(preferred_path) if f.endswith(".jpg")]
        grid_files = []
        for f in files:
            match = re.match(r"(\d+)_(\d+)\.jpg", f)
            if match and int(match.group(1)) == cells_count:
                canvas_idx = int(match.group(2))
                grid_start = (canvas_idx - 1) * cells_count + 1
                grid_end = canvas_idx * cells_count
                if grid_start <= end_frame and grid_end >= start_frame:
                    grid_files.append((canvas_idx, os.path.join(preferred_path, f)))

        if grid_files:
            grid_files.sort()
            return [path for idx, path in grid_files]

    grid_dirs = [d for d in os.listdir(folder_path) if d.endswith("grids") and os.path.isdir(os.path.join(folder_path, d))]
    potential_files = []

    for d in grid_dirs:
        d_path = os.path.join(folder_path, d)
        for f in os.listdir(d_path):
            if not f.endswith(".jpg"):
                continue
            match = re.match(r"(\d+)_(\d+)\.jpg", f)
            if match:
                c_count = int(match.group(1))
                canvas_idx = int(match.group(2))
                grid_start = (canvas_idx - 1) * c_count + 1
                grid_end = canvas_idx * c_count
                if grid_start <= end_frame and grid_end >= start_frame:
                    potential_files.append(
                        {
                            "path": os.path.join(d_path, f),
                            "cells_count": c_count,
                            "canvas_idx": canvas_idx,
                            "is_preferred": c_count == cells_count,
                        }
                    )

    if not potential_files:
        print(f"No grid images found in {folder_path}")
        return []

    found_counts = sorted(list(set(f["cells_count"] for f in potential_files)), key=lambda x: (x != cells_count, x))
    best_count = found_counts[0]

    final_grids = [f for f in potential_files if f["cells_count"] == best_count]
    final_grids.sort(key=lambda x: x["canvas_idx"])

    return [f["path"] for f in final_grids]

