import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from tqdm import tqdm
from grid_generator import generate_grids
from hard_samples import run_spotter_dig_hard_samples
from media_utils import find_target_folder, get_grid_files, parse_range, extract_audio


@dataclass
class TaskInputs:
    folder_path: Optional[str]
    media_path: Optional[str]
    image_paths: List[str]
    rows: int
    cols: int
    target_fps: float


def resolve_task_inputs(args, task_name: str, task_config: Dict, config_json: Dict) -> TaskInputs:
    output_base_dir = task_config.get("output_dir") or config_json.get("output_base_dir", "outputs")
    media_base_dir = task_config.get("media_dir") or config_json.get("media_base_dir", "medias")

    rows = task_config.get("grid_rows") or config_json.get("grid_rows", 4)
    cols = task_config.get("grid_cols") or config_json.get("grid_cols", 4)
    target_fps = task_config.get("target_fps") or config_json.get("target_fps", 6)

    media_substring = args.input_file or ""
    timestamp_suffix = args.suffix

    folder_path = None
    image_paths: List[str] = []

    if task_name == "adhoc-ask-clarity":
        folder_path = task_config.get("folder")
        if not folder_path:
            print("Error: 'folder' must be specified in config for adhoc-ask-clarity task.")
            sys.exit(1)

        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist.")
            sys.exit(1)

        print(f"Ad-hoc task: Using images from folder '{folder_path}'")
        image_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        image_paths.sort()

        if not image_paths:
            print(f"No images found in {folder_path}")
            sys.exit(0)

        return TaskInputs(folder_path, None, image_paths, rows, cols, target_fps)

    if not args.input_file:
        print("Error: -i/--input-file is required for normal tasks.")
        sys.exit(1)

    folder_path, media_path = find_target_folder(media_substring, output_base_dir, media_base_dir)
    if not folder_path:
        sys.exit(1)

    if task_name == "dig-hard-samples":
        if not media_path or not os.path.exists(media_path):
            print(f"Error: Media file not found for {folder_path}. Required for stripping.")
            sys.exit(1)

        image_paths = run_spotter_dig_hard_samples(folder_path, media_path, task_config, target_fps, timestamp_suffix)
        if not image_paths:
            sys.exit(0)

        return TaskInputs(folder_path, media_path, image_paths, rows, cols, target_fps)

    if task_name == "ocr-filtered":
        folder_path, media_path = find_target_folder(media_substring, output_base_dir, media_base_dir)
        if not folder_path:
            sys.exit(1)
            
        if not media_path or not os.path.exists(media_path):
             print(f"Error: Media file not found. Required for frame extraction.")
             sys.exit(1)
        
        # Merge frames from:
        # 1. Spotter results (required) - user selected frames
        # 2. Digger results (optional) - additional hard sample frames
        
        from hard_samples import find_latest_spotter_result, parse_selected_frames
        import yaml
        from frame_extractor import FrameExtractor
        import cv2
        
        results_dir = os.path.join(folder_path, "spotter-results")
        target_frames_set = set()
        
        # 1. Spotter results (primary source, required)
        spotter_yaml = find_latest_spotter_result(folder_path, timestamp_suffix)
        if spotter_yaml:
            print(f"Using spotter results: {spotter_yaml}")
            spotter_frames = parse_selected_frames(spotter_yaml)
            target_frames_set.update(spotter_frames)
            print(f"  -> {len(spotter_frames)} frames from spotter")
        else:
            print("No spotter results found. Please run spotter task first.")
            sys.exit(1)
        
        # 2. Digger results (optional, additive)
        digger_yaml = None
        if os.path.exists(results_dir):
            yamls = [f for f in os.listdir(results_dir) if f.startswith("digger_results_") and f.endswith(".yaml")]
            if yamls:
                yamls.sort()
                digger_yaml = os.path.join(results_dir, yamls[-1])
        
        if digger_yaml:
            print(f"Using digger results: {digger_yaml}")
            with open(digger_yaml, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data and isinstance(data, list):
                    digger_frames = set()
                    for item in data:
                        if "frame" in item:
                            digger_frames.add(int(item["frame"]))
                    target_frames_set.update(digger_frames)
                    print(f"  -> {len(digger_frames)} frames from digger")
        else:
            print("No digger results found. Using spotter results only.")
        
        target_frames = list(target_frames_set)
            
        if not target_frames:
            print("No frames found from spotter/digger results.")
            sys.exit(0)
            
        target_frames.sort()
        
        ocr_input_dir = os.path.join(folder_path, "ocr-input")
        os.makedirs(ocr_input_dir, exist_ok=True)
        
        image_paths = []
        
        # Extract audio if enabled (defaulting to True as per user request to add this feature)
        # We respect the config if present, but the user instruction implies this is a new default/feature.
        # Check explicit False to disable.
        if task_config.get("enable_audio", True):
            audio_path = os.path.join(ocr_input_dir, "audio.mp3")
            if extract_audio(media_path, audio_path):
                image_paths.append(audio_path)
                print(f"Included audio track: {audio_path}")

        print(f"Generating/Checking {len(target_frames)} frames for OCR...")
        
        extractor = FrameExtractor(media_path)
        frame_interval = max(1, int(round(extractor.fps / target_fps)))
        
        for frame_id in tqdm(target_frames, desc="Generating OCR frames"):
            target_frame_idx = (frame_id - 1) * frame_interval
            timestamp = target_frame_idx / extractor.fps
            filename = f"frame_{frame_id}_{timestamp:.2f}.jpg"
            filepath = os.path.join(ocr_input_dir, filename)
            
            if not os.path.exists(filepath):
                extractor.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = extractor.cap.read()
                if ret:
                    label = f"F:{frame_id} T:{timestamp:.2f}s"
                    frame = FrameExtractor.burn_in_info(frame, label)
                    FrameExtractor.imwrite_unicode(filepath, frame)
            
            image_paths.append(filepath)
            
        del extractor # Explicit cleanup
        
        print(f"Prepared {len(image_paths)} files in {ocr_input_dir}")
        return TaskInputs(folder_path, media_path, image_paths, rows, cols, target_fps)

    if media_path and os.path.exists(media_path):
        generate_grids(media_path, folder_path, rows, cols, target_fps)

    start_frame, end_frame = parse_range(args.range)
    print(f"Selected folder: {folder_path} (Task: {task_name})")
    print(f"Frame range: {start_frame} to {'end' if end_frame == float('inf') else end_frame}")

    image_paths = get_grid_files(folder_path, start_frame, end_frame, rows, cols)
    if not image_paths:
        print("No images found in that range.")
        sys.exit(0)

    return TaskInputs(folder_path, media_path, image_paths, rows, cols, target_fps)

