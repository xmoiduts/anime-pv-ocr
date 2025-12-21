import os
import re
import yaml
import cv2
import numpy as np
from pathlib import Path

def imread_unicode(path):
    """Read an image from a path that may contain non-ASCII characters (Windows)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def imwrite_unicode(path, img, quality=95):
    """Write an image to a path that may contain non-ASCII characters (Windows)."""
    try:
        ext = os.path.splitext(path)[1]
        if not ext:
            ext = ".jpg"
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, im_buf = cv2.imencode(ext, img, encode_param)
        if success:
            im_buf.tofile(path)
            return True
    except Exception as e:
        print(f"Error writing image {path}: {e}")
    return False

def find_latest_spotter_result(folder_path, timestamp_suffix=None):
    results_dir = os.path.join(folder_path, "spotter-results")
    if not os.path.exists(results_dir):
        print(f"Error: spotter-results directory not found in {folder_path}")
        return None
    
    yaml_files = [f for f in os.listdir(results_dir) if f.endswith(".yaml") or f.endswith(".yml")]
    if not yaml_files:
        print(f"Error: No spotter result YAML files found in {results_dir}")
        return None
    
    if timestamp_suffix:
        matched_files = [f for f in yaml_files if Path(f).stem.endswith(timestamp_suffix)]
        if not matched_files:
            print(f"Error: No spotter result YAML files matching suffix '{timestamp_suffix}' found in {results_dir}")
            return None
        matched_files.sort()
        return os.path.join(results_dir, matched_files[-1])
    else:
        yaml_files.sort()
        return os.path.join(results_dir, yaml_files[-1])

def parse_selected_frames(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except Exception as e:
            print(f"Error parsing YAML {yaml_path}: {e}")
            return set()
    
    selected = set()
    if not data or not isinstance(data, list):
        return selected
    
    for item in data:
        if not item: continue
        if "frame" in item:
            f_val = int(item["frame"])
            selected.add(f_val)
            # if "neighboring_frames" in item:
            #     nf = item["neighboring_frames"]
            #     if isinstance(nf, list) and len(nf) == 2:
            #         for f in range(int(nf[0]), int(nf[1]) + 1):
            #             selected.add(f)
        
        # subtitle_selection is now ignored as requested
    
    return selected

def run_spotter_dig_hard_samples(folder_path, media_path, task_config, target_fps, timestamp_suffix=None):
    yaml_path = find_latest_spotter_result(folder_path, timestamp_suffix)
    if not yaml_path:
        return []
    
    print(f"Using spotter result: {yaml_path}")
    selected_frames = parse_selected_frames(yaml_path)
    
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {media_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    expected_extracted = int(duration * target_fps)
    
    # Calculate excluded frames based on avoid_before and avoid_after
    avoid_before = task_config.get("avoid_before", 3)
    avoid_after = task_config.get("avoid_after", 3)
    
    excluded_frames = set()
    for f in selected_frames:
        for offset in range(-avoid_before, avoid_after + 1):
            excluded_frames.add(f + offset)
            
    all_frames = list(range(1, expected_extracted + 1))
    hard_frames = [f for f in all_frames if f not in excluded_frames]
    
    if not hard_frames:
        print(f"No hard samples found (all frames were avoided around {len(selected_frames)} spotter selections).")
        cap.release()
        return []
    
    print(f"Found {len(hard_frames)} hard samples (excluded {len(excluded_frames)} frames around {len(selected_frames)} spotter selections).")
    
    stripping = task_config.get("stripping", 4)
    grid_rows = task_config.get("grid_rows", 4)
    image_width = task_config.get("image_width", 1080)
    
    hard_samples_dir = os.path.join(folder_path, "hard-samples")
    os.makedirs(hard_samples_dir, exist_ok=True)
    
    output_image_paths = []
    num_hard = len(hard_frames)
    strips_per_image = grid_rows
    num_output_images = (num_hard + strips_per_image - 1) // strips_per_image
    
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    strip_h = h_orig // stripping
    
    # UI settings similar to spotter grid
    text_h = 40
    margin = 10
    
    # Scale factors
    scale = image_width / w_orig
    final_strip_h = int(strip_h * scale)
    final_text_h = int(text_h) # Keep text height fixed or scale? 
    # In spotter it was fixed 60 for 1000px width.
    
    cell_h = final_text_h + final_strip_h
    canvas_w = image_width + 2 * margin
    canvas_h = grid_rows * cell_h + (grid_rows + 1) * margin
    
    frame_interval = max(1, int(round(fps / target_fps)))

    for i in range(num_output_images):
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        batch_start = i * strips_per_image
        batch_end = min((i + 1) * strips_per_image, num_hard)
        
        frames_in_this_image = []
        for idx_in_batch, j in enumerate(range(batch_start, batch_end)):
            frame_id = hard_frames[j]
            strip_idx = j % stripping # offset 0 -> part 0, offset 1 -> part 1...
            
            # Extract frame from video
            target_frame_idx = (frame_id - 1) * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_id} (idx {target_frame_idx})")
                continue
                
            # Take strip
            y_start = strip_idx * strip_h
            y_end = y_start + strip_h
            strip = frame[y_start:y_end, :]
            
            # Resize strip
            resized_strip = cv2.resize(strip, (image_width, final_strip_h))
            
            # Calculate position on canvas
            y_cell = margin + idx_in_batch * (cell_h + margin)
            
            # Draw Text (Title Row)
            timestamp = target_frame_idx / fps
            label = f"Frame: {frame_id} ({timestamp:.2f}s) | Strip: {strip_idx+1}/{stripping}"
            cv2.putText(canvas, label, (margin + 10, y_cell + final_text_h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Draw Image Strip
            img_y = y_cell + final_text_h
            canvas[img_y:img_y+final_strip_h, margin:margin+image_width] = resized_strip
            
            # Draw boundary
            cv2.rectangle(canvas, (margin, img_y), (margin+image_width, img_y+final_strip_h), (0,0,0), 2)
            
            frames_in_this_image.append(frame_id)
        
        if not frames_in_this_image:
            continue

        output_name = f"hard_{i+1:03d}_{frames_in_this_image[0]}-{frames_in_this_image[-1]}.jpg"
        output_path = os.path.join(hard_samples_dir, output_name)
        imwrite_unicode(output_path, canvas)
        output_image_paths.append(output_path)
        
    cap.release()
    return output_image_paths

