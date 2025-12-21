import os
import argparse
import yaml
from PIL import Image, ImageDraw, ImageFont
import hashlib
import re
import sys
import cv2
import numpy as np

def get_expected_output_dir(filename, base_dir):
    basename = os.path.basename(filename)
    name_part = basename[:10]
    hash_part = hashlib.md5(basename.encode("utf-8")).hexdigest()[:8]
    name_part = "".join([c for c in name_part if c.isalnum() or c in (' ', '.', '_', '-')]).strip()
    return os.path.join(base_dir, f"{name_part}{hash_part}")

def find_target_folder(substring, outputs_dir, medias_dir):
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Search files in medias_dir first to get the "correct" derived folder
    media_matches = []
    if os.path.exists(medias_dir):
        files = [f for f in os.listdir(medias_dir) if os.path.isfile(os.path.join(medias_dir, f))]
        for f in files:
            if substring.lower() in f.lower():
                full_path = os.path.join(medias_dir, f)
                expected_dir = get_expected_output_dir(f, outputs_dir)
                media_matches.append((expected_dir, full_path))
    
    # 2. Search existing folders in outputs_dir
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
    
    # Non-interactive mode fallback if we can't get input, or just pick first if choice is not provided
    # For a standalone script used in terminal, we can use input().
    while True:
        try:
            choice = input("Enter the index of the folder to use (or press enter for 0): ")
            if not choice.strip():
                idx = 0
            else:
                idx = int(choice)
            if 0 <= idx < len(unique_dirs):
                target_dir = unique_dirs[idx]
                return target_dir, all_matches_dict[target_dir]
            else:
                print("Invalid index.")
        except (ValueError, EOFError):
            print("Using index 0.")
            target_dir = unique_dirs[0]
            return target_dir, all_matches_dict[target_dir]

def extract_frame_from_video(media_path, frame_id, target_fps):
    if not media_path or not os.path.exists(media_path):
        print(f"Media file not found: {media_path}")
        return None

    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        print(f"Could not open video: {media_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Invalid FPS in video.")
        cap.release()
        return None

    frame_interval = max(1, int(round(fps / target_fps)))
    # extracted frame ID starts at 1
    target_frame_idx = (frame_id - 1) * frame_interval

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read frame {target_frame_idx} from {media_path}")
        return None

    # Convert BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def create_pyramid(frame_image, frame_id, timestamp, output_parent, grid_rows, grid_cols, levels=5, scale_factor=0.9, target_width=None):
    if frame_image is None:
        return

    w, h = frame_image.size

    # 2. Create base grid image
    base_w = w * grid_cols
    base_h = h * grid_rows
    
    if target_width:
        scale = target_width / base_w
        base_w = int(target_width)
        base_h = int(base_h * scale)
    
    full_tiled_w = w * grid_cols
    full_tiled_h = h * grid_rows
    full_tiled = Image.new("RGB", (full_tiled_w, full_tiled_h))
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            full_tiled.paste(frame_image, (c * w, r * h))
            
    if target_width:
        base_image = full_tiled.resize((base_w, base_h), Image.Resampling.LANCZOS)
    else:
        base_image = full_tiled

    # Output directory for pyramid
    pyramid_dir = os.path.join(output_parent, f"pyramid_{frame_id}_{grid_cols}x{grid_rows}")
    os.makedirs(pyramid_dir, exist_ok=True)

    print(f"Generating pyramid in {pyramid_dir}...")
    print(f"Base image size: {base_w}x{base_h}")

    # 3. Generate levels
    current_zoom = 1.0
    for i in range(levels):
        crop_w = int(base_w * current_zoom)
        crop_h = int(base_h * current_zoom)
        
        cropped = base_image.crop((0, 0, crop_w, crop_h))
        resized = cropped.resize((base_w, base_h), Image.Resampling.LANCZOS)
        
        draw = ImageDraw.Draw(resized)
        if timestamp:
            text = f"{frame_id}, {timestamp} || Zoom: {current_zoom:.2f}"
        else:
            text = f"Frame: {frame_id}, Zoom: {current_zoom:.2f}"
        
        font_size = max(20, base_h // 30)
        try:
            font_paths = ["arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:\\Windows\\Fonts\\arial.ttf"]
            font = None
            for p in font_paths:
                try:
                    font = ImageFont.truetype(p, size=font_size)
                    break
                except:
                    continue
            if not font:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            
        bbox = draw.textbbox((20, 20), text, font=font)
        draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=(0, 0, 0))
        draw.text((20, 20), text, fill=(255, 255, 255), font=font)
        
        output_path = os.path.join(pyramid_dir, f"level_{i}_zoom_{current_zoom:.2f}.jpg")
        resized.save(output_path, quality=95)
        print(f"Saved level {i} (zoom {current_zoom:.2f}) to {output_path}")
        
        current_zoom *= scale_factor

def main():
    # Set default values from config if available
    config_path = "ocr-cli-config.yaml"
    default_rows = 4
    default_cols = 3
    default_outputs = "outputs"
    default_medias = "medias"
    default_fps = 6
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            spotter_cfg = config.get("spotter", {})
            default_rows = spotter_cfg.get("grid_rows", default_rows)
            default_cols = spotter_cfg.get("grid_cols", default_cols)
            default_outputs = spotter_cfg.get("output_dir", default_outputs)
            default_medias = spotter_cfg.get("media_dir", default_medias)
            default_fps = spotter_cfg.get("target_fps", default_fps)

    parser = argparse.ArgumentParser(description="Generate image pyramid for a frame.")
    parser.add_argument("--frame", type=int, required=True, help="Frame ID to use.")
    parser.add_argument("--folder", type=str, required=True, help="Folder substring to find frame.")
    parser.add_argument("--rows", type=int, default=default_rows, help=f"Grid rows (default: {default_rows}).")
    parser.add_argument("--cols", type=int, default=default_cols, help=f"Grid columns (default: {default_cols}).")
    parser.add_argument("--levels", type=int, default=5, help="Number of pyramid levels (default: 5).")
    parser.add_argument("--scale", type=float, default=0.9, help="Scale factor between levels (default: 0.9).")
    parser.add_argument("--width", type=int, help="Target width for the output images (e.g. 3080).")
    parser.add_argument("--fps", type=float, default=default_fps, help=f"Target FPS used during extraction (default: {default_fps}).")
    
    args = parser.parse_args()
    
    target_dir, media_path = find_target_folder(args.folder, default_outputs, default_medias)
    if not target_dir:
        print(f"Could not find folder matching '{args.folder}'")
        return

    if not media_path:
        print(f"Could not find original media file for '{args.folder}'.")
        # Try to guess? find_target_folder already searches medias_dir.
        # If media_path is None, it means the folder exists in outputs but no matching media was found.
        return
        
    print(f"Using media file: {media_path}")
    print(f"Extracting frame {args.frame} (Target FPS: {args.fps})...")
    
    frame_image = extract_frame_from_video(media_path, args.frame, args.fps)
    if frame_image is None:
        return

    # Try to get timestamp from CSV if available for the overlay
    timestamp = None
    csv_path = os.path.join(target_dir, "timestamps.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        try:
                            if int(parts[0]) == args.frame:
                                timestamp = f"{parts[1]}s"
                                break
                        except ValueError:
                            continue
        except:
            pass
            
    create_pyramid(frame_image, args.frame, timestamp, target_dir, args.rows, args.cols, args.levels, args.scale, args.width)

if __name__ == "__main__":
    main()

