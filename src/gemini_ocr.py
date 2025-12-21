import cv2
import os
import re
import sys
import json
import yaml
import argparse
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
import google.genai as genai
import google.genai.types as types
# Monkey-patch Pydantic models to allow extra fields (thinking_effort etc.)
# because SDK version might trail behind API features.
for cls in [types.GenerateContentConfig, types.ThinkingConfig, types.Part]:
    if hasattr(cls, 'model_config'):
        try:
            # Create a new config dict if it doesn't exist or update existing
            if cls.model_config is None:
                cls.model_config = {'extra': 'allow'}
            else:
                cls.model_config['extra'] = 'allow'
            # If Pydantic v2, we might need to rebuild.
            if hasattr(cls, 'model_rebuild'):
                cls.model_rebuild(force=True)
        except Exception as e:
            # Just ignore if we can't patch; might fail later but worth a try
            pass

from dotenv import load_dotenv
from PIL import Image
from hard_samples import run_spotter_dig_hard_samples, imread_unicode, imwrite_unicode
from cli_utils import get_cli_args

def load_yaml_config(path="ocr-cli-config.yaml"):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_config(path="config.json"):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_expected_output_dir(filename, base_dir):
    """Derived from extract_frames.py logic."""
    basename = os.path.basename(filename)
    # Using first 10 chars of filename + 8 chars of MD5 hash
    name_part = basename[:10]
    hash_part = hashlib.md5(basename.encode("utf-8")).hexdigest()[:8]
    # Sanitize name_part for path safety
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

    # Combine
    # Use a dict to unique by output folder, keeping the media path if we have it
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

def parse_range(range_str):
    if not range_str:
        return 1, float('inf')
    
    match = re.match(r"^(\d+)-(\d*)$", range_str)
    if not match:
        print(f"Invalid range format: {range_str}. Use 'start-end' or 'start-'.")
        return 1, float('inf')
    
    start = int(match.group(1))
    end_str = match.group(2)
    end = int(end_str) if end_str else float('inf')
    
    return start, end

def get_grid_files(folder_path, start_frame, end_frame, rows, cols):
    preferred_dir_name = f"{cols}x{rows}_grids"
    preferred_path = os.path.join(folder_path, preferred_dir_name)
    cells_count = rows * cols
    
    # Priority 1: Use the specific NxM_grids folder if it exists and has matching files
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

    # Priority 2: Fallback to old behavior (search all *grids dirs)
    grid_dirs = [d for d in os.listdir(folder_path) if d.endswith("grids") and os.path.isdir(os.path.join(folder_path, d))]
    potential_files = []
    
    for d in grid_dirs:
        d_path = os.path.join(folder_path, d)
        for f in os.listdir(d_path):
            if not f.endswith(".jpg"): continue
            match = re.match(r"(\d+)_(\d+)\.jpg", f)
            if match:
                c_count = int(match.group(1))
                canvas_idx = int(match.group(2))
                grid_start = (canvas_idx - 1) * c_count + 1
                grid_end = canvas_idx * c_count
                if grid_start <= end_frame and grid_end >= start_frame:
                    potential_files.append({
                        "path": os.path.join(d_path, f),
                        "cells_count": c_count,
                        "canvas_idx": canvas_idx,
                        "is_preferred": (c_count == cells_count)
                    })

    if not potential_files:
        print(f"No grid images found in {folder_path}")
        return []

    # If we have multiple cells_counts, prefer the one from config
    # To keep it simple, we'll just pick the first cells_count that matches preferred, 
    # or the first one we find.
    found_counts = sorted(list(set(f["cells_count"] for f in potential_files)), 
                         key=lambda x: (x != cells_count, x))
    best_count = found_counts[0]
    
    final_grids = [f for f in potential_files if f["cells_count"] == best_count]
    final_grids.sort(key=lambda x: x["canvas_idx"])
    
    return [f["path"] for f in final_grids]

def call_gemini(api_key, model_name, prompt, image_paths, base_url=None, media_resolution=None, thinking_level=None, exchange_rate=7.2, gemini_generation=None):
    client_kwargs = {'api_key': api_key}
    if base_url:
        client_kwargs['http_options'] = {'base_url': base_url}
        print(f"Using custom base URL: {base_url}")
    else:
        print("Using default Google Gemini API endpoint.")

    client = genai.Client(**client_kwargs)
    
    contents = [prompt]
    
    # Check if we should use per-part resolution (Gemini 3+)
    is_gemini_3_plus = (gemini_generation is not None and gemini_generation >= 3)
    use_per_part = is_gemini_3_plus and media_resolution is not None

    print(f"Preparing to upload {len(image_paths)} images (Per-part resolution: {use_per_part})...")
    for path in image_paths:
        try:
            if use_per_part:
                # Use types.Part for per-part resolution setting
                with open(path, "rb") as f:
                    img_bytes = f.read()
                # Create part with specific media_resolution
                # We pass it to allow any value (like ULTRA_HIGH) to pass through to API
                part = types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/jpeg",
                    media_resolution={"level": media_resolution}
                )
                contents.append(part)
            else:
                img = Image.open(path)
                contents.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    
    print(f"Calling Gemini API ({model_name})...")
    usage = None
    full_text = ""
    try:
        # Build request config from YAML spotter config.
        # monkey-patched SDK above to allow extra fields like thinking_level
        request_config = {}
        
        # Only set global media_resolution if NOT using per-part (pre-Gemini 3)
        if media_resolution is not None and not use_per_part:
            request_config["media_resolution"] = media_resolution
        
        # Support both naming conventions from user/YAML
        t_level = thinking_level 
        if t_level is not None:
            request_config["thinking_level"] = t_level
            # Also inject thinking_effort just in case, if user provided one
            # request_config["thinking_effort"] = t_level 
            # (Better not duplicate unless needed, sticking to what user provided)

        if not request_config:
            request_config = None
        
        # Correct streaming usage for google-genai SDK:
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=request_config,
        ):
            if chunk.usage_metadata:
                usage = chunk.usage_metadata

            if chunk.candidates:
                for cand in chunk.candidates:
                    if cand.content and cand.content.parts:
                        for part in cand.content.parts:
                            if part.text:
                                print(part.text, end="", flush=True)
                                full_text += part.text
                            
                            # Handle thought_signature (e.g. from specific models/proxies)
                            if hasattr(part, 'thought_signature') and part.thought_signature:
                                sig = part.thought_signature
                                val = sig
                                if isinstance(sig, bytes):
                                    val = sig.hex()
                                elif hasattr(sig, 'hex'):
                                    val = sig.hex()
                                # Convert long signature to SHA1 hex as requested
                                sig_sha1 = hashlib.sha1(val.encode('utf-8')).hexdigest()
                                print(f"\n[Signature(SHA1): {sig_sha1}]", end="", flush=True)
        print("\n")

        if usage:
            prompt_tokens = usage.prompt_token_count or 0
            candidate_tokens = usage.candidates_token_count or 0
            
            # Simple pricing model (USD per 1M tokens)
            # Default to Flash 2.0/3.0 rates if not specific
            price_map = {
                "gemini-3-flash": (0.50, 3.00),
                "gemini-2.5-flash-lite": (0.10, 0.40),
            }
            
            # Find closest match or default
            input_price, output_price = 0.10, 0.40
            for k, v in price_map.items():
                if k in model_name.lower():
                    input_price, output_price = v
                    break
            
            cost_usd = (prompt_tokens / 1_000_000) * input_price + (candidate_tokens / 1_000_000) * output_price
            cost_rmb = cost_usd * exchange_rate
            
            # Print image token statistics if images were provided
            num_images = len(image_paths)
            if num_images > 0:
                avg_img_tokens = prompt_tokens / num_images
                try:
                    print(f"I {num_images} images | Avg ~{avg_img_tokens:.1f} tk/img (total prompt / image count)")
                except UnicodeEncodeError:
                    print(f"IMG {num_images} images | Avg ~{avg_img_tokens:.1f} tk/img (total prompt / image count)")

            print(f"^^ {prompt_tokens} tk  v {candidate_tokens} tk  $ {cost_usd:.4f} Y {cost_rmb:.4f}")
        
        return full_text

    except Exception as e:
        print(f"\nError calling Gemini API: {e}")
        return None

def find_context_cards():
    """Look for context-cards/*.yaml in current and parent directories."""
    current_path = Path.cwd()
    # Check current and 2 levels up
    for _ in range(3):
        card_dir = current_path / "context-cards"
        if card_dir.exists() and card_dir.is_dir():
            cards = list(card_dir.glob("*.yaml")) + list(card_dir.glob("*.yml"))
            if cards:
                return cards
        current_path = current_path.parent
    return []

def generate_grids(media_path, output_dir, rows, cols, target_fps, jpeg_quality=95):
    """Generate grid images if they don't exist."""
    cells_count = rows * cols
    grids_dir_name = f"{cols}x{rows}_grids"
    grids_dir = os.path.join(output_dir, grids_dir_name)
    os.makedirs(grids_dir, exist_ok=True)

    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {media_path}.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    expected_extracted = int(duration * target_fps)
    expected_grids = (expected_extracted + cells_count - 1) // cells_count

    # Check if all grids exist
    existing_grids = [f for f in os.listdir(grids_dir) if f.endswith(".jpg") and f.startswith(f"{cells_count}_")]
    if len(existing_grids) >= expected_grids:
        print(f"Grids already exist in {grids_dir}. Skipping generation.")
        cap.release()
        return grids_dir

    print(f"Generating grids in {grids_dir}...")
    frame_interval = max(1, int(round(fps / target_fps)))
    
    # Grid UI settings
    cell_w = 1000
    margin = 20
    text_h = 60
    
    # Calculate cell height based on 16:9 aspect ratio or original video ratio
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return None
    
    h_orig, w_orig = first_frame.shape[:2]
    cell_h = int(cell_w * (h_orig / w_orig)) + text_h
    
    canvas_w = cols * cell_w + (cols + 1) * margin
    canvas_h = rows * cell_h + (rows + 1) * margin
    
    img_area_w = cell_w
    img_area_h = cell_h - text_h

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    buffer = []
    frame_count = 0
    extracted_count = 0
    canvas_idx = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            buffer.append({
                "frame": frame,
                "index": extracted_count + 1,
                "timestamp": timestamp
            })
            extracted_count += 1
            
            if len(buffer) == cells_count:
                save_batch(buffer, canvas_idx, grids_dir, cells_count, rows, cols, 
                          canvas_w, canvas_h, cell_w, cell_h, img_area_w, img_area_h, 
                          margin, text_h, jpeg_quality)
                buffer = []
                canvas_idx += 1
        
        frame_count += 1
        
    if buffer:
        save_batch(buffer, canvas_idx, grids_dir, cells_count, rows, cols, 
                  canvas_w, canvas_h, cell_w, cell_h, img_area_w, img_area_h, 
                  margin, text_h, jpeg_quality)

    cap.release()
    return grids_dir

def save_batch(buffer, canvas_idx, output_dir, cells_count, rows, cols,
               canvas_w, canvas_h, cell_w, cell_h, img_w, img_h, 
               margin, text_h, jpeg_quality):
    
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    for i, item in enumerate(buffer):
        r = i // cols
        c = i % cols
        
        x_cell = margin + c * (cell_w + margin)
        y_cell = margin + r * (cell_h + margin)
        
        # Draw Text
        text = f"{item['index']} {item['timestamp']:.2f}s"
        cv2.putText(canvas, text, (x_cell, y_cell + text_h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Resize and Draw Image
        frame = item["frame"]
        h_f, w_f = frame.shape[:2]
        scale = min(img_w / w_f, img_h / h_f)
        new_w, new_h = int(w_f * scale), int(h_f * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        
        img_x = x_cell + (img_w - new_w) // 2
        img_y = y_cell + text_h + (img_h - new_h) // 2
        
        canvas[img_y:img_y+new_h, img_x:img_x+new_w] = resized
        cv2.rectangle(canvas, (img_x, img_y), (img_x+new_w, img_y+new_h), (0,0,0), 2)
    
    output_path = os.path.join(output_dir, f"{cells_count}_{canvas_idx}.jpg")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    is_success, im_buf = cv2.imencode(".jpg", canvas, encode_param)
    if is_success:
        im_buf.tofile(output_path)
    else:
        print(f"Error: Failed to encode {output_path}")

def main():
    cli_data = get_cli_args()
    args = cli_data["args"]
    task_name = cli_data["task_name"]
    task_config = cli_data["task_config"]
    yaml_config = cli_data["yaml_config"]
    base_url = cli_data["base_url"]
    config = cli_data["config_json"]
    
    fee_config = yaml_config.get("fee", {})
    exchange_rate = fee_config.get("exchange_rate", 7.2)
    
    # Resolution for media substring and optional timestamp suffix
    # Colon splitting logic removed as requested.
    media_substring = args.input_file or ""
    timestamp_suffix = args.suffix # Now passed via explicit argument
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file or environment variables.")
        sys.exit(1)
        
    # Get model configuration from yaml_config
    model_configs = yaml_config.get("model", {})
    model_info = model_configs.get(args.model, {})
    gemini_generation = model_info.get("gemini-generation")

    if args.hello:
        print(f"Running hello world test (Task: {task_name})")
        print(f"Model: {args.model} (Generation: {gemini_generation})")
        call_gemini(
            api_key, 
            args.model, 
            "Hi", 
            [], 
            base_url, 
            media_resolution=task_config.get("media_resolution"),
            thinking_level=task_config.get("thinking_level"),
            exchange_rate=exchange_rate,
            gemini_generation=gemini_generation,
        )
        sys.exit(0)
        
    config = load_config() # This is config.json for pathing/grids
    
    # Path configuration
    output_base_dir = task_config.get("output_dir") or config.get("output_base_dir", "outputs")
    media_base_dir = task_config.get("media_dir") or config.get("media_base_dir", "medias")
    
    # Preferred cells count - check YAML first, then config.json
    rows = task_config.get("grid_rows") or config.get("grid_rows", 4)
    cols = task_config.get("grid_cols") or config.get("grid_cols", 4)
    target_fps = task_config.get("target_fps") or config.get("target_fps", 6)
    preferred_cells_count = rows * cols

    folder_path = None
    image_paths = []

    # Handle adhoc-ask-clarity task differently
    if task_name == "adhoc-ask-clarity":
        folder_path = task_config.get("folder")
        if not folder_path:
            print("Error: 'folder' must be specified in config for adhoc-ask-clarity task.")
            sys.exit(1)
        
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist.")
            sys.exit(1)
            
        print(f"Ad-hoc task: Using images from folder '{folder_path}'")
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths.sort()
        
        if not image_paths:
            print(f"No images found in {folder_path}")
            sys.exit(0)
    else:
        # Input file is required for normal tasks
        if not args.input_file:
            print("Error: -i/--input-file is required for normal tasks.")
            sys.exit(1)

    # Parse media_substring and timestamp_suffix handled above
    # media_substring, timestamp_suffix were set before Resolve base_url

    folder_path, media_path = find_target_folder(media_substring, output_base_dir, media_base_dir)
    if not folder_path:
        sys.exit(1)

    # If media file is found, check if we need to generate grids
    # Skip grid generation for dig task if media_path exists, but run_spotter_dig_hard_samples will need it
    if task_name == "spotter-dig-hard-samples":
        if not media_path or not os.path.exists(media_path):
            print(f"Error: Media file not found for {folder_path}. Required for stripping.")
            sys.exit(1)
        
        image_paths = run_spotter_dig_hard_samples(folder_path, media_path, task_config, target_fps, timestamp_suffix)
        if not image_paths:
            sys.exit(0)
    else: # task: spotter
        if media_path and os.path.exists(media_path):
            generate_grids(media_path, folder_path, rows, cols, target_fps)
        
        start_frame, end_frame = parse_range(args.range)
        print(f"Selected folder: {folder_path} (Task: {task_name})")
        print(f"Frame range: {start_frame} to {'end' if end_frame == float('inf') else end_frame}")
        
        image_paths = get_grid_files(folder_path, start_frame, end_frame, rows, cols)
        if not image_paths:
            print("No images found in that range.")
            sys.exit(0)

    print(f"Found {len(image_paths)} images to process.")
    
    # Load prompt
    prompt_content = ""
    if os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_content = f.read()
    else:
        print(f"Warning: Prompt file {args.prompt_file} not found. Using basic prompt.")
        prompt_content = "Recognize the text in these images. Focus on lyrics and dialogue."

    # Add spotter_prompt if specified in config (useful for adhoc-ask-clarity)
    spotter_prompt_path = task_config.get("spotter_prompt_file")
    if spotter_prompt_path and os.path.exists(spotter_prompt_path):
        print(f"Adding spotter prompt from {spotter_prompt_path}")
        with open(spotter_prompt_path, "r", encoding="utf-8") as f:
            spotter_prompt_content = f.read()
            prompt_content += f"\n\n# Original Spotter Prompt\n{spotter_prompt_content}\n"

    # Add context cards if any
    context_cards = find_context_cards()
    if context_cards:
        print(f"Found {len(context_cards)} context cards. Adding to prompt.")
        prompt_content += "\n\n# Context Information (Background)\n"
        for card in context_cards:
            with open(card, "r", encoding="utf-8") as f:
                prompt_content += f"\n--- {card.name} ---\n{f.read()}\n"

    response_text = call_gemini(
        api_key,
        args.model,
        prompt_content,
        image_paths,
        base_url,
        media_resolution=task_config.get("media_resolution"),
        thinking_level=task_config.get("thinking_level"),
        exchange_rate=exchange_rate,
        gemini_generation=gemini_generation,
    )

    # Save response if it's a spotter task
    if response_text and task_name == "spotter" and folder_path:
        results_dir = os.path.join(folder_path, "spotter-results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Output timestamp in seconds as requested
        # Using %Y%m%d_%H%M%S format which is standard for filenames and includes seconds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(results_dir, f"{timestamp}.yaml")
        
        # Try to find YAML content in response
        yaml_content = response_text
        if "```yaml" in response_text:
            match = re.search(r"```yaml\n(.*?)\n```", response_text, re.DOTALL)
            if match:
                yaml_content = match.group(1)
        elif "```" in response_text:
            match = re.search(r"```\n(.*?)\n```", response_text, re.DOTALL)
            if match:
                yaml_content = match.group(1)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"Saved spotter results to {output_file}")

if __name__ == "__main__":
    main()

