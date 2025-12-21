import cv2
import json
import os
import hashlib
import numpy as np

def load_config(path="config.json"):
    if not os.path.exists(path):
        print(f"Config file {path} not found. Creating default.")
        # Default config content provided in previous step, but just in case
        return {}
        
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_output_dir(filename, base_dir):
    basename = os.path.basename(filename)
    # Using first 10 chars of filename + 8 chars of MD5 hash
    name_part = basename[:10]
    hash_part = hashlib.md5(basename.encode("utf-8")).hexdigest()[:8]
    # Sanitize name_part for path safety
    name_part = "".join([c for c in name_part if c.isalnum() or c in (' ', '.', '_', '-')]).strip()
    return os.path.join(base_dir, f"{name_part}{hash_part}")

def process_video(config):
    input_path = config.get("input_file")
    if not input_path or not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Invalid FPS in video.")
        return
        
    target_fps = config.get("target_fps", 6)
    # Calculate skip interval
    # We want 1 frame every (1/target_fps) seconds.
    # Video has 1 frame every (1/fps) seconds.
    # So we take every (fps / target_fps) frames.
    frame_interval = max(1, int(round(fps / target_fps)))
    
    print(f"Video FPS: {fps}. Target FPS: {target_fps}. Interval: {frame_interval} frames.")
    
    output_dir = get_output_dir(input_path, config.get("output_base_dir", "outputs"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Grid settings
    rows = config.get("grid_rows", 4)
    cols = config.get("grid_cols", 4)
    cells_count = rows * cols
    
    grids_dir = os.path.join(output_dir, f"{cells_count}grids")
    os.makedirs(grids_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Grids directory: {grids_dir}")
    
    csv_path = os.path.join(output_dir, "timestamps.csv")
    print(f"CSV path: {csv_path}")
    
    canvas_w = config.get("canvas_width", 4000)
    canvas_h = config.get("canvas_height", 3000)
    margin = config.get("margin", 20)
    text_h = config.get("text_area_height", 80)
    
    # Text settings
    font_scale = config.get("font_scale", 1.5)
    text_thickness = config.get("text_thickness", 2)
    text_color = tuple(config.get("text_color_bgr", [0, 0, 255]))
    jpeg_quality = config.get("jpeg_quality", 95)
    
    # Calculate cell size
    # Total width = (cols * cell_w) + ((cols + 1) * margin)
    cell_w = (canvas_w - (cols + 1) * margin) // cols
    cell_h = (canvas_h - (rows + 1) * margin) // rows
    
    # Image area inside cell (below text)
    # We reserve text_h for the text above the image
    img_area_w = cell_w
    img_area_h = cell_h - text_h
    
    if img_area_h <= 0:
        print("Error: Text area height is too large for the calculated cell height.")
        return
    
    buffer = [] # Stores (frame, frame_idx, timestamp)
    
    frame_count = 0
    extracted_count = 0
    canvas_idx = 1
    
    with open(csv_path, "w", encoding="utf-8") as csv_file:
        csv_file.write("index,timestamp_seconds\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                # Clone frame to avoid issues if we modify it (though we resize later)
                buffer.append({
                    "frame": frame,
                    "index": extracted_count + 1,
                    "timestamp": timestamp
                })
                
                csv_file.write(f"{extracted_count + 1},{timestamp:.2f}\n")
                extracted_count += 1
                
                if len(buffer) == rows * cols:
                    save_batch(buffer, config, grids_dir, canvas_idx, 
                              cells_count, cell_w, cell_h, img_area_w, img_area_h, margin, text_h,
                              font_scale, text_thickness, text_color, jpeg_quality)
                    buffer = []
                    canvas_idx += 1
            
            frame_count += 1
            
        # Flush remaining
        if buffer:
            save_batch(buffer, config, grids_dir, canvas_idx, 
                      cells_count, cell_w, cell_h, img_area_w, img_area_h, margin, text_h,
                      font_scale, text_thickness, text_color, jpeg_quality)

    cap.release()
    print(f"Done. Processed {frame_count} frames. Extracted {extracted_count} images.")

def save_batch(buffer, config, output_dir, canvas_idx, 
               cells_count, cell_w, cell_h, img_w, img_h, margin, text_h,
               font_scale, text_thickness, text_color, jpeg_quality):
               
    canvas = np.ones((config["canvas_height"], config["canvas_width"], 3), dtype=np.uint8) * 255 # White background
    
    cols = config["grid_cols"]
    
    for i, item in enumerate(buffer):
        r = i // cols
        c = i % cols
        
        # Calculate cell top-left
        x_cell = margin + c * (cell_w + margin)
        y_cell = margin + r * (cell_h + margin)
        
        # Draw Text
        text = f"{item['index']} {item['timestamp']:.2f}s"
        
        # Text position
        text_x = x_cell
        # Approximate baseline for text to sit nicely in the text area
        # text_h is total height reserved. Let's put text near bottom of that reserved area.
        text_y = y_cell + text_h - 15 
        
        cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, text_color, text_thickness, cv2.LINE_AA)
        
        # Resize and Draw Image
        frame = item["frame"]
        
        # Calculate scaling to FIT inside img_w x img_h
        h_f, w_f = frame.shape[:2]
        scale = min(img_w / w_f, img_h / h_f)
        new_w = int(w_f * scale)
        new_h = int(h_f * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Center in image area
        # Image area starts at y_cell + text_h
        area_start_y = y_cell + text_h
        
        img_x = x_cell + (img_w - new_w) // 2
        img_y = area_start_y + (img_h - new_h) // 2 
        
        canvas[img_y:img_y+new_h, img_x:img_x+new_w] = resized
        
        # Draw border around the image (not the whole cell, just the image frame)
        cv2.rectangle(canvas, (img_x, img_y), (img_x+new_w, img_y+new_h), (0,0,0), 2)
    
    output_path = os.path.join(output_dir, f"{cells_count}_{canvas_idx}.jpg")
    # cv2.imwrite doesn't support unicode paths on Windows well.
    # Using imencode + tofile workaround.
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    is_success, im_buf = cv2.imencode(".jpg", canvas, encode_param)
    if is_success:
        im_buf.tofile(output_path)
        print(f"Saved {output_path}")
    else:
        print(f"Error: Failed to encode image for {output_path}")

if __name__ == "__main__":
    cfg = load_config()
    process_video(cfg)

