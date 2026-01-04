import os
import re
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm
from frame_extractor import FrameExtractor

def save_batch(
    buffer,
    canvas_idx,
    output_dir,
    cells_count,
    rows,
    cols,
    canvas_w,
    canvas_h,
    cell_w,
    cell_h,
    img_w,
    img_h,
    margin,
    text_h,
    jpeg_quality,
):
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for i, item in enumerate(buffer):
        r = i // cols
        c = i % cols

        x_cell = margin + c * (cell_w + margin)
        y_cell = margin + r * (cell_h + margin)

        text = f"{item['index']} {item['timestamp']:.2f}s"
        cv2.putText(
            canvas,
            text,
            (x_cell, y_cell + text_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        frame = item["frame"]
        h_f, w_f = frame.shape[:2]
        scale = min(img_w / w_f, img_h / h_f)
        new_w, new_h = int(w_f * scale), int(h_f * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        img_x = x_cell + (img_w - new_w) // 2
        img_y = y_cell + text_h + (img_h - new_h) // 2

        canvas[img_y : img_y + new_h, img_x : img_x + new_w] = resized
        cv2.rectangle(canvas, (img_x, img_y), (img_x + new_w, img_y + new_h), (0, 0, 0), 2)

    output_path = os.path.join(output_dir, f"{cells_count}_{canvas_idx}.jpg")
    FrameExtractor.imwrite_unicode(output_path, canvas, jpeg_quality)


def generate_grids(media_path: str, output_dir: str, rows: int, cols: int, target_fps: float, jpeg_quality: int = 95) -> Optional[str]:
    """Generate grid images if they don't exist."""
    cells_count = rows * cols
    grids_dir_name = f"{cols}x{rows}_grids"
    grids_dir = os.path.join(output_dir, grids_dir_name)
    os.makedirs(grids_dir, exist_ok=True)

    try:
        extractor = FrameExtractor(media_path)
    except ValueError as e:
        print(e)
        return None

    expected_extracted = int(extractor.duration * target_fps)
    expected_grids = (expected_extracted + cells_count - 1) // cells_count

    existing_grids = [f for f in os.listdir(grids_dir) if f.endswith(".jpg") and f.startswith(f"{cells_count}_")]
    if len(existing_grids) >= expected_grids:
        print(f"Grids already exist in {grids_dir}. Skipping generation.")
        del extractor
        return grids_dir

    print(f"Generating grids in {grids_dir}...")
    
    cell_w = 1000
    margin = 20
    text_h = 60
    
    extractor.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = extractor.cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        del extractor
        return None
        
    h_orig, w_orig = first_frame.shape[:2]
    cell_h = int(cell_w * (h_orig / w_orig)) + text_h

    canvas_w = cols * cell_w + (cols + 1) * margin
    canvas_h = rows * cell_h + (rows + 1) * margin

    img_area_w = cell_w
    img_area_h = cell_h - text_h
    
    buffer = []
    canvas_idx = 1
    
    # Use the generator
    frame_gen = extractor.extract_frames(target_fps=target_fps)
    
    with tqdm(total=expected_extracted, desc="Generating grids") as pbar:
        for frame_info in frame_gen:
            buffer.append({
                "frame": frame_info.frame,
                "index": frame_info.frame_id,
                "timestamp": frame_info.timestamp
            })
            
            if len(buffer) == cells_count:
                save_batch(
                    buffer,
                    canvas_idx,
                    grids_dir,
                    cells_count,
                    rows,
                    cols,
                    canvas_w,
                    canvas_h,
                    cell_w,
                    cell_h,
                    img_area_w,
                    img_area_h,
                    margin,
                    text_h,
                    jpeg_quality,
                )
                buffer = []
                canvas_idx += 1
            
            pbar.update(1)
            
    if buffer:
        save_batch(
            buffer,
            canvas_idx,
            grids_dir,
            cells_count,
            rows,
            cols,
            canvas_w,
            canvas_h,
            cell_w,
            cell_h,
            img_area_w,
            img_area_h,
            margin,
            text_h,
            jpeg_quality,
        )

    del extractor
    return grids_dir
