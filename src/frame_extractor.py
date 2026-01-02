import os
import cv2
import numpy as np
from typing import List, Optional, Tuple, Generator
from dataclasses import dataclass

@dataclass
class FrameInfo:
    frame_id: int # Logical index in the extracted sequence (1-based)
    original_frame_index: int # Actual frame index in the video file
    timestamp: float
    frame: np.ndarray

class FrameExtractor:
    def __init__(self, media_path: str):
        self.media_path = media_path
        self.cap = cv2.VideoCapture(media_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video {media_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def extract_frames(
        self, 
        start_timestamp: float = 0.0,
        end_timestamp: float = float('inf'),
        target_fps: Optional[float] = None,
        base_timestamp: Optional[float] = None
    ) -> Generator[FrameInfo, None, None]:
        """
        Generator that yields frames based on specified parameters.
        
        Args:
            start_timestamp: Start time in seconds
            end_timestamp: End time in seconds
            target_fps: Desired FPS. If None, uses original video FPS.
            base_timestamp: If provided, frame intervals are calculated relative to this timestamp
                          instead of 0.0. Useful for expanding selections around a specific point.
        """
        if target_fps is None or target_fps <= 0:
            target_fps = self.fps

        frame_interval = max(1, int(round(self.fps / target_fps)))
        
        # Determine starting frame index
        if base_timestamp is not None:
            # Align to the grid defined by base_timestamp
            # We want to find the first frame >= start_timestamp that fits the interval relative to base_timestamp
            # t = base_timestamp + n * interval_sec
            interval_sec = frame_interval / self.fps
            
            # n_start = ceil((start_timestamp - base_timestamp) / interval_sec)
            n_start = int(np.ceil((start_timestamp - base_timestamp) / interval_sec))
            first_t = base_timestamp + n_start * interval_sec
            
            # Convert back to frame index
            current_frame_idx = int(round(first_t * self.fps))
        else:
            # Simple start from 0
            start_frame_idx = int(round(start_timestamp * self.fps))
            current_frame_idx = start_frame_idx
            
            # Align to interval grid if we assume grid starts at 0
            remainder = current_frame_idx % frame_interval
            if remainder != 0:
                current_frame_idx += (frame_interval - remainder)

        # Extraction loop
        extracted_count = 0
        
        while True:
            # Check bounds
            current_timestamp = current_frame_idx / self.fps
            if current_timestamp > end_timestamp or current_frame_idx >= self.total_frames:
                break
                
            if current_frame_idx < 0:
                current_frame_idx += frame_interval
                continue

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                break

            extracted_count += 1
            yield FrameInfo(
                frame_id=extracted_count, # This is relative to this extraction session
                original_frame_index=current_frame_idx,
                timestamp=current_timestamp,
                frame=frame
            )
            
            current_frame_idx += frame_interval

    @staticmethod
    def burn_in_info(frame: np.ndarray, text: str, font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Helper to draw text with background on frame."""
        h, w = frame.shape[:2]
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Background rectangle
        cv2.rectangle(frame, (0, 0), (text_w + 10, text_h + 20), (0, 0, 0), -1)
        
        # Text
        cv2.putText(frame, text, (5, text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return frame

    @staticmethod
    def imwrite_unicode(path: str, img: np.ndarray, quality: int = 95) -> bool:
        """Write an image to a path that may contain non-ASCII characters."""
        try:
            ext = os.path.splitext(path)[1]
            if not ext:
                ext = ".jpg"
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            success, im_buf = cv2.imencode(ext, img, encode_param)
            if success:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(path), exist_ok=True)
                im_buf.tofile(path)
                return True
        except Exception as e:
            print(f"Error writing image {path}: {e}")
        return False

