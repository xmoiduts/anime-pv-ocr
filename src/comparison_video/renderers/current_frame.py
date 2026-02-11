"""
Current Frame Renderer.

Displays the OCR frame being processed at the current timestamp.
Uses fill-forward logic: shows the most recent OCR frame up to current time.
"""

from typing import Dict, Optional
import bisect

import cv2
import numpy as np

from .base import BaseRenderer, RenderRegion


class CurrentFrameRenderer(BaseRenderer):
    """
    Renders the current OCR frame panel.
    
    This panel shows the frame that was sent to OCR at or before the current timestamp.
    """
    
    def __init__(
        self,
        region: RenderRegion,
        frame_cache: Dict[int, np.ndarray],
        ocr_frame_timestamps: Dict[int, float],
        target_fps: float = 6.0,
    ):
        """
        Initialize the current frame renderer.
        
        Args:
            region: The region to render into.
            frame_cache: Dictionary mapping frame_id to BGR image.
            ocr_frame_timestamps: Dictionary mapping frame_id to timestamp.
            target_fps: Target FPS used for frame extraction.
        """
        super().__init__(region)
        self.frame_cache = frame_cache
        self.ocr_frame_timestamps = ocr_frame_timestamps
        self.target_fps = target_fps
        # Cache rendered panel by active OCR frame id.
        self._render_cache: Dict[int, np.ndarray] = {}
        
        # Sorted list of frame IDs for binary search
        self._sorted_frame_ids = sorted(ocr_frame_timestamps.keys())
        
        self.set_placeholder("No OCR Frame")
    
    def _get_current_frame_id(self, timestamp: float) -> Optional[int]:
        """
        Get the frame ID that should be displayed at the given timestamp.
        
        Uses fill-forward logic: returns the most recent frame ID whose
        timestamp is <= current timestamp.
        """
        if not self._sorted_frame_ids:
            return None

        # OCR frame ids are ordered by timestamp under fixed target_fps mapping.
        current_frame_id = int(timestamp * self.target_fps) + 1
        pos = bisect.bisect_right(self._sorted_frame_ids, current_frame_id) - 1
        if pos < 0:
            return None
        return self._sorted_frame_ids[pos]
    
    def render(self, timestamp: float, frame_idx: int) -> np.ndarray:
        """Render the current OCR frame."""
        frame_id = self._get_current_frame_id(timestamp)
        
        if frame_id is None or frame_id not in self.frame_cache:
            return self.render_placeholder()

        cached = self._render_cache.get(frame_id)
        if cached is not None:
            return cached
        
        frame = self.frame_cache[frame_id]
        
        # Fit frame to region
        output = self.fit_image_to_region(frame, self.region.width, self.region.height)
        
        # Draw frame info overlay
        frame_ts = self.ocr_frame_timestamps.get(frame_id, 0)
        info_text = f"Frame {frame_id} @ {frame_ts:.2f}s"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
        
        # Draw background for text
        cv2.rectangle(output, (5, 5), (text_w + 15, text_h + 15), (0, 0, 0), -1)
        cv2.putText(output, info_text, (10, text_h + 8), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        self._render_cache[frame_id] = output
        return output
    
    def preload_frames(self, media_path: str, frame_ids: list, target_fps: float) -> None:
        """
        Preload frames from media file into cache.
        
        Args:
            media_path: Path to video file.
            frame_ids: List of frame IDs to preload.
            target_fps: Target FPS for frame extraction.
        """
        import sys
        sys.path.insert(0, str(__file__).rsplit("comparison_video", 1)[0])
        
        try:
            from frame_extractor import FrameExtractor
        except ImportError:
            # Fallback to direct cv2 extraction
            cap = cv2.VideoCapture(media_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(round(fps / target_fps)))
            
            for frame_id in frame_ids:
                target_frame_idx = (frame_id - 1) * frame_interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = cap.read()
                if ret:
                    self.frame_cache[frame_id] = frame
                    self.ocr_frame_timestamps[frame_id] = target_frame_idx / fps
            
            cap.release()
            return
        
        extractor = FrameExtractor(media_path)
        frame_interval = max(1, int(round(extractor.fps / target_fps)))
        
        for frame_id in frame_ids:
            target_frame_idx = (frame_id - 1) * frame_interval
            extractor.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = extractor.cap.read()
            if ret:
                self.frame_cache[frame_id] = frame
                self.ocr_frame_timestamps[frame_id] = target_frame_idx / extractor.fps
        
        del extractor
        
        # Update sorted frame IDs
        self._sorted_frame_ids = sorted(self.ocr_frame_timestamps.keys())
        self._render_cache.clear()


