"""
Original Video Renderer.

Displays the original video playback at native frame rate.
"""

from typing import Optional

import cv2
import numpy as np

from .base import BaseRenderer, RenderRegion


class OriginalVideoRenderer(BaseRenderer):
    """
    Renders the original video panel.
    
    Reads frames directly from the video file and displays them.
    """
    
    def __init__(
        self,
        region: RenderRegion,
        media_path: str,
    ):
        """
        Initialize the original video renderer.
        
        Args:
            region: The region to render into.
            media_path: Path to the original video file.
        """
        super().__init__(region)
        self.media_path = media_path
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0
        self._total_frames: int = 0
        self._current_pos: int = 0  # Current position in video (next frame to read)
        self._last_frame_idx: int = -1
        self._last_frame: Optional[np.ndarray] = None
        
        self._open_video()
        self.set_placeholder("Video Unavailable")
    
    def _open_video(self) -> None:
        """Open the video capture."""
        self._cap = cv2.VideoCapture(self.media_path)
        if self._cap.isOpened():
            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._current_pos = 0  # Start at frame 0
        else:
            print(f"Warning: Could not open video: {self.media_path}")
    
    @property
    def fps(self) -> float:
        """Return the video frame rate."""
        return self._fps
    
    @property
    def total_frames(self) -> int:
        """Return the total number of frames."""
        return self._total_frames
    
    @property
    def duration(self) -> float:
        """Return the video duration in seconds."""
        return self._total_frames / self._fps if self._fps > 0 else 0
    
    def render(self, timestamp: float, frame_idx: int) -> np.ndarray:
        """Render the original video frame."""
        if self._cap is None or not self._cap.isOpened():
            return self.render_placeholder()
        
        # Use frame_idx directly for original video playback
        if frame_idx != self._last_frame_idx:
            # Optimization: use sequential read instead of seek when possible
            # This avoids expensive GOP seek operations, especially for VP9
            if frame_idx == self._current_pos:
                # Sequential case: just read next frame (fast)
                ret, frame = self._cap.read()
                if ret:
                    self._current_pos += 1
            elif frame_idx == self._current_pos - 1:
                # Same frame as last read - use cached
                ret = True
                frame = self._last_frame
            else:
                # Non-sequential: need to seek (slow but unavoidable)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                self._current_pos = frame_idx
                ret, frame = self._cap.read()
                if ret:
                    self._current_pos += 1
            
            if ret and frame is not None:
                self._last_frame = frame
                self._last_frame_idx = frame_idx
            elif self._last_frame is not None:
                # Use last valid frame if read fails
                pass
            else:
                return self.render_placeholder()
        
        if self._last_frame is None:
            return self.render_placeholder()
        
        # Fit frame to region
        output = self.fit_image_to_region(self._last_frame, self.region.width, self.region.height)
        
        # Draw timestamp overlay
        time_text = f"{timestamp:.2f}s / {self.duration:.2f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(time_text, font, font_scale, thickness)
        
        # Draw at bottom-right
        x = self.region.width - text_w - 15
        y = self.region.height - 15
        
        # Background
        cv2.rectangle(output, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(output, time_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        return output
    
    def close(self) -> None:
        """Release the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()

