"""
Base renderer class for all comparison video sub-panels.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class RenderRegion:
    """Defines a rectangular region for rendering."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height


class BaseRenderer(ABC):
    """Abstract base class for all panel renderers."""
    
    def __init__(self, region: RenderRegion):
        """
        Initialize the renderer with its designated region.
        
        Args:
            region: The rectangular region this renderer is responsible for.
        """
        self.region = region
        self._placeholder_text: Optional[str] = None
    
    @abstractmethod
    def render(self, timestamp: float, frame_idx: int) -> np.ndarray:
        """
        Render the panel content for a given timestamp.
        
        Args:
            timestamp: Current video timestamp in seconds.
            frame_idx: Current video frame index.
            
        Returns:
            BGR image array of shape (height, width, 3).
        """
        pass
    
    def set_placeholder(self, text: str) -> None:
        """Set placeholder text to display when data is unavailable."""
        self._placeholder_text = text
    
    def render_placeholder(self) -> np.ndarray:
        """Render a placeholder panel when data is unavailable."""
        import cv2
        
        # Create white background
        canvas = np.ones((self.region.height, self.region.width, 3), dtype=np.uint8) * 255
        
        # Draw border
        cv2.rectangle(canvas, (0, 0), (self.region.width - 1, self.region.height - 1), (200, 200, 200), 2)
        
        # Draw placeholder text
        text = self._placeholder_text or "No Data"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (self.region.width - text_w) // 2
        text_y = (self.region.height + text_h) // 2
        
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (150, 150, 150), thickness, cv2.LINE_AA)
        
        return canvas
    
    @staticmethod
    def apply_fade(image: np.ndarray, opacity: float = 0.5, bg_color: tuple = (255, 255, 255)) -> np.ndarray:
        """
        Apply fade effect by blending with background color.
        
        Args:
            image: Input BGR image.
            opacity: How much of the original image to keep (0.0 = fully faded, 1.0 = no fade).
            bg_color: Background color to blend with (BGR).
            
        Returns:
            Faded image.
        """
        bg = np.full_like(image, bg_color, dtype=np.uint8)
        return cv2.addWeighted(image, opacity, bg, 1.0 - opacity, 0)
    
    @staticmethod
    def fit_image_to_region(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Resize image to fit within target dimensions while maintaining aspect ratio.
        Centers the image on a white background.
        
        Args:
            image: Input BGR image.
            target_width: Target width.
            target_height: Target height.
            
        Returns:
            Resized and centered image.
        """
        import cv2
        
        h, w = image.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center the image
        canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas

