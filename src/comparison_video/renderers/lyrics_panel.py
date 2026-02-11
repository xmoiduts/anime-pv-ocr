"""
Lyrics Panel Renderer.

Displays the recognized lyrics text for the current OCR frame.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None

from .base import BaseRenderer, RenderRegion


class LyricsPanelRenderer(BaseRenderer):
    """
    Renders the lyrics panel showing recognized text.
    """
    
    def __init__(
        self,
        region: RenderRegion,
        lyrics_data: Dict[int, str],
        ocr_frame_timestamps: Dict[int, float],
        font_scale: float = 1.2,
        font_color: Tuple[int, int, int] = (0, 0, 0),
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Initialize the lyrics panel renderer.
        
        Args:
            region: The region to render into.
            lyrics_data: Dictionary mapping frame_id to lyrics text.
            ocr_frame_timestamps: Dictionary mapping frame_id to timestamp.
            font_scale: Font scale for lyrics text.
            font_color: Font color (BGR).
            bg_color: Background color (BGR).
        """
        super().__init__(region)
        self.lyrics_data = lyrics_data
        self.ocr_frame_timestamps = ocr_frame_timestamps
        self.font_scale = font_scale
        self.font_color = font_color
        self.bg_color = bg_color
        self._pil_font: Optional[object] = None
        self._fallback_to_cv2 = Image is None or ImageDraw is None or ImageFont is None
        # Cache rendered panel by active OCR frame id.
        self._render_cache: Dict[int, np.ndarray] = {}
        
        # Sorted list of frame IDs for lookup
        self._sorted_frame_ids = sorted(ocr_frame_timestamps.keys())
        
        self.set_placeholder("No Lyrics")
        self._init_cjk_font()

    def _init_cjk_font(self) -> None:
        """Initialize a CJK-capable font for Pillow rendering."""
        if self._fallback_to_cv2 or ImageFont is None:
            return
        font_size = max(16, int(24 * self.font_scale))
        font_candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/NotoSansCJK-Regular.ttc",
        ]
        for font_path in font_candidates:
            try:
                self._pil_font = ImageFont.truetype(font_path, font_size)
                return
            except Exception:
                continue

        # Fallback to default bitmap font (ASCII-limited).
        try:
            self._pil_font = ImageFont.load_default()
        except Exception:
            self._pil_font = None
            self._fallback_to_cv2 = True
    
    def _get_current_frame_id(self, timestamp: float) -> Optional[int]:
        """Get the frame ID for the current timestamp using fill-forward."""
        if not self._sorted_frame_ids:
            return None
        
        result = None
        for frame_id in self._sorted_frame_ids:
            frame_ts = self.ocr_frame_timestamps.get(frame_id, 0)
            if frame_ts <= timestamp:
                result = frame_id
            else:
                break
        return result
    
    def _wrap_text_cv2(self, text: str, max_width: int, font: int, font_scale: float, thickness: int) -> list:
        """Wrap text to fit within max_width (OpenCV fallback)."""
        words = list(text)  # Split into characters for CJK support
        lines = []
        current_line = ""
        
        for char in words:
            test_line = current_line + char
            (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char
        
        if current_line:
            lines.append(current_line)
        
        return lines

    def _wrap_text_pil(self, text: str, max_width: int, draw: object) -> list:
        """Wrap text to fit within max_width using Pillow font metrics."""
        if self._pil_font is None:
            return [text] if text else []

        chars = list(text)
        lines = []
        current_line = ""
        for ch in chars:
            test_line = current_line + ch
            bbox = draw.textbbox((0, 0), test_line, font=self._pil_font)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = ch
        if current_line:
            lines.append(current_line)
        return lines
    
    def render(self, timestamp: float, frame_idx: int) -> np.ndarray:
        """Render the lyrics panel."""
        # Get current frame and lyrics
        frame_id = self._get_current_frame_id(timestamp)
        
        if frame_id is None:
            return self.render_placeholder()

        cached = self._render_cache.get(frame_id)
        if cached is not None:
            return cached

        # Create background
        canvas = np.full((self.region.height, self.region.width, 3), self.bg_color, dtype=np.uint8)
        
        lyrics = self.lyrics_data.get(frame_id, "")
        
        if not lyrics:
            # Show frame info even without lyrics
            info_text = f"Frame {frame_id} - No lyrics"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(info_text, font, 0.6, 1)
            text_x = (self.region.width - text_w) // 2
            text_y = (self.region.height + text_h) // 2
            cv2.putText(canvas, info_text, (text_x, text_y), font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
            self._render_cache[frame_id] = canvas
            return canvas
        
        margin = 20
        max_text_width = self.region.width - 2 * margin
        font = cv2.FONT_HERSHEY_SIMPLEX

        if not self._fallback_to_cv2 and self._pil_font is not None and Image is not None and ImageDraw is not None:
            pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            lines = self._wrap_text_pil(lyrics, max_text_width, draw)
            line_height = int(self._pil_font.size * 1.4)
            total_height = len(lines) * line_height
            y = max(margin, (self.region.height - total_height) // 2)

            rgb_color = (self.font_color[2], self.font_color[1], self.font_color[0])
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=self._pil_font)
                text_w = bbox[2] - bbox[0]
                x = max(margin, (self.region.width - text_w) // 2)
                draw.text((x, y), line, font=self._pil_font, fill=rgb_color)
                y += line_height

            canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # OpenCV fallback when Pillow font is unavailable.
            thickness = 2
            lines = self._wrap_text_cv2(lyrics, max_text_width, font, self.font_scale, thickness)
            line_heights = []
            for line in lines:
                (_w, h), baseline = cv2.getTextSize(line, font, self.font_scale, thickness)
                line_heights.append(h + baseline + 10)

            total_height = sum(line_heights)
            y = (self.region.height - total_height) // 2 + line_heights[0] if line_heights else self.region.height // 2
            for i, line in enumerate(lines):
                (w, _h), _ = cv2.getTextSize(line, font, self.font_scale, thickness)
                x = (self.region.width - w) // 2
                cv2.putText(canvas, line, (x, y), font, self.font_scale, self.font_color, thickness, cv2.LINE_AA)
                y += line_heights[i] if i < len(line_heights) else 0
        
        # Draw frame info at bottom
        frame_ts = self.ocr_frame_timestamps.get(frame_id, 0)
        info_text = f"Frame {frame_id} @ {frame_ts:.2f}s"
        (info_w, info_h), _ = cv2.getTextSize(info_text, font, 0.5, 1)
        cv2.putText(canvas, info_text, (10, self.region.height - 10), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        self._render_cache[frame_id] = canvas
        return canvas
    
    def update_data(self, lyrics_data: Dict[int, str], timestamps: Dict[int, float]) -> None:
        """Update lyrics data and timestamps."""
        self.lyrics_data = lyrics_data
        self.ocr_frame_timestamps = timestamps
        self._sorted_frame_ids = sorted(timestamps.keys())
        self._render_cache.clear()


