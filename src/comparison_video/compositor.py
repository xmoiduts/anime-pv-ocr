"""
Frame Compositor for comparison video.

Combines rendered outputs from all sub-panels into a single output frame.
"""

from typing import Dict, Optional

import cv2
import numpy as np

from .layout import LayoutManager
from .perf_trace import PerfTracer
from .renderers.base import BaseRenderer, RenderRegion


class FrameCompositor:
    """
    Composites rendered panels into a single output frame.
    """
    
    def __init__(self, layout: LayoutManager, tracer: Optional[PerfTracer] = None):
        """
        Initialize the compositor.
        
        Args:
            layout: LayoutManager instance defining panel positions.
        """
        self.layout = layout
        self.renderers: Dict[str, BaseRenderer] = {}
        self.tracer = tracer
        
        # Pre-allocate output buffer
        w, h = layout.output_size
        self._output_buffer = np.zeros((h, w, 3), dtype=np.uint8)
    
    def register_renderer(self, name: str, renderer: BaseRenderer) -> None:
        """
        Register a renderer for a named region.
        
        Args:
            name: Region name (must match layout region names).
            renderer: Renderer instance for this region.
        """
        # Validate that the region exists
        region = self.layout.get_region(name)
        if renderer.region != region:
            print(f"Warning: Renderer region doesn't match layout region for '{name}'")
        self.renderers[name] = renderer
    
    def composite_frame(self, timestamp: float, frame_idx: int) -> np.ndarray:
        """
        Composite a single output frame.
        
        Args:
            timestamp: Current video timestamp in seconds.
            frame_idx: Current video frame index.
            
        Returns:
            Composited BGR image of shape (height, width, 3).
        """
        tracer = self.tracer
        if tracer:
            with tracer.span("compositor.frame_total", frame_idx=frame_idx, args={"timestamp_s": timestamp}):
                return self._composite_frame_impl(timestamp, frame_idx)
        return self._composite_frame_impl(timestamp, frame_idx)

    def _composite_frame_impl(self, timestamp: float, frame_idx: int) -> np.ndarray:
        # Clear buffer to white
        self._output_buffer.fill(255)

        # Render and place each panel
        for name, renderer in self.renderers.items():
            tracer = self.tracer
            try:
                if tracer:
                    with tracer.span(f"renderer.{name}.render", frame_idx=frame_idx):
                        panel_image = renderer.render(timestamp, frame_idx)
                else:
                    panel_image = renderer.render(timestamp, frame_idx)

                region = renderer.region

                # Validate dimensions
                ph, pw = panel_image.shape[:2]
                if pw != region.width or ph != region.height:
                    if tracer:
                        with tracer.span(f"renderer.{name}.resize", frame_idx=frame_idx):
                            panel_image = cv2.resize(panel_image, (region.width, region.height))
                    else:
                        panel_image = cv2.resize(panel_image, (region.width, region.height))

                # Place panel in output buffer
                self._output_buffer[region.y:region.y2, region.x:region.x2] = panel_image

            except Exception as e:
                print(f"Warning: Failed to render '{name}': {e}")
                # Use placeholder
                placeholder = renderer.render_placeholder()
                region = renderer.region
                self._output_buffer[region.y:region.y2, region.x:region.x2] = placeholder

        # Draw panel borders
        tracer = self.tracer
        if tracer:
            with tracer.span("compositor.draw_borders", frame_idx=frame_idx):
                self._draw_borders()
            with tracer.span("compositor.copy_output", frame_idx=frame_idx):
                return self._output_buffer.copy()

        self._draw_borders()
        return self._output_buffer.copy()
    
    def _draw_borders(self) -> None:
        """Draw borders between panels."""
        border_color = (100, 100, 100)  # Gray
        border_thickness = 1
        
        regions = self.layout.get_all_regions()
        
        for name, region in regions.items():
            cv2.rectangle(
                self._output_buffer,
                (region.x, region.y),
                (region.x2 - 1, region.y2 - 1),
                border_color,
                border_thickness
            )
    
    def get_output_size(self) -> tuple:
        """Return (width, height) of output frames."""
        return self.layout.output_size


