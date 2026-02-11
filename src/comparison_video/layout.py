"""
Layout Manager for comparison video.

Calculates responsive sizes for all sub-panels based on the output resolution
and configurable ratios.
"""

from dataclasses import dataclass
from typing import Dict, Any

from .renderers.base import RenderRegion


@dataclass
class LayoutConfig:
    """Configuration for video layout proportions."""
    # Total output size
    output_width: int = 1920
    output_height: int = 1080
    
    # Left/right split
    left_ratio: float = 0.4  # Left panel takes 40% of width
    
    # Left panel vertical split (top: current frame, bottom: lyrics)
    left_top_ratio: float = 0.5  # Current frame takes 50% of left height
    
    # Right panel vertical split (top: original video, bottom: grids)
    right_top_ratio: float = 0.6  # Original video takes 60% of right height
    
    # Right bottom horizontal split (dig-hard vs spotter grids)
    right_bottom_left_ratio: float = 0.5  # dig-hard takes 50% of right-bottom width


class LayoutManager:
    """
    Manages layout calculations for the comparison video.
    
    Layout structure:
    +------------------+--------------------------------+
    |                  |                                |
    |   Current Frame  |       Original Video           |
    |   (left_top)     |       (right_top)              |
    |                  |                                |
    +------------------+--------------------------------+
    |                  |  dig-hard     |  spotter       |
    |   Lyrics Panel   |  strips       |  12-grid       |
    |   (left_bottom)  |  (rb_left)    |  (rb_right)    |
    +------------------+---------------+---------------+
    """
    
    def __init__(self, config: LayoutConfig = None):
        self.config = config or LayoutConfig()
        self._regions: Dict[str, RenderRegion] = {}
        self._calculate_regions()
    
    def _calculate_regions(self) -> None:
        """Calculate all panel regions based on config."""
        cfg = self.config
        
        # Main column widths
        left_width = int(cfg.output_width * cfg.left_ratio)
        right_width = cfg.output_width - left_width
        
        # Left panel heights
        left_top_height = int(cfg.output_height * cfg.left_top_ratio)
        left_bottom_height = cfg.output_height - left_top_height
        
        # Right panel heights
        right_top_height = int(cfg.output_height * cfg.right_top_ratio)
        right_bottom_height = cfg.output_height - right_top_height
        
        # Right bottom widths
        rb_left_width = int(right_width * cfg.right_bottom_left_ratio)
        rb_right_width = right_width - rb_left_width
        
        # Define all regions
        self._regions = {
            "current_frame": RenderRegion(
                x=0,
                y=0,
                width=left_width,
                height=left_top_height
            ),
            "lyrics_panel": RenderRegion(
                x=0,
                y=left_top_height,
                width=left_width,
                height=left_bottom_height
            ),
            "original_video": RenderRegion(
                x=left_width,
                y=0,
                width=right_width,
                height=right_top_height
            ),
            "strip_grid": RenderRegion(
                x=left_width,
                y=right_top_height,
                width=rb_left_width,
                height=right_bottom_height
            ),
            "spotter_grid": RenderRegion(
                x=left_width + rb_left_width,
                y=right_top_height,
                width=rb_right_width,
                height=right_bottom_height
            ),
        }
    
    def get_region(self, name: str) -> RenderRegion:
        """Get a named region."""
        if name not in self._regions:
            raise ValueError(f"Unknown region: {name}. Available: {list(self._regions.keys())}")
        return self._regions[name]
    
    def get_all_regions(self) -> Dict[str, RenderRegion]:
        """Get all regions."""
        return self._regions.copy()
    
    @property
    def output_size(self) -> tuple:
        """Return (width, height) of output video."""
        return (self.config.output_width, self.config.output_height)
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "LayoutManager":
        """
        Create LayoutManager from a configuration dictionary.
        
        Expected keys in config_dict (all optional):
        - output_width: int
        - output_height: int
        - left_ratio: float
        - left_top_ratio: float
        - right_top_ratio: float
        - right_bottom_left_ratio: float
        """
        layout_cfg = config_dict.get("layout", {})
        
        kwargs = {}
        for key in ["output_width", "output_height", "left_ratio", 
                    "left_top_ratio", "right_top_ratio", "right_bottom_left_ratio"]:
            if key in layout_cfg:
                kwargs[key] = layout_cfg[key]
        
        return cls(LayoutConfig(**kwargs))


