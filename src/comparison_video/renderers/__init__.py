"""
Renderers for comparison video sub-panels.
"""

from .base import BaseRenderer
from .current_frame import CurrentFrameRenderer
from .lyrics_panel import LyricsPanelRenderer
from .original_video import OriginalVideoRenderer
from .strip_grid import StripGridRenderer
from .spotter_grid import SpotterGridRenderer

__all__ = [
    "BaseRenderer",
    "CurrentFrameRenderer",
    "LyricsPanelRenderer",
    "OriginalVideoRenderer",
    "StripGridRenderer",
    "SpotterGridRenderer",
]


