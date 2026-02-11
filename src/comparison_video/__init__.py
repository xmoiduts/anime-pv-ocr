"""
Comparison Video Generator Module.

Generates side-by-side comparison videos showing the OCR pipeline results:
- Current OCR frame being processed
- Recognized lyrics text
- Original video playback
- dig-hard strips view
- spotter 12-grid view
"""

from .layout import LayoutManager
from .timeline import TimelineBuilder
from .compositor import FrameCompositor
from .ffmpeg_writer import FFmpegWriter

__all__ = [
    "LayoutManager",
    "TimelineBuilder",
    "FrameCompositor",
    "FFmpegWriter",
]


