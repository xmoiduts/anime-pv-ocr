"""
Timeline Builder for comparison video.

Maps video timestamps to the appropriate content for each panel:
- Which OCR frame to display
- What lyrics to show
- Which strip/grid page is active
- Which cells should be highlighted
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import yaml

from .state_logic import get_active_index_at_or_before


@dataclass
class OcrFrameInfo:
    """Information about a single OCR-processed frame."""
    frame_id: int
    timestamp: float
    lyric: str = ""
    note: Optional[str] = None


@dataclass
class SpotterFrameInfo:
    """Information about a spotter-selected frame."""
    frame_id: int
    source: str  # "spotter" or "digger"


@dataclass
class DiggerStripInfo:
    """Information about a dig-hard strip."""
    frame_id: int
    strip_idx: int
    description: str = ""


@dataclass
class HardSampleStripInfo:
    """Information about a reconstructed hard-sample strip."""
    frame_id: int
    strip_idx: int
    selected_by_digger: bool = False


@dataclass
class FrameState:
    """State of all panels at a specific video timestamp."""
    video_frame_idx: int
    timestamp: float
    
    # Current frame panel state
    current_ocr_frame_id: Optional[int] = None
    current_lyrics: str = ""
    
    # Strip grid state
    active_strip_page: int = 0
    strip_highlight_indices: List[int] = field(default_factory=list)
    
    # Spotter grid state
    active_spotter_page: int = 0
    spotter_highlight_indices: List[int] = field(default_factory=list)


class TimelineBuilder:
    """
    Builds a timeline mapping video timestamps to panel states.
    
    Loads data from:
    - spotter YAML: frames selected by spotter
    - digger YAML: frames selected by dig-hard-samples
    - ocr YAML: lyrics for each frame
    """
    
    def __init__(
        self,
        video_fps: float,
        video_duration: float,
        target_fps: float = 6.0,
        stripping: int = 5,
        digger_grid_rows: int = 5,
        avoid_before: int = 3,
        avoid_after: int = 3,
        spotter_grid_size: Tuple[int, int] = (4, 3),  # cols, rows
    ):
        """
        Initialize the timeline builder.
        
        Args:
            video_fps: Original video frame rate.
            video_duration: Video duration in seconds.
            target_fps: Target FPS used for frame extraction.
            stripping: Number of strips used in dig-hard-samples.
            digger_grid_rows: Number of strips displayed per dig-hard page.
            avoid_before: Frames before selected spotter frame to avoid.
            avoid_after: Frames after selected spotter frame to avoid.
            spotter_grid_size: (cols, rows) for spotter grid.
        """
        self.video_fps = video_fps
        self.video_duration = video_duration
        self.target_fps = target_fps
        self.stripping = stripping
        self.digger_grid_rows = digger_grid_rows
        self.avoid_before = avoid_before
        self.avoid_after = avoid_after
        self.spotter_cols, self.spotter_rows = spotter_grid_size
        self.spotter_cells_per_page = self.spotter_cols * self.spotter_rows
        self.total_sampled_frames = int(self.video_duration * self.target_fps)
        
        # Data containers
        self.ocr_frames: List[OcrFrameInfo] = []
        self.spotter_frames: List[SpotterFrameInfo] = []
        self.digger_strips: List[DiggerStripInfo] = []
        self.hard_sample_strips: List[HardSampleStripInfo] = []
        
        # Sets for quick lookup
        self.spotter_frame_ids: Set[int] = set()
        self.digger_frame_ids: Set[int] = set()
        self.all_selected_frame_ids: Set[int] = set()
        self.all_sampled_frame_ids: List[int] = list(range(1, self.total_sampled_frames + 1))
        
        # Precomputed timeline (lazy)
        self._timeline: Optional[List[FrameState]] = None
    
    def load_spotter_yaml(self, yaml_path: str) -> None:
        """Load spotter results from YAML."""
        if not os.path.exists(yaml_path):
            print(f"Warning: Spotter YAML not found: {yaml_path}")
            return
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, list):
            return
        
        for item in data:
            if not isinstance(item, dict):
                continue
            if "frame" in item:
                frame_id = int(item["frame"])
                self.spotter_frames.append(SpotterFrameInfo(frame_id=frame_id, source="spotter"))
                self.spotter_frame_ids.add(frame_id)
                self.all_selected_frame_ids.add(frame_id)
    
    def load_digger_yaml(self, yaml_path: str) -> None:
        """Load digger results from YAML."""
        if not os.path.exists(yaml_path):
            print(f"Warning: Digger YAML not found: {yaml_path}")
            return
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, list):
            return
        
        for item in data:
            if not isinstance(item, dict):
                continue
            if "frame" in item:
                frame_id = int(item["frame"])
                strip_idx = int(item.get("strip", 1)) - 1  # Convert to 0-based
                desc = item.get("description", "")
                
                self.digger_strips.append(DiggerStripInfo(
                    frame_id=frame_id,
                    strip_idx=strip_idx,
                    description=desc
                ))
                self.digger_frame_ids.add(frame_id)
                self.all_selected_frame_ids.add(frame_id)
                
                # Add to spotter frames if not already there
                if frame_id not in self.spotter_frame_ids:
                    self.spotter_frames.append(SpotterFrameInfo(frame_id=frame_id, source="digger"))

    def _rebuild_hard_sample_strips(self) -> None:
        """
        Rebuild hard-sample strips following `run_spotter_dig_hard_samples` logic.

        Logic parity:
        - Candidate frames are fixed-FPS sampled frame IDs [1..N]
        - Exclude [f-avoid_before, f+avoid_after] around each spotter-selected frame
        - Strip index cycles with candidate order (`j % stripping`)
        """
        excluded_frames: Set[int] = set()
        for frame_id in self.spotter_frame_ids:
            for offset in range(-self.avoid_before, self.avoid_after + 1):
                excluded = frame_id + offset
                if 1 <= excluded <= self.total_sampled_frames:
                    excluded_frames.add(excluded)

        hard_frames = [f for f in self.all_sampled_frame_ids if f not in excluded_frames]

        rebuilt: List[HardSampleStripInfo] = []
        for j, frame_id in enumerate(hard_frames):
            rebuilt.append(HardSampleStripInfo(
                frame_id=frame_id,
                strip_idx=j % max(1, self.stripping),
                selected_by_digger=frame_id in self.digger_frame_ids,
            ))

        self.hard_sample_strips = rebuilt
    
    def load_ocr_yaml(self, yaml_path: str) -> None:
        """Load OCR results from YAML."""
        if not os.path.exists(yaml_path):
            print(f"Warning: OCR YAML not found: {yaml_path}")
            return
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, list):
            return
        
        for item in data:
            if not isinstance(item, dict):
                continue
            
            frame_id = item.get("frame")
            if frame_id is None:
                continue
            
            self.ocr_frames.append(OcrFrameInfo(
                frame_id=int(frame_id),
                timestamp=float(item.get("timestamp", 0)),
                lyric=str(item.get("lyric", "")),
                note=item.get("note"),
            ))
        
        # Sort by frame_id
        self.ocr_frames.sort(key=lambda x: x.frame_id)
    
    def load_from_folder(self, folder_path: str) -> None:
        """
        Load all YAML files from a results folder.
        
        Args:
            folder_path: Path to the output folder containing spotter-results/.
        """
        results_dir = os.path.join(folder_path, "spotter-results")
        if not os.path.exists(results_dir):
            print(f"Warning: Results directory not found: {results_dir}")
            return
        
        # Find latest spotter YAML (not digger_results_*)
        spotter_yamls = [
            f for f in os.listdir(results_dir)
            if f.endswith(".yaml") and not f.startswith("digger_results_") and not f.startswith("ocr_results_")
        ]
        if spotter_yamls:
            spotter_yamls.sort()
            self.load_spotter_yaml(os.path.join(results_dir, spotter_yamls[-1]))
        
        # Find latest digger YAML
        digger_yamls = [
            f for f in os.listdir(results_dir)
            if f.startswith("digger_results_") and f.endswith(".yaml")
        ]
        if digger_yamls:
            digger_yamls.sort()
            self.load_digger_yaml(os.path.join(results_dir, digger_yamls[-1]))
        
        # Find latest OCR YAML
        ocr_yamls = [
            f for f in os.listdir(results_dir)
            if f.startswith("ocr_results_") and f.endswith(".yaml")
        ]
        if ocr_yamls:
            ocr_yamls.sort()
            self.load_ocr_yaml(os.path.join(results_dir, ocr_yamls[-1]))

        # Rebuild hard-sample strips after loading spotter + digger data.
        self._rebuild_hard_sample_strips()
    
    def _frame_id_to_timestamp(self, frame_id: int) -> float:
        """Convert frame_id (1-based) to timestamp in seconds."""
        return (frame_id - 1) / self.target_fps
    
    def _timestamp_to_video_frame(self, timestamp: float) -> int:
        """Convert timestamp to video frame index (0-based)."""
        return int(timestamp * self.video_fps)
    
    def _get_current_ocr_frame(self, timestamp: float) -> Optional[OcrFrameInfo]:
        """
        Get the OCR frame that should be displayed at the given timestamp.
        
        Uses fill-forward logic: displays the most recent OCR frame whose
        timestamp is <= current timestamp.
        """
        if not self.ocr_frames:
            return None
        
        result = None
        for frame_info in self.ocr_frames:
            frame_ts = self._frame_id_to_timestamp(frame_info.frame_id)
            if frame_ts <= timestamp:
                result = frame_info
            else:
                break
        return result
    
    def _get_active_strip_page_and_highlight(self, timestamp: float) -> Tuple[int, List[int]]:
        """
        Get the active strip page and highlighted strip indices.
        
        Returns:
            (page_index, list of highlighted strip indices within that page)
        """
        if not self.hard_sample_strips:
            return (0, [])
        
        sorted_strips = sorted(self.hard_sample_strips, key=lambda x: x.frame_id)
        
        current_frame_id = int(timestamp * self.target_fps) + 1
        visible_rows = max(1, self.digger_grid_rows)
        sorted_frame_ids = [s.frame_id for s in sorted_strips]

        # Sync with StripGridRenderer behavior:
        # use the nearest strip frame at/before current frame;
        # if current is before all strips, fall back to the first strip frame.
        active_frame_id: Optional[int] = None
        active_idx = get_active_index_at_or_before(sorted_frame_ids, current_frame_id, fallback_to_first=True)
        if active_idx is not None:
            active_frame_id = sorted_frame_ids[active_idx]

        page_idx = 0
        highlight_indices = []
        
        for i, strip in enumerate(sorted_strips):
            if strip.frame_id <= (active_frame_id or current_frame_id):
                page_idx = i // visible_rows
            if active_frame_id is not None and strip.frame_id == active_frame_id:
                highlight_indices.append(i % visible_rows)
        
        return (page_idx, highlight_indices)
    
    def _get_active_spotter_page_and_highlight(self, timestamp: float) -> Tuple[int, List[int]]:
        """
        Get the active spotter grid page and highlighted cell indices.
        
        Returns:
            (page_index, list of highlighted cell indices within that page)
        """
        if not self.all_sampled_frame_ids:
            return (0, [])
        
        current_frame_id = int(timestamp * self.target_fps) + 1
        idx = get_active_index_at_or_before(self.all_sampled_frame_ids, current_frame_id, fallback_to_first=True)
        if idx is None:
            return (0, [])
        page_idx = idx // self.spotter_cells_per_page
        return (page_idx, [idx % self.spotter_cells_per_page])
    
    def build_timeline(self) -> List[FrameState]:
        """
        Build the complete timeline for the video.
        
        Returns:
            List of FrameState objects, one per video frame.
        """
        if self._timeline is not None:
            return self._timeline
        
        total_video_frames = int(self.video_duration * self.video_fps)
        timeline = []
        
        for video_frame_idx in range(total_video_frames):
            timestamp = video_frame_idx / self.video_fps
            
            # Get current OCR frame
            ocr_frame = self._get_current_ocr_frame(timestamp)
            
            # Get strip grid state
            strip_page, strip_highlights = self._get_active_strip_page_and_highlight(timestamp)
            
            # Get spotter grid state
            spotter_page, spotter_highlights = self._get_active_spotter_page_and_highlight(timestamp)
            
            state = FrameState(
                video_frame_idx=video_frame_idx,
                timestamp=timestamp,
                current_ocr_frame_id=ocr_frame.frame_id if ocr_frame else None,
                current_lyrics=ocr_frame.lyric if ocr_frame else "",
                active_strip_page=strip_page,
                strip_highlight_indices=strip_highlights,
                active_spotter_page=spotter_page,
                spotter_highlight_indices=spotter_highlights,
            )
            timeline.append(state)
        
        self._timeline = timeline
        return timeline
    
    def get_frame_state(self, video_frame_idx: int) -> Optional[FrameState]:
        """Get the state for a specific video frame."""
        timeline = self.build_timeline()
        if 0 <= video_frame_idx < len(timeline):
            return timeline[video_frame_idx]
        return None
    
    def get_all_ocr_frame_ids(self) -> List[int]:
        """Get sorted list of all OCR frame IDs."""
        return sorted([f.frame_id for f in self.ocr_frames])
    
    def get_spotter_frame_source(self, frame_id: int) -> Optional[str]:
        """Get the source of a spotter frame ('spotter' or 'digger')."""
        if frame_id in self.spotter_frame_ids:
            return "spotter"
        if frame_id in self.digger_frame_ids:
            return "digger"
        return None

    def get_all_sampled_frame_ids(self) -> List[int]:
        """Get all fixed-FPS sampled frame IDs."""
        return list(self.all_sampled_frame_ids)

