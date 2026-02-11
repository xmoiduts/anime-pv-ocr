"""
Strip Grid Renderer.

Displays dig-hard sample strips with highlighting for selected frames.
Shows all strips with scrolling, highlighting the current time position.
"""

import queue
import threading
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .base import BaseRenderer, RenderRegion
from ..state_logic import get_active_index_at_or_before

if TYPE_CHECKING:
    from ..perf_trace import PerfTracer


class StripGridRenderer(BaseRenderer):
    """
    Renders the dig-hard strip grid panel.
    
    Shows horizontal strips extracted from video frames.
    - Unselected strips are faded (50% opacity with white background)
    - Active strip uses blue border if OCR-selected, gray border otherwise
    - Active strip is the nearest strip frame at/before current timeline frame
    """
    
    def __init__(
        self,
        region: RenderRegion,
        media_path: str,
        strip_frames: List[Dict],  # List of {frame_id, strip_idx, selected_by_digger}
        target_fps: float = 6.0,
        stripping: int = 5,
        visible_rows: int = 5,
        highlight_color: Tuple[int, int, int] = (255, 0, 0),  # Blue in BGR
        highlight_thickness: int = 3,
        fade_opacity: float = 0.5,
        enable_prefetch: bool = True,
        prefetch_depth: int = 2,
        tracer: Optional["PerfTracer"] = None,
    ):
        """
        Initialize the strip grid renderer.
        
        Args:
            region: The region to render into.
            media_path: Path to the original video file.
            strip_frames: List of reconstructed hard-sample strip dicts.
            target_fps: Target FPS used for frame extraction.
            stripping: Number of horizontal strips per frame.
            visible_rows: Number of strips shown per page.
            highlight_color: Color for OCR-selected active strip border (BGR).
            highlight_thickness: Thickness of highlight border.
            fade_opacity: Opacity for non-selected strips (0-1).
        """
        super().__init__(region)
        self.media_path = media_path
        self.strip_frames = strip_frames
        self.target_fps = target_fps
        self.stripping = stripping
        self.visible_rows = max(1, visible_rows)
        self.highlight_color = highlight_color
        self.highlight_thickness = highlight_thickness
        self.fade_opacity = fade_opacity
        self.enable_prefetch = enable_prefetch
        self.prefetch_depth = max(0, prefetch_depth)
        self.tracer = tracer
        # Empirical threshold: sparse page windows prefer pure random seek.
        self._prefetch_seek_only_density_threshold = 20.0
        
        # Video properties
        self._cap: Optional[cv2.VideoCapture] = None
        self._prefetch_cap: Optional[cv2.VideoCapture] = None
        self._video_fps: float = 0
        self._video_height: int = 0
        self._video_width: int = 0

        # Shared cache lock for strip/page caches.
        self._cache_lock = threading.RLock()
        
        # Cache for rendered strips
        self._strip_cache: Dict[Tuple[int, int], np.ndarray] = {}
        
        # Selected frame IDs set (selected by digger to send OCR)
        self._selected_frame_ids = {
            d["frame_id"]
            for d in strip_frames
            if "frame_id" in d and d.get("selected_by_digger", False)
        }
        
        # Sorted frame list for display
        self._sorted_digger = sorted(strip_frames, key=lambda x: x.get("frame_id", 0))
        self._sorted_frame_ids = [d.get("frame_id", 0) for d in self._sorted_digger]
        self._frame_to_index = {frame_id: i for i, frame_id in enumerate(self._sorted_frame_ids)}
        self._inactive_highlight_color = (160, 160, 160)  # Gray

        self._num_visible_strips = min(self.visible_rows, len(self._sorted_digger))
        self._strip_display_height = max(1, self.region.height // max(1, self._num_visible_strips))
        self._strip_display_width = self.region.width - 10  # Small margin
        self._total_pages = (
            (len(self._sorted_digger) + self._num_visible_strips - 1) // self._num_visible_strips
            if self._num_visible_strips > 0
            else 0
        )

        # Static page caches (without current highlight)
        self._page_cache: Dict[int, np.ndarray] = {}
        # page_idx -> frame_id -> (x, y, width, height, is_selected)
        self._page_rows: Dict[int, Dict[int, Tuple[int, int, int, int, bool]]] = {}

        # Prefetch worker state
        self._prefetch_stop = threading.Event()
        self._prefetch_queue: "queue.Queue[int]" = queue.Queue()
        self._prefetch_pending: set[int] = set()
        self._prefetch_pending_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        
        self._open_video()
        self._open_prefetch_video()
        self._start_prefetch_worker()
        self.set_placeholder("No dig-hard Data")
    
    def _open_video(self) -> None:
        """Open video for strip extraction."""
        self._cap = cv2.VideoCapture(self.media_path)
        if self._cap.isOpened():
            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._video_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._video_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _open_prefetch_video(self) -> None:
        """Open dedicated capture for prefetch worker."""
        if not self.enable_prefetch or self.prefetch_depth <= 0:
            return
        cap = cv2.VideoCapture(self.media_path)
        if cap.isOpened():
            self._prefetch_cap = cap
        else:
            cap.release()
            self.enable_prefetch = False

    def _start_prefetch_worker(self) -> None:
        """Start background worker for future page prefetch."""
        if not self.enable_prefetch or self.prefetch_depth <= 0 or self._prefetch_cap is None:
            return
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop,
            name="strip-grid-prefetch",
            daemon=True,
        )
        self._prefetch_thread.start()

    def _prefetch_loop(self) -> None:
        """Worker thread: build requested pages with dedicated capture."""
        while not self._prefetch_stop.is_set():
            try:
                page_idx = self._prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if self.tracer:
                    with self.tracer.span(
                        "renderer.strip_grid.prefetch_page",
                        args={"page_idx": page_idx, "mode": "prefetch"},
                    ):
                        self._prefetch_page_strips_sequential(page_idx=page_idx, cap=self._prefetch_cap)
                        self._get_or_build_page_canvas(page_idx, cap=self._prefetch_cap)
                else:
                    self._prefetch_page_strips_sequential(page_idx=page_idx, cap=self._prefetch_cap)
                    self._get_or_build_page_canvas(page_idx, cap=self._prefetch_cap)
            finally:
                with self._prefetch_pending_lock:
                    self._prefetch_pending.discard(page_idx)

    def _schedule_prefetch(self, page_idx: int) -> None:
        """Schedule future page build if not cached/inflight."""
        if not self.enable_prefetch or self.prefetch_depth <= 0:
            return
        if page_idx < 0 or page_idx >= self._total_pages:
            return
        with self._cache_lock:
            if page_idx in self._page_cache:
                return
        with self._prefetch_pending_lock:
            if page_idx in self._prefetch_pending:
                return
            self._prefetch_pending.add(page_idx)
        self._prefetch_queue.put(page_idx)

    def _trace_mode_for_cap(self, cap: Optional[cv2.VideoCapture]) -> str:
        """Return trace mode label for a capture object."""
        return "prefetch" if cap is self._prefetch_cap else "render"

    def _trace_span(self, name: str, args: Dict[str, object]):
        """Return tracer span context or no-op context."""
        if self.tracer:
            return self.tracer.span(name, args=args)
        return nullcontext()

    def _page_entries(self, page_idx: int) -> List[Dict]:
        """Return strip entries shown on a page."""
        page_start = page_idx * self._num_visible_strips
        page_end = min(page_start + self._num_visible_strips, len(self._sorted_digger))
        return [self._sorted_digger[idx] for idx in range(page_start, page_end)]

    def _prefetch_page_strips_sequential(
        self,
        page_idx: int,
        cap: Optional[cv2.VideoCapture],
    ) -> None:
        """
        Prefetch missing strip rows via one seek + sequential reads.

        Uses page-first random seek and sequential scan to reduce repeated seeks.
        """
        if cap is None or not cap.isOpened():
            return
        entries = self._page_entries(page_idx)
        if not entries:
            return

        frame_interval = max(1, int(round(self._video_fps / self.target_fps)))
        target_items: List[Tuple[int, int, int]] = []  # (video_idx, frame_id, strip_idx)
        with self._cache_lock:
            for entry in entries:
                frame_id = entry.get("frame_id", 0)
                strip_idx = entry.get("strip_idx", 0)
                cache_key = (frame_id, strip_idx)
                if cache_key in self._strip_cache:
                    continue
                video_idx = (frame_id - 1) * frame_interval
                target_items.append((video_idx, frame_id, strip_idx))

        if not target_items:
            return

        target_items.sort(key=lambda x: x[0])
        first_idx = target_items[0][0]
        last_idx = target_items[-1][0]
        target_by_video_idx = {video_idx: (frame_id, strip_idx) for video_idx, frame_id, strip_idx in target_items}
        requested = len(target_items)
        hits = 0
        strip_height = self._video_height // self.stripping

        window_frames = max(0, last_idx - first_idx + 1)
        density = (window_frames / requested) if requested > 0 else float("inf")
        use_seek_only = density >= self._prefetch_seek_only_density_threshold

        with self._trace_span(
            "renderer.strip_grid.prefetch_strategy",
            {
                "page_idx": page_idx,
                "mode": "prefetch",
                "requested": requested,
                "window_frames": window_frames,
                "window_per_strip": density,
                "threshold": self._prefetch_seek_only_density_threshold,
                "seek_only": use_seek_only,
            },
        ):
            pass

        if use_seek_only:
            with self._trace_span(
                "renderer.strip_grid.prefetch_seek_only_window",
                {
                    "page_idx": page_idx,
                    "mode": "prefetch",
                    "requested": requested,
                    "first_video_idx": first_idx,
                    "last_video_idx": last_idx,
                    "window_frames": window_frames,
                },
            ):
                for _video_idx, frame_id, strip_idx in target_items:
                    strip = self._get_strip(frame_id, strip_idx, cap=cap)
                    if strip is not None:
                        hits += 1

                with self._trace_span(
                    "renderer.strip_grid.prefetch_seek_only_result",
                    {
                        "page_idx": page_idx,
                        "mode": "prefetch",
                        "requested": requested,
                        "hits": hits,
                        "misses": max(0, requested - hits),
                    },
                ):
                    pass
            return

        with self._trace_span(
            "renderer.strip_grid.prefetch_seq_window",
            {
                "page_idx": page_idx,
                "mode": "prefetch",
                "requested": requested,
                "first_video_idx": first_idx,
                "last_video_idx": last_idx,
                "window_frames": window_frames,
                "window_per_strip": density,
            },
        ):
            with self._trace_span(
                "renderer.strip_grid.prefetch_seq_seek",
                {"page_idx": page_idx, "mode": "prefetch", "seek_to": first_idx},
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)

            current_idx = first_idx
            while current_idx <= last_idx and hits < requested:
                ret, frame = cap.read()
                if not ret:
                    break

                target = target_by_video_idx.get(current_idx)
                if target is not None:
                    frame_id, strip_idx = target
                    cache_key = (frame_id, strip_idx)
                    y_start = strip_idx * strip_height
                    y_end = y_start + strip_height
                    strip = frame[y_start:y_end, :]
                    with self._cache_lock:
                        if cache_key not in self._strip_cache:
                            self._strip_cache[cache_key] = strip.copy()
                    hits += 1

                current_idx += 1

            with self._trace_span(
                "renderer.strip_grid.prefetch_seq_result",
                {
                    "page_idx": page_idx,
                    "mode": "prefetch",
                    "requested": requested,
                    "hits": hits,
                    "misses": max(0, requested - hits),
                },
            ):
                pass

    def _get_strip(
        self,
        frame_id: int,
        strip_idx: int,
        cap: Optional[cv2.VideoCapture] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract a horizontal strip from a video frame.
        
        Args:
            frame_id: The frame ID (1-based).
            strip_idx: The strip index (0-based).
            
        Returns:
            The extracted strip as BGR image, or None if extraction fails.
        """
        cache_key = (frame_id, strip_idx)
        with self._cache_lock:
            cached = self._strip_cache.get(cache_key)
        if cached is not None:
            return cached
        
        local_cap = cap or self._cap
        if local_cap is None or not local_cap.isOpened():
            return None
        
        # Calculate video frame index
        frame_interval = max(1, int(round(self._video_fps / self.target_fps)))
        video_frame_idx = (frame_id - 1) * frame_interval
        mode = self._trace_mode_for_cap(local_cap)

        with self._trace_span(
            "renderer.strip_grid.decode_frame",
            {"frame_id": frame_id, "video_frame_idx": video_frame_idx, "mode": mode},
        ):
            local_cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
            ret, frame = local_cap.read()
        
        if not ret:
            return None
        
        # Extract strip
        strip_height = self._video_height // self.stripping
        y_start = strip_idx * strip_height
        y_end = y_start + strip_height
        
        strip = frame[y_start:y_end, :]
        with self._cache_lock:
            self._strip_cache[cache_key] = strip
        return strip

    def _get_or_build_page_canvas(
        self,
        page_idx: int,
        cap: Optional[cv2.VideoCapture] = None,
    ) -> np.ndarray:
        """Build (once) and return static page canvas without current highlight."""
        with self._cache_lock:
            cached = self._page_cache.get(page_idx)
        if cached is not None:
            return cached

        trace_mode = self._trace_mode_for_cap(cap)

        def _build_page() -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int, int, bool]]]:
            canvas = np.ones((self.region.height, self.region.width, 3), dtype=np.uint8) * 255
            rows: Dict[int, Tuple[int, int, int, int, bool]] = {}

            for i, dig_info in enumerate(self._page_entries(page_idx)):
                frame_id = dig_info.get("frame_id", 0)
                strip_idx = dig_info.get("strip_idx", 0)
                is_selected = bool(dig_info.get("selected_by_digger", False))

                with self._trace_span(
                    "renderer.strip_grid.row_total",
                    {"page_idx": page_idx, "frame_id": frame_id, "mode": trace_mode},
                ):
                    strip = self._get_strip(frame_id, strip_idx, cap=cap)
                if strip is None:
                    continue

                with self._trace_span(
                    "renderer.strip_grid.row_resize",
                    {"page_idx": page_idx, "frame_id": frame_id, "mode": trace_mode},
                ):
                    display_strip = cv2.resize(strip, (self._strip_display_width, self._strip_display_height - 5))
                if not is_selected:
                    with self._trace_span(
                        "renderer.strip_grid.row_fade",
                        {"page_idx": page_idx, "frame_id": frame_id, "mode": trace_mode},
                    ):
                        display_strip = self.apply_fade(display_strip, self.fade_opacity)

                y_pos = i * self._strip_display_height
                x_pos = 5
                h, w = display_strip.shape[:2]
                with self._trace_span(
                    "renderer.strip_grid.row_blit",
                    {"page_idx": page_idx, "frame_id": frame_id, "mode": trace_mode},
                ):
                    canvas[y_pos:y_pos + h, x_pos:x_pos + w] = display_strip

                info_text = f"F{frame_id} S{strip_idx + 1}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                with self._trace_span(
                    "renderer.strip_grid.row_draw_id",
                    {"page_idx": page_idx, "frame_id": frame_id, "mode": trace_mode},
                ):
                    cv2.putText(canvas, info_text, (x_pos + 5, y_pos + 20), font, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(canvas, info_text, (x_pos + 5, y_pos + 20), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                rows[frame_id] = (x_pos, y_pos, w, h, is_selected)

            return canvas, rows

        if self.tracer and cap is self._cap:
            with self.tracer.span("renderer.strip_grid.build_page_sync", args={"page_idx": page_idx, "mode": "render"}):
                canvas, rows = _build_page()
        else:
            canvas, rows = _build_page()

        with self._trace_span(
            "renderer.strip_grid.page_commit_cache",
            {"page_idx": page_idx, "mode": trace_mode},
        ):
            with self._cache_lock:
                existing = self._page_cache.get(page_idx)
                if existing is not None:
                    return existing
                self._page_rows[page_idx] = rows
                self._page_cache[page_idx] = canvas
                return canvas
    
    def _frame_id_to_timestamp(self, frame_id: int) -> float:
        """Convert frame_id to timestamp."""
        return (frame_id - 1) / self.target_fps
    
    def render(self, timestamp: float, frame_idx: int) -> np.ndarray:
        """Render the strip grid panel."""
        if not self._sorted_digger:
            return self.render_placeholder()
        
        # Determine which page of strips to show
        current_frame_id = int(timestamp * self.target_fps) + 1

        # Highlight behavior:
        # Use the nearest strip frame at or before current frame.
        # If current is before all strips, fall back to the first strip frame.
        active_frame_id = None
        active_idx = get_active_index_at_or_before(self._sorted_frame_ids, current_frame_id, fallback_to_first=True)
        if active_idx is not None:
            active_frame_id = self._sorted_frame_ids[active_idx]

        page_idx = 0
        if active_frame_id is not None:
            page_idx = self._frame_to_index.get(active_frame_id, 0) // max(1, self._num_visible_strips)

        canvas = self._get_or_build_page_canvas(page_idx).copy()

        # Schedule future pages for background build.
        for offset in range(1, self.prefetch_depth + 1):
            self._schedule_prefetch(page_idx + offset)

        # Draw current highlight on top of static page.
        if active_frame_id is not None:
            with self._cache_lock:
                rows = self._page_rows.get(page_idx, {})
            row = rows.get(active_frame_id)
            if row is not None:
                x_pos, y_pos, width, height, is_selected = row
                border_color = self.highlight_color if is_selected else self._inactive_highlight_color
                cv2.rectangle(
                    canvas,
                    (x_pos, y_pos),
                    (x_pos + width, y_pos + height),
                    border_color,
                    self.highlight_thickness
                )

        return canvas
    
    def close(self) -> None:
        """Release video capture."""
        self._prefetch_stop.set()
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
            self._prefetch_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._prefetch_cap is not None:
            self._prefetch_cap.release()
            self._prefetch_cap = None
    
    def __del__(self):
        self.close()


