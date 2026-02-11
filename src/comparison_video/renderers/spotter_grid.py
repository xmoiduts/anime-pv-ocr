"""
Spotter Grid Renderer.

Displays 12-grid (4x3) view of spotter-selected frames.
Shows all grids with scrolling, highlighting current position.
Uses different colors for spotter vs digger frames.
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


class SpotterGridRenderer(BaseRenderer):
    """
    Renders the spotter 12-grid panel.
    
    Shows frames in a 4-column x 3-row grid layout.
    - Unselected frames are faded (50% opacity with white background)
    - Spotter frames have green border
    - Digger frames have blue border
    - Current time position has thick highlight
    """
    
    def __init__(
        self,
        region: RenderRegion,
        media_path: str,
        sampled_frames: List[Dict],  # List of {frame_id, source}, source can be None
        target_fps: float = 6.0,
        grid_cols: int = 4,
        grid_rows: int = 3,
        spotter_color: Tuple[int, int, int] = (0, 255, 0),  # Green in BGR
        digger_color: Tuple[int, int, int] = (255, 0, 0),   # Blue in BGR
        highlight_thickness: int = 3,
        fade_opacity: float = 0.5,
        enable_prefetch: bool = True,
        prefetch_depth: int = 2,
        tracer: Optional["PerfTracer"] = None,
    ):
        """
        Initialize the spotter grid renderer.
        
        Args:
            region: The region to render into.
            media_path: Path to the original video file.
            sampled_frames: List of sampled frame info dicts with 'frame_id' and optional 'source'.
            target_fps: Target FPS used for frame extraction.
            grid_cols: Number of columns in the grid.
            grid_rows: Number of rows in the grid.
            spotter_color: Color for spotter frame borders (BGR).
            digger_color: Color for digger frame borders (BGR).
            highlight_thickness: Thickness of highlight border.
            fade_opacity: Opacity for non-current frames (0-1).
        """
        super().__init__(region)
        self.media_path = media_path
        self.sampled_frames = sampled_frames
        self.target_fps = target_fps
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.cells_per_page = grid_cols * grid_rows
        self.spotter_color = spotter_color
        self.digger_color = digger_color
        self.highlight_thickness = highlight_thickness
        self.fade_opacity = fade_opacity
        self.enable_prefetch = enable_prefetch
        self.prefetch_depth = max(0, prefetch_depth)
        self.tracer = tracer
        
        # Video properties
        self._cap: Optional[cv2.VideoCapture] = None
        self._prefetch_cap: Optional[cv2.VideoCapture] = None
        self._video_fps: float = 0

        # Shared cache lock for frame/cell/page caches.
        self._cache_lock = threading.RLock()
        
        # Frame cache
        self._frame_cache: Dict[int, np.ndarray] = {}

        # Cell cache: frame_id -> {"normal": np.ndarray, "faded": np.ndarray}
        # These are preprocessed to target cell size to avoid per-frame resize/fade.
        self._cell_cache: Dict[int, Dict[str, np.ndarray]] = {}

        # Page cache: page_idx -> base canvas (all cells faded, thin borders, texts).
        # Per-frame render only copies page and draws current-cell overlay/highlight.
        self._page_cache: Dict[int, np.ndarray] = {}

        # Page slot lookup: page_idx -> frame_id -> (x, y, cell_width, cell_height, source)
        self._page_slots: Dict[int, Dict[int, Tuple[int, int, int, int, Optional[str]]]] = {}
        
        # Build frame info lookup
        self._frame_info: Dict[int, Dict] = {}
        
        for f in sampled_frames:
            frame_id = f.get("frame_id", 0)
            self._frame_info[frame_id] = f
        
        # Sorted frames
        self._sorted_frames = sorted(sampled_frames, key=lambda x: x.get("frame_id", 0))
        self._sorted_frame_ids = [f.get("frame_id", 0) for f in self._sorted_frames]
        self._total_pages = (len(self._sorted_frames) + self.cells_per_page - 1) // self.cells_per_page

        # Prefetch worker state
        self._prefetch_stop = threading.Event()
        self._prefetch_queue: "queue.Queue[int]" = queue.Queue()
        self._prefetch_pending: set[int] = set()
        self._prefetch_pending_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        
        self._open_video()
        self._open_prefetch_video()
        self._start_prefetch_worker()
        self.set_placeholder("No Spotter Data")
    
    def _open_video(self) -> None:
        """Open video for frame extraction."""
        self._cap = cv2.VideoCapture(self.media_path)
        if self._cap.isOpened():
            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)

    def _open_prefetch_video(self) -> None:
        """Open dedicated video capture for prefetch worker."""
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
            name="spotter-grid-prefetch",
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
                        "renderer.spotter_grid.prefetch_page",
                        args={"page_idx": page_idx, "mode": "prefetch"},
                    ):
                        self._prefetch_page_frames_sequential(
                            page_idx=page_idx,
                            cap=self._prefetch_cap,
                        )
                        self._get_or_build_page_canvas(
                            page_idx=page_idx,
                            margin=2,
                            cell_width=(self.region.width - 2 * (self.grid_cols + 1)) // self.grid_cols,
                            cell_height=(self.region.height - 2 * (self.grid_rows + 1)) // self.grid_rows,
                            cap=self._prefetch_cap,
                        )
                else:
                    self._prefetch_page_frames_sequential(
                        page_idx=page_idx,
                        cap=self._prefetch_cap,
                    )
                    self._get_or_build_page_canvas(
                        page_idx=page_idx,
                        margin=2,
                        cell_width=(self.region.width - 2 * (self.grid_cols + 1)) // self.grid_cols,
                        cell_height=(self.region.height - 2 * (self.grid_rows + 1)) // self.grid_rows,
                        cap=self._prefetch_cap,
                    )
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

    def _page_frame_ids(self, page_idx: int) -> List[int]:
        """Return frame ids shown on a page."""
        page_start = page_idx * self.cells_per_page
        page_end = min(page_start + self.cells_per_page, len(self._sorted_frames))
        return [
            self._sorted_frames[idx].get("frame_id", 0)
            for idx in range(page_start, page_end)
        ]

    def _prefetch_page_frames_sequential(
        self,
        page_idx: int,
        cap: Optional[cv2.VideoCapture],
    ) -> None:
        """
        Prefetch all missing page frames via one seek + sequential reads.

        This avoids 12 independent random seeks for one spotter page.
        """
        if cap is None or not cap.isOpened():
            return

        frame_ids = self._page_frame_ids(page_idx)
        if not frame_ids:
            return

        frame_interval = max(1, int(round(self._video_fps / self.target_fps)))
        targets: List[Tuple[int, int]] = []
        with self._cache_lock:
            for frame_id in frame_ids:
                if frame_id in self._frame_cache:
                    continue
                video_frame_idx = (frame_id - 1) * frame_interval
                targets.append((video_frame_idx, frame_id))

        if not targets:
            return

        targets.sort(key=lambda x: x[0])
        first_idx = targets[0][0]
        last_idx = targets[-1][0]
        target_by_video_idx = {video_idx: frame_id for video_idx, frame_id in targets}
        requested = len(targets)
        hits = 0

        with self._trace_span(
            "renderer.spotter_grid.prefetch_seq_window",
            {
                "page_idx": page_idx,
                "mode": "prefetch",
                "requested": requested,
                "first_video_idx": first_idx,
                "last_video_idx": last_idx,
                "window_frames": max(0, last_idx - first_idx + 1),
            },
        ):
            with self._trace_span(
                "renderer.spotter_grid.prefetch_seq_seek",
                {"page_idx": page_idx, "mode": "prefetch", "seek_to": first_idx},
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)

            current_idx = first_idx
            while current_idx <= last_idx and hits < requested:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id = target_by_video_idx.get(current_idx)
                if frame_id is not None:
                    with self._cache_lock:
                        # Do not overwrite if render thread already cached it.
                        if frame_id not in self._frame_cache:
                            self._frame_cache[frame_id] = frame.copy()
                    hits += 1

                current_idx += 1

            with self._trace_span(
                "renderer.spotter_grid.prefetch_seq_result",
                {
                    "page_idx": page_idx,
                    "mode": "prefetch",
                    "requested": requested,
                    "hits": hits,
                    "misses": max(0, requested - hits),
                },
            ):
                pass

    def _get_frame(self, frame_id: int, cap: Optional[cv2.VideoCapture] = None) -> Optional[np.ndarray]:
        """
        Get a video frame by frame_id.
        
        Args:
            frame_id: The frame ID (1-based).
            
        Returns:
            The frame as BGR image, or None if extraction fails.
        """
        with self._cache_lock:
            cached = self._frame_cache.get(frame_id)
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
            "renderer.spotter_grid.decode_frame",
            {"frame_id": frame_id, "video_frame_idx": video_frame_idx, "mode": mode},
        ):
            local_cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
            ret, frame = local_cap.read()
        
        if ret:
            with self._cache_lock:
                self._frame_cache[frame_id] = frame
            return frame
        return None
    
    def _frame_id_to_timestamp(self, frame_id: int) -> float:
        """Convert frame_id to timestamp."""
        return (frame_id - 1) / self.target_fps

    def _get_cell_variants(
        self,
        frame_id: int,
        cell_width: int,
        cell_height: int,
        cap: Optional[cv2.VideoCapture] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Get preprocessed cell images for a frame.

        Returns:
            Dict with keys:
            - "normal": fitted cell image
            - "faded": faded cell image for non-current state
            or None if frame extraction fails.
        """
        with self._cache_lock:
            cached = self._cell_cache.get(frame_id)
        if cached is not None:
            return cached

        mode = self._trace_mode_for_cap(cap)
        frame = self._get_frame(frame_id, cap=cap)
        if frame is None:
            return None

        with self._trace_span(
            "renderer.spotter_grid.cell_resize",
            {"frame_id": frame_id, "mode": mode},
        ):
            normal = self.fit_image_to_region(frame, cell_width, cell_height)
        with self._trace_span(
            "renderer.spotter_grid.cell_fade",
            {"frame_id": frame_id, "mode": mode},
        ):
            faded = self.apply_fade(normal, self.fade_opacity)
        cached = {"normal": normal, "faded": faded}
        with self._cache_lock:
            self._cell_cache[frame_id] = cached
        return cached

    def _get_or_build_page_canvas(
        self,
        page_idx: int,
        margin: int,
        cell_width: int,
        cell_height: int,
        cap: Optional[cv2.VideoCapture] = None,
    ) -> np.ndarray:
        """Build (once) and return page base canvas for page_idx."""
        with self._cache_lock:
            cached = self._page_cache.get(page_idx)
        if cached is not None:
            return cached

        trace_mode = self._trace_mode_for_cap(cap)

        def _build_page() -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int, int, Optional[str]]]]:
            canvas = np.ones((self.region.height, self.region.width, 3), dtype=np.uint8) * 255
            slots: Dict[int, Tuple[int, int, int, int, Optional[str]]] = {}

            page_start = page_idx * self.cells_per_page
            page_end = min(page_start + self.cells_per_page, len(self._sorted_frames))

            for i, idx in enumerate(range(page_start, page_end)):
                frame_info = self._sorted_frames[idx]
                frame_id = frame_info.get("frame_id", 0)
                source = frame_info.get("source")

                # Calculate grid position
                row = i // self.grid_cols
                col = i % self.grid_cols
                x = margin + col * (cell_width + margin)
                y = margin + row * (cell_height + margin)

                with self._trace_span(
                    "renderer.spotter_grid.cell_total",
                    {"page_idx": page_idx, "frame_id": frame_id, "mode": trace_mode},
                ):
                    cell_variants = self._get_cell_variants(frame_id, cell_width, cell_height, cap=cap)
                if cell_variants is not None:
                    # Brightness is based on OCR selection, not current playback frame:
                    # - selected (spotter/digger): normal brightness
                    # - unselected: faded
                    is_selected = source in ("spotter", "digger")
                    base_cell = cell_variants["normal"] if is_selected else cell_variants["faded"]

                    with self._trace_span(
                        "renderer.spotter_grid.cell_blit_faded",
                        {"page_idx": page_idx, "frame_id": frame_id, "mode": trace_mode},
                    ):
                        canvas[y:y + cell_height, x:x + cell_width] = base_cell

                    if source == "spotter":
                        border_color = self.spotter_color
                    elif source == "digger":
                        border_color = self.digger_color
                    else:
                        border_color = (160, 160, 160)

                    with self._trace_span(
                        "renderer.spotter_grid.cell_draw_border",
                        {"page_idx": page_idx, "frame_id": frame_id, "mode": trace_mode},
                    ):
                        cv2.rectangle(
                            canvas,
                            (x, y),
                            (x + cell_width - 1, y + cell_height - 1),
                            border_color,
                            1,
                        )

                    # Draw frame ID only once when page is built
                    id_text = str(frame_id)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.35
                    with self._trace_span(
                        "renderer.spotter_grid.cell_draw_id",
                        {"page_idx": page_idx, "frame_id": frame_id, "mode": trace_mode},
                    ):
                        (tw, th), _ = cv2.getTextSize(id_text, font, font_scale, 1)
                        cv2.rectangle(canvas, (x + 2, y + 2), (x + tw + 6, y + th + 6), (0, 0, 0), -1)
                        cv2.putText(canvas, id_text, (x + 4, y + th + 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

                slots[frame_id] = (x, y, cell_width, cell_height, source)

            # Draw page indicator (static per page)
            with self._trace_span(
                "renderer.spotter_grid.page_draw_indicator",
                {"page_idx": page_idx, "mode": trace_mode},
            ):
                total_pages = (len(self._sorted_frames) + self.cells_per_page - 1) // self.cells_per_page
                page_text = f"Page {page_idx + 1}/{total_pages}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, _), _ = cv2.getTextSize(page_text, font, 0.4, 1)
                cv2.putText(
                    canvas,
                    page_text,
                    (self.region.width - tw - 5, self.region.height - 5),
                    font,
                    0.4,
                    (100, 100, 100),
                    1,
                    cv2.LINE_AA,
                )
            return canvas, slots

        if self.tracer and cap is self._cap:
            with self.tracer.span("renderer.spotter_grid.build_page_sync", args={"page_idx": page_idx}):
                canvas, slots = _build_page()
        else:
            canvas, slots = _build_page()

        with self._trace_span(
            "renderer.spotter_grid.page_commit_cache",
            {"page_idx": page_idx, "mode": trace_mode},
        ):
            with self._cache_lock:
                existing = self._page_cache.get(page_idx)
                if existing is not None:
                    return existing
                self._page_slots[page_idx] = slots
                self._page_cache[page_idx] = canvas
                return canvas
    
    def render(self, timestamp: float, frame_idx: int) -> np.ndarray:
        """Render the spotter grid panel."""
        if not self._sorted_frames:
            return self.render_placeholder()

        # Calculate cell dimensions
        margin = 2
        cell_width = (self.region.width - margin * (self.grid_cols + 1)) // self.grid_cols
        cell_height = (self.region.height - margin * (self.grid_rows + 1)) // self.grid_rows

        # Determine current frame and page
        current_frame_id = int(timestamp * self.target_fps) + 1
        current_idx = get_active_index_at_or_before(self._sorted_frame_ids, current_frame_id, fallback_to_first=True)
        if current_idx is None:
            return self.render_placeholder()
        current_frame_id = self._sorted_frame_ids[current_idx]
        page_idx = current_idx // self.cells_per_page

        # Base page is static and cached. Copy for per-frame dynamic overlay.
        canvas = self._get_or_build_page_canvas(page_idx, margin, cell_width, cell_height).copy()

        # Schedule future pages for background build.
        for offset in range(1, self.prefetch_depth + 1):
            self._schedule_prefetch(page_idx + offset)

        # Current frame only controls highlight border thickness/color.
        # Cell brightness is already encoded in the cached page by source selection.
        with self._cache_lock:
            current_slots = self._page_slots.get(page_idx, {})
        slot = current_slots.get(current_frame_id)
        if slot is not None:
            x, y, cw, ch, source = slot
            if source == "spotter":
                border_color = self.spotter_color
            elif source == "digger":
                border_color = self.digger_color
            else:
                border_color = (160, 160, 160)

            cv2.rectangle(
                canvas,
                (x, y),
                (x + cw - 1, y + ch - 1),
                border_color,
                self.highlight_thickness,
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


