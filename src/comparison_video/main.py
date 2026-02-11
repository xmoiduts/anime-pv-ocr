"""
Main entry point for comparison video generation.

Usage:
    python -m comparison_video.main -i <media_substring> [options]
"""

import argparse
import os
import sys
from time import perf_counter_ns
from typing import Dict, Any, Optional

import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison_video.layout import LayoutManager, LayoutConfig
from comparison_video.perf_trace import PerfTracer
from comparison_video.timeline import TimelineBuilder
from comparison_video.compositor import FrameCompositor
from comparison_video.ffmpeg_writer import FFmpegWriter
from comparison_video.renderers import (
    CurrentFrameRenderer,
    LyricsPanelRenderer,
    OriginalVideoRenderer,
    StripGridRenderer,
    SpotterGridRenderer,
)
from comparison_video.renderers.base import RenderRegion

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _resolve_project_path(path: str) -> str:
    """
    Resolve CLI paths against project root for stable behavior.

    This keeps defaults working no matter where the command is executed from.
    """
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _safe_print(message: str) -> None:
    """Print text without crashing on Windows console encoding issues."""
    try:
        print(message)
    except UnicodeEncodeError:
        escaped = message.encode("unicode_escape").decode("ascii")
        print(escaped)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_media_and_folder(
    substring: str,
    outputs_dir: str = "outputs",
    medias_dir: str = "medias"
) -> tuple:
    """
    Find media file and output folder by substring match.
    
    Returns:
        (folder_path, media_path) or (None, None) if not found.
    """
    # Import from parent module
    try:
        from media_utils import find_target_folder
        return find_target_folder(substring, outputs_dir, medias_dir)
    except ImportError:
        pass
    
    # Fallback implementation
    import hashlib
    
    def get_expected_output_dir(filename: str, base_dir: str) -> str:
        basename = os.path.basename(filename)
        name_part = basename[:10]
        hash_part = hashlib.md5(basename.encode("utf-8")).hexdigest()[:8]
        name_part = "".join([c for c in name_part if c.isalnum() or c in (" ", ".", "_", "-")]).strip()
        return os.path.join(base_dir, f"{name_part}{hash_part}")
    
    # Search for matching media file
    if os.path.exists(medias_dir):
        for f in os.listdir(medias_dir):
            if substring.lower() in f.lower():
                media_path = os.path.join(medias_dir, f)
                folder_path = get_expected_output_dir(f, outputs_dir)
                return folder_path, media_path
    
    # Search for matching output folder
    if os.path.exists(outputs_dir):
        for f in os.listdir(outputs_dir):
            if substring.lower() in f.lower():
                folder_path = os.path.join(outputs_dir, f)
                return folder_path, None
    
    return None, None


def build_renderers(
    layout: LayoutManager,
    media_path: str,
    timeline: TimelineBuilder,
    task_config: Dict[str, Any],
    tracer: Optional[PerfTracer] = None,
) -> Dict[str, any]:
    """
    Build all renderers with their data.
    
    Args:
        layout: LayoutManager instance.
        media_path: Path to source video.
        timeline: TimelineBuilder with loaded data.
        task_config: Task configuration dict.
        
    Returns:
        Dictionary of renderer name -> renderer instance.
    """
    renderers = {}
    effects_config = task_config.get("effects", {})
    prefetch_config = task_config.get("prefetch", {})
    prefetch_enabled = bool(prefetch_config.get("enabled", True))
    prefetch_depth = max(0, int(prefetch_config.get("depth", 2)))
    
    # 1. Original Video Renderer
    region = layout.get_region("original_video")
    renderers["original_video"] = OriginalVideoRenderer(region, media_path)
    
    # 2. Current Frame Renderer
    region = layout.get_region("current_frame")
    frame_cache = {}
    ocr_timestamps = {}
    
    # Build OCR frame timestamps from timeline
    for ocr_frame in timeline.ocr_frames:
        ocr_timestamps[ocr_frame.frame_id] = timeline._frame_id_to_timestamp(ocr_frame.frame_id)
    
    current_frame_renderer = CurrentFrameRenderer(
        region=region,
        frame_cache=frame_cache,
        ocr_frame_timestamps=ocr_timestamps,
        target_fps=timeline.target_fps,
    )
    
    # Preload OCR frames
    ocr_frame_ids = [f.frame_id for f in timeline.ocr_frames]
    if ocr_frame_ids:
        current_frame_renderer.preload_frames(media_path, ocr_frame_ids, timeline.target_fps)
    
    renderers["current_frame"] = current_frame_renderer
    
    # 3. Lyrics Panel Renderer
    region = layout.get_region("lyrics_panel")
    lyrics_data = {f.frame_id: f.lyric for f in timeline.ocr_frames}
    
    renderers["lyrics_panel"] = LyricsPanelRenderer(
        region=region,
        lyrics_data=lyrics_data,
        ocr_frame_timestamps=ocr_timestamps,
    )
    
    # 4. Strip Grid Renderer (dig-hard)
    region = layout.get_region("strip_grid")
    hard_sample_strips = [
        {
            "frame_id": s.frame_id,
            "strip_idx": s.strip_idx,
            "selected_by_digger": s.selected_by_digger,
        }
        for s in timeline.hard_sample_strips
    ]
    
    renderers["strip_grid"] = StripGridRenderer(
        region=region,
        media_path=media_path,
        strip_frames=hard_sample_strips,
        target_fps=timeline.target_fps,
        stripping=timeline.stripping,
        visible_rows=timeline.digger_grid_rows,
        highlight_color=tuple(effects_config.get("digger_color", [255, 0, 0])),
        fade_opacity=effects_config.get("fade_opacity", 0.5),
        enable_prefetch=prefetch_enabled,
        prefetch_depth=prefetch_depth,
        tracer=tracer,
    )
    
    # 5. Spotter Grid Renderer
    region = layout.get_region("spotter_grid")
    sampled_frames = [
        {"frame_id": frame_id, "source": timeline.get_spotter_frame_source(frame_id)}
        for frame_id in timeline.get_all_sampled_frame_ids()
    ]
    
    renderers["spotter_grid"] = SpotterGridRenderer(
        region=region,
        media_path=media_path,
        sampled_frames=sampled_frames,
        target_fps=timeline.target_fps,
        grid_cols=timeline.spotter_cols,
        grid_rows=timeline.spotter_rows,
        spotter_color=tuple(effects_config.get("spotter_color", [0, 255, 0])),
        digger_color=tuple(effects_config.get("digger_color", [255, 0, 0])),
        fade_opacity=effects_config.get("fade_opacity", 0.5),
        enable_prefetch=prefetch_enabled,
        prefetch_depth=prefetch_depth,
        tracer=tracer,
    )
    
    return renderers


def generate_comparison_video(
    media_path: str,
    folder_path: str,
    output_path: str,
    task_config: Dict[str, Any],
    dig_hard_config: Optional[Dict[str, Any]] = None,
    enable_perf_trace: bool = False,
    perf_trace_path: Optional[str] = None,
    enable_preview: bool = False,
) -> bool:
    """
    Generate the comparison video.
    
    Args:
        media_path: Path to source video.
        folder_path: Path to output folder containing results.
        output_path: Path for output video file.
        task_config: Task configuration dict.
        dig_hard_config: dig-hard-samples task configuration dict.
        enable_perf_trace: Whether to enable high-precision performance tracing.
        perf_trace_path: Path to trace output JSON file.
        enable_preview: Whether to show live preview window (default: False).
        
    Returns:
        True if successful, False otherwise.
    """
    import cv2
    
    tracer = PerfTracer(enabled=enable_perf_trace, output_path=perf_trace_path)

    # Get video properties
    with tracer.span("init.open_video"):
        cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {media_path}")
        return False
    
    with tracer.span("init.read_video_meta"):
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / video_fps
        cap.release()
    
    print(f"Source video: {video_fps:.2f} fps, {video_duration:.2f}s, {total_frames} frames")
    
    # Get configuration
    target_fps = task_config.get("target_fps", 6.0)
    stripping = task_config.get("stripping", 5)
    spotter_grid = task_config.get("spotter_grid", {"cols": 4, "rows": 3})
    dig_hard_config = dig_hard_config or {}
    digger_grid_rows = dig_hard_config.get("grid_rows", stripping)
    avoid_before = dig_hard_config.get("avoid_before", 3)
    avoid_after = dig_hard_config.get("avoid_after", 3)
    
    # Create layout
    with tracer.span("init.layout"):
        layout = LayoutManager.from_config(task_config)
        output_width, output_height = layout.output_size
    print(f"Output video: {output_width}x{output_height} @ {video_fps:.2f} fps")
    
    # Build timeline
    with tracer.span("init.timeline_construct"):
        timeline = TimelineBuilder(
            video_fps=video_fps,
            video_duration=video_duration,
            target_fps=target_fps,
            stripping=stripping,
            digger_grid_rows=digger_grid_rows,
            avoid_before=avoid_before,
            avoid_after=avoid_after,
            spotter_grid_size=(spotter_grid.get("cols", 4), spotter_grid.get("rows", 3)),
        )
    
    # Load data from folder
    with tracer.span("init.timeline_load"):
        timeline.load_from_folder(folder_path)
    
    print(f"Loaded: {len(timeline.spotter_frames)} spotter frames, "
          f"{len(timeline.digger_strips)} digger strips, "
          f"{len(timeline.ocr_frames)} OCR frames")
    
    # Build renderers
    with tracer.span("init.build_renderers"):
        renderers = build_renderers(layout, media_path, timeline, task_config, tracer=tracer)
    
    # Create compositor
    with tracer.span("init.compositor_setup"):
        compositor = FrameCompositor(layout, tracer=tracer)
        for name, renderer in renderers.items():
            compositor.register_renderer(name, renderer)
    
    # Generate video
    preview_config = task_config.get("preview", {})
    preview_enabled = enable_preview
    preview_window_name = "Comparison Video Preview"
    preview_scale = preview_config.get("scale", 0.5)  # Scale for preview window
    
    if preview_enabled:
        cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window_name, int(output_width * preview_scale), int(output_height * preview_scale))
        print("\nPreview Controls:")
        print("  - Press 'P' or SPACE to toggle preview on/off")
        print("  - Press 'Q' or ESC to quit (saves video up to current frame)")
        print("  - Preview window is enabled by default\n")
    
    writer = FFmpegWriter(
        output_path=output_path,
        width=output_width,
        height=output_height,
        fps=video_fps,
        audio_source_path=media_path,
    )
    try:
        with tracer.span("ffmpeg.open"):
            writer.open()

        for frame_idx in tqdm(range(total_frames), desc="Generating video"):
            frame_loop_start_ns = perf_counter_ns()
            timestamp = frame_idx / video_fps

            # Composite frame
            with tracer.span("pipeline.composite_frame", frame_idx=frame_idx):
                output_frame = compositor.composite_frame(timestamp, frame_idx)

            # Write to video
            with tracer.span("pipeline.ffmpeg_write_frame", frame_idx=frame_idx):
                writer.write_frame(output_frame)

            # Show preview if enabled
            if preview_enabled:
                with tracer.span("preview.imshow", frame_idx=frame_idx):
                    cv2.imshow(preview_window_name, output_frame)

                # Process keyboard events
                with tracer.span("preview.waitKey", frame_idx=frame_idx):
                    key = cv2.waitKey(1) & 0xFF

                # Toggle preview with 'p' or space
                if key == ord('p') or key == ord('P') or key == 32:  # 32 is space
                    preview_enabled = not preview_enabled
                    if preview_enabled:
                        cv2.imshow(preview_window_name, output_frame)
                        print(f"\n[Frame {frame_idx}] Preview enabled")
                    else:
                        cv2.destroyWindow(preview_window_name)
                        print(f"\n[Frame {frame_idx}] Preview disabled")

                # Quit with 'q' or ESC
                elif key == ord('q') or key == ord('Q') or key == 27:  # 27 is ESC
                    print(f"\n[Frame {frame_idx}] User requested quit. Finalizing video...")
                    break

            tracer.add_counter(
                "frame.loop_wall_ns",
                float(perf_counter_ns() - frame_loop_start_ns),
                frame_idx=frame_idx,
            )
    
    finally:
        with tracer.span("ffmpeg.close"):
            writer.close()

        # Cleanup preview window
        if enable_preview:
            cv2.destroyAllWindows()
        
        # Cleanup renderers
        for renderer in renderers.values():
            if hasattr(renderer, "close"):
                renderer.close()

        tracer.flush()
        if enable_perf_trace and perf_trace_path:
            print(f"Performance trace saved to: {perf_trace_path}")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comparison video showing OCR pipeline results."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Substring to match media file or output folder name."
    )
    parser.add_argument(
        "-o", "--output",
        help="Output video path. Defaults to <folder>/comparison_video.mp4"
    )
    parser.add_argument(
        "-c", "--config",
        default="ocr-cli-config.yaml",
        help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--media-dir",
        default="medias",
        help="Directory containing source media files."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory containing pipeline output folders."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Enable live preview window during video generation. Press 'P' to toggle, 'Q' to quit."
    )
    parser.add_argument(
        "--trace-perf",
        action="store_true",
        help="Enable high-precision performance tracing and export Chrome trace JSON."
    )
    parser.add_argument(
        "--trace-file",
        help="Performance trace output path (default: <folder>/comparison_video_trace.json)."
    )
    
    args = parser.parse_args()

    config_path = _resolve_project_path(args.config)
    media_dir = _resolve_project_path(args.media_dir)
    output_dir = _resolve_project_path(args.output_dir)
    
    # Load config
    config = load_config(config_path)
    task_root = config.get("task", {})
    task_config = task_root.get("make-comparison-videos", {})
    dig_hard_config = task_root.get("dig-hard-samples", {})
    
    # Merge with defaults
    defaults = {
        "target_fps": 6.0,
        "stripping": 5,
        "spotter_grid": {"cols": 4, "rows": 3},
        "layout": {
            "output_width": 1920,
            "output_height": 1080,
            "left_ratio": 0.4,
            "left_top_ratio": 0.5,
            "right_top_ratio": 0.6,
            "right_bottom_left_ratio": 0.5,
        },
        "effects": {
            "fade_opacity": 0.5,
            "spotter_color": [0, 255, 0],
            "digger_color": [255, 0, 0],
            "highlight_color": [0, 0, 255],
        },
        "prefetch": {
            "enabled": True,
            "depth": 2,
        },
    }
    
    # Deep merge
    for key, value in defaults.items():
        if key not in task_config:
            task_config[key] = value
        elif isinstance(value, dict):
            for k, v in value.items():
                if k not in task_config[key]:
                    task_config[key][k] = v
    
    # Find media and folder
    folder_path, media_path = find_media_and_folder(
        args.input,
        output_dir,
        media_dir,
    )
    
    if not folder_path:
        print(f"Error: No matching folder found for '{args.input}'")
        sys.exit(1)
    
    if not media_path or not os.path.exists(media_path):
        print(f"Error: Media file not found for '{args.input}'")
        sys.exit(1)
    
    _safe_print(f"Media: {media_path}")
    _safe_print(f"Folder: {folder_path}")
    
    # Determine output path
    output_path = args.output
    if not output_path:
        output_path = os.path.join(folder_path, "comparison_video.mp4")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Generate video
    perf_trace_path = args.trace_file or os.path.join(folder_path, "comparison_video_trace.json")
    success = generate_comparison_video(
        media_path=media_path,
        folder_path=folder_path,
        output_path=output_path,
        task_config=task_config,
        dig_hard_config=dig_hard_config,
        enable_perf_trace=args.trace_perf,
        perf_trace_path=perf_trace_path,
        enable_preview=args.preview,
    )
    
    if success:
        print(f"\nComparison video saved to: {output_path}")
    else:
        print("\nFailed to generate comparison video.")
        sys.exit(1)


if __name__ == "__main__":
    main()

