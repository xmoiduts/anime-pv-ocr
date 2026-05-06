"""
Main entry point for booklet OCR.

Usage:
    python -m booklet_ocr.main
    python -m booklet_ocr.main --dry-run
"""

import argparse
import os
import sys

# Add src directory to path for imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from booklet_ocr.config import MODULE_NAME, load_runtime_context, resolve_project_path
from booklet_ocr.input_listing import build_input_listing
from booklet_ocr.io_utils import create_output_dir, save_outputs
from prompt_builder import build_prompt
from provider.gemini import call_gemini


def _safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        escaped = message.encode("unicode_escape").decode("ascii")
        print(escaped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize text from booklet images with Gemini.")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Override local config YAML path (default: src/booklet_ocr/config.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected files and output naming details without sending a Gemini request.",
    )
    return parser.parse_args()


def _format_megapixels(pixel_count: int) -> str:
    return f"{pixel_count / 1_000_000:.2f}MP"


def main() -> int:
    args = parse_args()
    try:
        runtime = load_runtime_context(args.config)
        input_source_folder = resolve_project_path(runtime.local_config.input_source_folder)
        listing = build_input_listing(input_source_folder, runtime.local_config)
    except Exception as exc:
        _safe_print(f"Error: {exc}")
        return 1

    if not listing.media_paths:
        _safe_print(f"No eligible booklet media files found under {input_source_folder}")
        return 0

    _safe_print(f"Module: {MODULE_NAME}")
    _safe_print(f"Model: {runtime.local_config.model}")
    _safe_print(f"Source: {input_source_folder}")
    _safe_print(f"Selected {len(listing.media_paths)} file(s) for a single Gemini request.")

    if len(listing.media_paths) > runtime.local_config.warn_file_count_over:
        _safe_print(
            f"Warning: file count {len(listing.media_paths)} exceeds "
            f"{runtime.local_config.warn_file_count_over}. No auto-splitting will be applied."
        )

    _safe_print("Selected media:")
    for index, path in enumerate(listing.media_paths, start=1):
        pixel_count = listing.pixel_counts[str(path)]
        resolution_level = listing.per_file_media_resolution[str(path)]
        _safe_print(
            f"{index:02d}. {path.name} | pixels={pixel_count} ({_format_megapixels(pixel_count)}) "
            f"| resolution={resolution_level}"
        )

    output_dir, timestamp_prefix = create_output_dir(
        runtime.project_root,
        runtime.local_config.output_root,
        create=not args.dry_run,
    )
    _safe_print(f"Output directory: {output_dir}")
    _safe_print(f"Output prefix: {timestamp_prefix}")
    if args.dry_run:
        _safe_print("Dry run only. No Gemini request was sent.")
        return 0

    prompt_text = build_prompt(str(runtime.prompt_path))
    gemini_generation = runtime.model_configs.get(runtime.local_config.model, {}).get("gemini-generation")
    result = call_gemini(
        runtime.api_key,
        runtime.local_config.model,
        prompt_text,
        [str(path) for path in listing.media_paths],
        runtime.local_config.base_url or os.getenv("GEMINI_BASE_URL"),
        thinking_level=runtime.local_config.thinking_level,
        exchange_rate=runtime.exchange_rate,
        gemini_generation=gemini_generation,
        pricing_table=runtime.pricing_table,
        per_file_media_resolution=listing.per_file_media_resolution,
    )
    if not result or not result.response_text:
        _safe_print("Gemini returned no text.")
        return 1

    try:
        outputs = save_outputs(
            output_dir=output_dir,
            timestamp_prefix=timestamp_prefix,
            raw_response=result.response_text,
            raw_log_suffix=runtime.local_config.raw_log_suffix,
            yaml_suffix=runtime.local_config.yaml_suffix,
            lrc_suffix=runtime.local_config.lrc_suffix,
            thought_text=result.thought_text,
        )
    except Exception as exc:
        _safe_print(f"Error: {exc}")
        return 1

    _safe_print(f"Saved raw output: {outputs.raw_log_path.name}")
    _safe_print(f"Saved YAML output: {outputs.yaml_path.name}")
    _safe_print(f"Saved LRC output: {outputs.lrc_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
