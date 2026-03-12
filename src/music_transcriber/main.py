"""
Main entry point for music transcription.

Usage:
    python main.py -L 2
    python -m music_transcriber.main -L 2
"""

import argparse
import os
import queue
import signal
import sys
from dataclasses import dataclass
from threading import Event, Thread
from typing import Optional

# Add src directory to path for imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_client import call_gemini
from music_transcriber.config import MODULE_NAME, format_timestamp, load_runtime_context, resolve_project_path
from music_transcriber.io_utils import SavedOutputs, save_outputs
from music_transcriber.media_batch import MediaJob, list_latest_media_jobs
from prompt_builder import build_prompt


@dataclass(frozen=True)
class JobResult:
    job: MediaJob
    outputs: SavedOutputs


@dataclass(frozen=True)
class JobFailure:
    job: MediaJob
    error: str


def _safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        escaped = message.encode("unicode_escape").decode("ascii")
        print(escaped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe latest music folders into LRC with Gemini.")
    parser.add_argument(
        "-L",
        "--latest-medias",
        type=int,
        default=None,
        help="Latest media-name folders to process, sorted by newest contained media file time descending.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Override local config YAML path (default: src/music_transcriber/config.yaml).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override concurrent Gemini request count.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sorted media-folder selection details without sending Gemini requests.",
    )
    return parser.parse_args()


def _transcribe_job(
    job: MediaJob,
    prompt_text: str,
    runtime,
    gemini_generation: Optional[float],
    cancel_event: Event,
) -> JobResult:
    if cancel_event.is_set():
        raise KeyboardInterrupt("Cancelled before job start.")
    _safe_print(
        f"[Start] {job.folder_name} -> {len(job.media_paths)} media file(s)"
    )
    response_text = call_gemini(
        runtime.api_key,
        runtime.local_config.model,
        prompt_text,
        [str(path) for path in job.media_paths],
        runtime.local_config.base_url or os.getenv("GEMINI_BASE_URL"),
        media_resolution=runtime.local_config.media_resolution,
        thinking_level=runtime.local_config.thinking_level,
        exchange_rate=runtime.exchange_rate,
        gemini_generation=gemini_generation,
        pricing_table=runtime.pricing_table,
        cancel_event=cancel_event,
    )
    if not response_text:
        raise RuntimeError(f"Gemini returned no text for '{job.folder_name}'.")

    outputs = save_outputs(
        job.output_dir,
        response_text,
        runtime.local_config.raw_log_filename,
        runtime.local_config.lrc_filename,
    )
    return JobResult(job=job, outputs=outputs)


def _print_job_listing(jobs: list[MediaJob], header: str) -> None:
    _safe_print(header)
    for index, job in enumerate(jobs, start=1):
        _safe_print(
            f"{index:02d}. {job.folder_name} | latest-media-time={format_timestamp(job.sort_mtime)} "
            f"| newest={job.newest_media_path.name} | media-count={len(job.media_paths)}"
        )


def _run_jobs(
    jobs: list[MediaJob],
    prompt_text: str,
    runtime,
    gemini_generation: Optional[float],
    max_workers: int,
    cancel_event: Event,
) -> tuple[list[JobResult], list[JobFailure]]:
    pending_jobs: "queue.Queue[MediaJob]" = queue.Queue()
    result_queue: "queue.Queue[object]" = queue.Queue()

    for job in jobs:
        pending_jobs.put(job)

    def worker() -> None:
        while not cancel_event.is_set():
            try:
                job = pending_jobs.get_nowait()
            except queue.Empty:
                return
            try:
                result = _transcribe_job(
                    job,
                    prompt_text,
                    runtime,
                    gemini_generation,
                    cancel_event,
                )
                result_queue.put(result)
            except KeyboardInterrupt as exc:
                cancel_event.set()
                result_queue.put(JobFailure(job=job, error=str(exc) or "Interrupted"))
                return
            except Exception as exc:
                result_queue.put(JobFailure(job=job, error=str(exc)))
            finally:
                pending_jobs.task_done()

    threads = [
        Thread(target=worker, name=f"music-transcriber-{idx + 1}", daemon=True)
        for idx in range(max_workers)
    ]
    for thread in threads:
        thread.start()

    expected_results = len(jobs)
    completed_results = 0
    successes: list[JobResult] = []
    failures: list[JobFailure] = []
    while completed_results < expected_results:
        item = result_queue.get()
        completed_results += 1
        if isinstance(item, JobResult):
            successes.append(item)
            _safe_print(
                f"[Done] {item.job.folder_name} -> "
                f"{item.outputs.raw_log_path.name}, {item.outputs.lrc_path.name}"
            )
            continue

        failures.append(item)
        _safe_print(f"[Failed] {item.job.folder_name}: {item.error}")

    return successes, failures


def main() -> int:
    args = parse_args()
    runtime = load_runtime_context(args.config)

    latest_count = (
        args.latest_medias
        if args.latest_medias is not None
        else runtime.local_config.default_latest_medias
    )
    media_source_folder = resolve_project_path(runtime.local_config.media_source_folder)
    jobs = list_latest_media_jobs(
        media_source_folder=media_source_folder,
        latest_count=latest_count,
        local_config=runtime.local_config,
        project_root=runtime.project_root,
    )
    if not jobs:
        _safe_print(f"No eligible media folders found under {media_source_folder}")
        return 0

    gemini_generation = runtime.model_configs.get(runtime.local_config.model, {}).get("gemini-generation")
    worker_limit = args.workers if args.workers is not None else runtime.local_config.max_workers
    max_workers = max(1, min(worker_limit, len(jobs)))

    _safe_print(f"Module: {MODULE_NAME}")
    _safe_print(f"Model: {runtime.local_config.model}")
    _safe_print(f"Source: {media_source_folder}")
    _safe_print(f"Selected {len(jobs)} folder(s); running with {max_workers} worker(s).")

    _print_job_listing(jobs, "Sorted selection:")
    if args.dry_run:
        _safe_print("Dry run only. No Gemini requests were sent.")
        return 0

    prompt_text = build_prompt(str(runtime.prompt_path))
    cancel_event = Event()
    original_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame) -> None:
        cancel_event.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        _, failures = _run_jobs(
            jobs=jobs,
            prompt_text=prompt_text,
            runtime=runtime,
            gemini_generation=gemini_generation,
            max_workers=max_workers,
            cancel_event=cancel_event,
        )
    except KeyboardInterrupt:
        cancel_event.set()
        _safe_print("Interrupted by Ctrl+C. Cancelling in-flight Gemini requests.")
        return 130
    finally:
        signal.signal(signal.SIGINT, original_handler)

    if failures:
        _safe_print(f"Completed with failures: {', '.join(item.job.folder_name for item in failures)}")
        return 1

    _safe_print("All selected media folders completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
