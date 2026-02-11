"""
High precision performance tracing utilities for comparison video generation.

Trace output uses Chrome Trace Event format, viewable in chrome://tracing
or Perfetto UI (https://ui.perfetto.dev).
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Dict, Iterator, List, Optional


@dataclass
class _SpanToken:
    name: str
    start_ns: int
    frame_idx: Optional[int]
    args: Dict[str, object]


class PerfTracer:
    """Collects high-precision spans and writes Chrome trace JSON."""

    def __init__(self, enabled: bool = False, output_path: Optional[str] = None):
        self.enabled = enabled
        self.output_path = output_path
        self._base_perf_ns = perf_counter_ns()
        self._base_wall_ns = time.time_ns()
        self._events: List[Dict[str, object]] = []
        self._pid = os.getpid()

    def _thread_id(self) -> int:
        return threading.get_ident()

    def begin(self, name: str, frame_idx: Optional[int] = None, args: Optional[Dict[str, object]] = None) -> _SpanToken:
        if not self.enabled:
            return _SpanToken(name=name, start_ns=0, frame_idx=frame_idx, args={})
        return _SpanToken(
            name=name,
            start_ns=perf_counter_ns(),
            frame_idx=frame_idx,
            args=args.copy() if args else {},
        )

    def end(self, token: _SpanToken) -> None:
        if not self.enabled:
            return
        end_ns = perf_counter_ns()
        start_ns = token.start_ns
        dur_ns = max(0, end_ns - start_ns)
        event_args = dict(token.args)
        if token.frame_idx is not None:
            event_args["frame_idx"] = token.frame_idx

        self._events.append(
            {
                "name": token.name,
                "ph": "X",
                "pid": self._pid,
                "tid": self._thread_id(),
                "ts": (start_ns - self._base_perf_ns) / 1000.0,
                "dur": dur_ns / 1000.0,
                "args": event_args,
            }
        )

    @contextmanager
    def span(self, name: str, frame_idx: Optional[int] = None, args: Optional[Dict[str, object]] = None) -> Iterator[None]:
        token = self.begin(name, frame_idx=frame_idx, args=args)
        try:
            yield
        finally:
            self.end(token)

    def add_counter(self, name: str, value: float, frame_idx: Optional[int] = None, args: Optional[Dict[str, object]] = None) -> None:
        if not self.enabled:
            return
        counter_args = {"value": value}
        if args:
            counter_args.update(args)
        if frame_idx is not None:
            counter_args["frame_idx"] = frame_idx
        now_ns = perf_counter_ns()
        self._events.append(
            {
                "name": name,
                "ph": "C",
                "pid": self._pid,
                "tid": self._thread_id(),
                "ts": (now_ns - self._base_perf_ns) / 1000.0,
                "args": counter_args,
            }
        )

    def flush(self) -> None:
        if not self.enabled or not self.output_path:
            return

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        payload = {
            "displayTimeUnit": "ms",
            "metadata": {
                "base_wall_time_ns": self._base_wall_ns,
            },
            "traceEvents": self._events,
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
