"""
FFmpeg Writer for comparison video.

Pipes raw video frames to ffmpeg for encoding without saving intermediate files.
"""

import subprocess
import threading
import os
from typing import Optional

import numpy as np


class FFmpegWriter:
    """
    Writes video frames to ffmpeg via stdin pipe.
    """
    
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx264",
        preset: str = "fast",
        crf: int = 23,
        pixel_format: str = "yuv420p",
        audio_source_path: Optional[str] = None,
    ):
        """
        Initialize the FFmpeg writer.
        
        Args:
            output_path: Path for output video file.
            width: Video width in pixels.
            height: Video height in pixels.
            fps: Output frame rate.
            codec: Video codec (default: libx264).
            preset: Encoding preset (default: fast).
            crf: Constant Rate Factor for quality (default: 23).
            pixel_format: Output pixel format (default: yuv420p for compatibility).
            audio_source_path: Optional path to source media whose audio should be muxed.
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.preset = preset
        self.crf = crf
        self.pixel_format = pixel_format
        self.audio_source_path = audio_source_path
        
        self._process: Optional[subprocess.Popen] = None
        self._frame_count = 0
        self._stderr_buffer = []
        self._stderr_thread: Optional[threading.Thread] = None
    
    def _read_stderr(self) -> None:
        """Read stderr in a separate thread to prevent blocking."""
        if self._process and self._process.stderr:
            for line in iter(self._process.stderr.readline, b''):
                if line:
                    self._stderr_buffer.append(line.decode('utf-8', errors='ignore'))
    
    def _build_command(self) -> list:
        """Build the ffmpeg command."""
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",  # OpenCV uses BGR
            "-r", str(self.fps),
            "-i", "-",  # Read from stdin
        ]

        use_audio = bool(self.audio_source_path and os.path.exists(self.audio_source_path))
        if use_audio:
            # Input 1: source media for audio track
            cmd.extend(["-i", self.audio_source_path])
            # Keep rendered video and optionally map source audio if present.
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])

        cmd.extend([
            "-c:v", self.codec,
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", self.pixel_format,
        ])

        if use_audio:
            # Re-encode to AAC for wide MP4 compatibility.
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

        cmd.extend([
            "-movflags", "+faststart",  # Move metadata to beginning for streaming
            self.output_path,
        ])
        return cmd
    
    def open(self) -> None:
        """Open the ffmpeg process."""
        if self._process is not None:
            raise RuntimeError("FFmpeg writer is already open")
        
        cmd = self._build_command()
        
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        
        # Start stderr reading thread to prevent blocking
        self._stderr_buffer = []
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        
        self._frame_count = 0
        print(f"Started ffmpeg encoder for {self.output_path}")
    
    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to the video.
        
        Args:
            frame: BGR image array of shape (height, width, 3).
        """
        if self._process is None:
            raise RuntimeError("FFmpeg writer is not open. Call open() first.")
        
        if self._process.stdin is None:
            raise RuntimeError("FFmpeg stdin is not available")
        
        # Validate frame dimensions
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            raise ValueError(f"Frame size ({w}x{h}) doesn't match expected ({self.width}x{self.height})")
        
        # Write raw frame data
        self._process.stdin.write(frame.tobytes())
        self._frame_count += 1
    
    def close(self) -> None:
        """Close the ffmpeg process and finalize the video."""
        if self._process is None:
            return
        
        if self._process.stdin:
            self._process.stdin.close()
        
        # Wait for ffmpeg to finish
        self._process.wait()
        
        # Wait for stderr thread to finish
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=2.0)
        
        # Check for errors
        if self._process.returncode != 0:
            stderr = ''.join(self._stderr_buffer)
            print(f"FFmpeg error (return code {self._process.returncode}):")
            print(stderr[-1000:] if len(stderr) > 1000 else stderr)
        else:
            print(f"Video saved: {self.output_path} ({self._frame_count} frames)")
        
        self._process = None
        self._stderr_thread = None
    
    def __enter__(self) -> "FFmpegWriter":
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    @property
    def frame_count(self) -> int:
        """Return the number of frames written."""
        return self._frame_count

