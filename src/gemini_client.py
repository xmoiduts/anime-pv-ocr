import ctypes
import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tqdm import tqdm
import google.genai as genai
import google.genai.types as types
from PIL import Image

from config_loader import PricingTable

ANSI_GRAY = "\033[90m"
ANSI_RESET = "\033[0m"

# Monkey-patch Pydantic models to allow extra fields (thinking_effort etc.)
# because SDK version might trail behind API features.
for cls in [types.GenerateContentConfig, types.ThinkingConfig, types.Part]:
    if hasattr(cls, "model_config"):
        try:
            if cls.model_config is None:
                cls.model_config = {"extra": "allow"}
            else:
                cls.model_config["extra"] = "allow"
            if hasattr(cls, "model_rebuild"):
                cls.model_rebuild(force=True)
        except Exception:
            # Ignoring patch failures keeps runtime compatible across SDK versions.
            pass


@dataclass(frozen=True)
class GeminiCallResult:
    response_text: str
    thought_text: str = ""
    prompt_token_count: int = 0
    candidate_token_count: int = 0


def _enable_windows_ansi() -> bool:
    if os.name != "nt":
        return sys.stdout.isatty()
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        if handle == 0:
            return False
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
            return False
        enable_vt = 0x0004
        if mode.value & enable_vt:
            return True
        return kernel32.SetConsoleMode(handle, mode.value | enable_vt) != 0
    except Exception:
        return False


SUPPORTS_ANSI_COLOR = _enable_windows_ansi()

AUDIO_MIME_TYPES = {
    ".aac": "audio/aac",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
}

IMAGE_MIME_TYPES = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


def _stream_write(text: str, color: Optional[str] = None) -> None:
    if not text:
        return
    payload = text
    if color and SUPPORTS_ANSI_COLOR:
        payload = f"{color}{text}{ANSI_RESET}"
    try:
        sys.stdout.write(payload)
    except UnicodeEncodeError:
        escaped = text.encode("unicode_escape").decode("ascii")
        if color and SUPPORTS_ANSI_COLOR:
            escaped = f"{color}{escaped}{ANSI_RESET}"
        sys.stdout.write(escaped)
    sys.stdout.flush()


def _supports_thought_summaries(model_name: str, gemini_generation: Optional[float]) -> bool:
    if gemini_generation is not None:
        return gemini_generation >= 2.5
    lowered_name = model_name.lower()
    return lowered_name.startswith("gemini-3") or lowered_name.startswith("gemini-2.5")


def _normalize_path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _resolve_media_mime_type(path: str) -> str:
    extension = os.path.splitext(path)[1].lower()
    if extension in AUDIO_MIME_TYPES:
        return AUDIO_MIME_TYPES[extension]
    return IMAGE_MIME_TYPES.get(extension, "image/jpeg")


def call_gemini(
    api_key: str,
    model_name: str,
    prompt: str,
    image_paths: List[str],
    base_url: Optional[str] = None,
    media_resolution: Optional[str] = None,
    thinking_level: Optional[str] = None,
    exchange_rate: float = 7.2,
    gemini_generation: Optional[float] = None,
    pricing_table: Optional[PricingTable] = None,
    cancel_event: Optional[Any] = None,
    per_file_media_resolution: Optional[Dict[str, str]] = None,
) -> Optional[GeminiCallResult]:
    """
    Invoke Gemini API with streaming response while preserving the project's
    existing calling semantics. Pricing calculation is configurable through
    PricingTable; defaults align with prior hardcoded values when absent.
    """
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["http_options"] = {"base_url": base_url}
        print(f"Using custom base URL: {base_url}")
    else:
        print("Using default Google Gemini API endpoint.")

    client = genai.Client(**client_kwargs)

    contents = [prompt]

    is_gemini_3_plus = gemini_generation is not None and gemini_generation >= 3
    normalized_resolution_map = {
        _normalize_path_key(path): level
        for path, level in (per_file_media_resolution or {}).items()
        if level
    }
    use_per_part = is_gemini_3_plus and (media_resolution is not None or bool(normalized_resolution_map))

    print(f"Preparing to upload {len(image_paths)} media files (Per-part resolution: {use_per_part})...")
    for path in tqdm(image_paths, desc="Uploading media"):
        if cancel_event and cancel_event.is_set():
            raise KeyboardInterrupt("Gemini request cancelled before upload completed.")
        try:
            lower_path = path.lower()
            if lower_path.endswith(('.mp3', '.wav', '.m4a', '.aiff', '.aif', '.aac', '.ogg', '.flac')):
                mime_type = _resolve_media_mime_type(path)
                with open(path, "rb") as f:
                    media_bytes = f.read()
                part = types.Part.from_bytes(
                    data=media_bytes,
                    mime_type=mime_type,
                )
                contents.append(part)
                continue

            if use_per_part:
                mime_type = _resolve_media_mime_type(path)
                resolution_level = normalized_resolution_map.get(_normalize_path_key(path), media_resolution)
                with open(path, "rb") as f:
                    img_bytes = f.read()
                part_kwargs = dict(
                    data=img_bytes,
                    mime_type=mime_type,
                )
                if resolution_level is not None:
                    part_kwargs["media_resolution"] = {"level": resolution_level}
                part = types.Part.from_bytes(**part_kwargs)
                contents.append(part)
            else:
                img = Image.open(path)
                contents.append(img)
        except Exception as e:
            print(f"Error loading media {path}: {e}")

    print(f"Calling Gemini API ({model_name})...")
    usage = None
    full_text = ""
    thought_text = ""
    try:
        request_config = {}

        if media_resolution is not None and not use_per_part:
            request_config["media_resolution"] = media_resolution

        t_level = thinking_level
        if t_level is not None:
            request_config["thinking_level"] = t_level
        if _supports_thought_summaries(model_name, gemini_generation):
            request_config["thinking_config"] = {"include_thoughts": True}

        if not request_config:
            request_config = None

        active_stream = None
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=request_config,
        ):
            if cancel_event and cancel_event.is_set():
                raise KeyboardInterrupt("Gemini request cancelled while receiving response.")
            if chunk.usage_metadata:
                usage = chunk.usage_metadata

            if chunk.candidates:
                for cand in chunk.candidates:
                    if cand.content and cand.content.parts:
                        for part in cand.content.parts:
                            if part.text:
                                is_thought = bool(getattr(part, "thought", False))
                                if is_thought:
                                    if active_stream != "thought":
                                        if active_stream is not None:
                                            _stream_write("\n")
                                        _stream_write("[Thinking]\n", color=ANSI_GRAY)
                                        active_stream = "thought"
                                    _stream_write(part.text, color=ANSI_GRAY)
                                    thought_text += part.text
                                else:
                                    if active_stream == "thought":
                                        _stream_write("\n[Response]\n")
                                    active_stream = "answer"
                                    _stream_write(part.text)
                                    full_text += part.text

                            if hasattr(part, "thought_signature") and part.thought_signature:
                                sig = part.thought_signature
                                val = sig
                                if isinstance(sig, bytes):
                                    val = sig.hex()
                                elif hasattr(sig, "hex"):
                                    val = sig.hex()
                                sig_sha1 = hashlib.sha1(val.encode("utf-8")).hexdigest()
                                print(f"\n[Signature(SHA1): {sig_sha1}]", end="", flush=True)
        if active_stream is not None:
            _stream_write("\n\n")
        else:
            print("\n")

        if usage:
            prompt_tokens = usage.prompt_token_count or 0
            candidate_tokens = usage.candidates_token_count or 0

            # Configurable pricing; falls back to previous defaults.
            fallback_price = (0.10, 0.40)
            if pricing_table:
                input_price, output_price = pricing_table.resolve(model_name)
            else:
                input_price, output_price = fallback_price

            cost_usd = (prompt_tokens / 1_000_000) * input_price + (candidate_tokens / 1_000_000) * output_price
            cost_rmb = cost_usd * exchange_rate

            num_images = len(image_paths)
            if num_images > 0:
                avg_img_tokens = prompt_tokens / num_images
                try:
                    print(f"I {num_images} images | Avg ~{avg_img_tokens:.1f} tk/img (total prompt / image count)")
                except UnicodeEncodeError:
                    print(f"IMG {num_images} images | Avg ~{avg_img_tokens:.1f} tk/img (total prompt / image count)")

            print(f"↑↑ {prompt_tokens} tk  ↓ {candidate_tokens} tk  $ {cost_usd:.4f} ￥ {cost_rmb:.4f}")

        prompt_tokens = usage.prompt_token_count if usage and usage.prompt_token_count else 0
        candidate_tokens = usage.candidates_token_count if usage and usage.candidates_token_count else 0
        return GeminiCallResult(
            response_text=full_text,
            thought_text=thought_text,
            prompt_token_count=prompt_tokens,
            candidate_token_count=candidate_tokens,
        )

    except KeyboardInterrupt:
        print("\nGemini request interrupted.")
        raise
    except Exception as e:
        print(f"\nError calling Gemini API: {e}")
        return None

