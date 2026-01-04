import hashlib
from typing import List, Optional

from tqdm import tqdm
import google.genai as genai
import google.genai.types as types
from PIL import Image

from config_loader import PricingTable

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
) -> Optional[str]:
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
    use_per_part = is_gemini_3_plus and media_resolution is not None

    print(f"Preparing to upload {len(image_paths)} media files (Per-part resolution: {use_per_part})...")
    for path in tqdm(image_paths, desc="Uploading media"):
        try:
            lower_path = path.lower()
            if lower_path.endswith(('.mp3', '.wav', '.aiff', '.aac', '.ogg', '.flac')):
                mime_type = "audio/mp3"
                if lower_path.endswith('.wav'): mime_type = "audio/wav"
                elif lower_path.endswith('.aac'): mime_type = "audio/aac"
                elif lower_path.endswith('.ogg'): mime_type = "audio/ogg"
                elif lower_path.endswith('.flac'): mime_type = "audio/flac"
                
                with open(path, "rb") as f:
                    media_bytes = f.read()
                part = types.Part.from_bytes(
                    data=media_bytes,
                    mime_type=mime_type,
                )
                contents.append(part)
                continue

            if use_per_part:
                with open(path, "rb") as f:
                    img_bytes = f.read()
                part = types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/jpeg",
                    media_resolution={"level": media_resolution},
                )
                contents.append(part)
            else:
                img = Image.open(path)
                contents.append(img)
        except Exception as e:
            print(f"Error loading media {path}: {e}")

    print(f"Calling Gemini API ({model_name})...")
    usage = None
    full_text = ""
    try:
        request_config = {}

        if media_resolution is not None and not use_per_part:
            request_config["media_resolution"] = media_resolution

        t_level = thinking_level
        if t_level is not None:
            request_config["thinking_level"] = t_level

        if not request_config:
            request_config = None

        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=request_config,
        ):
            if chunk.usage_metadata:
                usage = chunk.usage_metadata

            if chunk.candidates:
                for cand in chunk.candidates:
                    if cand.content and cand.content.parts:
                        for part in cand.content.parts:
                            if part.text:
                                print(part.text, end="", flush=True)
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

            print(f"^^ {prompt_tokens} tk  v {candidate_tokens} tk  $ {cost_usd:.4f} Y {cost_rmb:.4f}")

        return full_text

    except Exception as e:
        print(f"\nError calling Gemini API: {e}")
        return None

