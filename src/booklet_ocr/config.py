import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from config_loader import PricingTable, build_pricing_table, get_exchange_rate, load_env, load_yaml_config

MODULE_NAME = "booklet-ocr"
MODULE_DIR = Path(__file__).resolve().parent
SRC_DIR = MODULE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_LOCAL_CONFIG = MODULE_DIR / "config.yaml"
MAIN_CONFIG_PATH = PROJECT_ROOT / "ocr-cli-config.yaml"


def resolve_project_path(path: str) -> Path:
    if os.path.isabs(path):
        return Path(path)
    return (PROJECT_ROOT / path).resolve()


def _normalize_extension(ext: str) -> str:
    normalized = (ext or "").strip().lower()
    if not normalized:
        return ""
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


@dataclass
class LocalConfig:
    model: str
    prompt_file: str
    input_source_folder: str
    supported_media_extensions: List[str]  # required fields above, optional below
    finishing_ocr_folder: str = ""
    done_folder: str = ""
    default_lookback_minutes: int = 30
    output_root: str = "outputs/booklet-ocr"
    raw_log_suffix: str = "response_raw.log"
    yaml_suffix: str = "lyrics.yaml"
    lrc_suffix: str = "lyrics.lrc"
    pixel_threshold_for_ultra_high: int = 6_000_000
    warn_file_count_over: int = 20
    small_image_media_resolution: str = "MEDIA_RESOLUTION_HIGH"
    large_image_media_resolution: str = "MEDIA_RESOLUTION_ULTRA_HIGH"
    thinking_level: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class RuntimeContext:
    project_root: Path
    module_dir: Path
    main_config_path: Path
    local_config_path: Path
    yaml_config: Dict[str, Any]
    model_configs: Dict[str, Any]
    local_config: LocalConfig
    pricing_table: PricingTable
    exchange_rate: float
    api_key: str
    prompt_path: Path


def _load_local_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_local_config(path: Optional[str] = None) -> tuple[LocalConfig, Path]:
    config_path = Path(path).resolve() if path else DEFAULT_LOCAL_CONFIG
    raw = _load_local_yaml(config_path)
    extensions = [
        normalized
        for normalized in (_normalize_extension(item) for item in raw.get("supported_media_extensions", []))
        if normalized
    ]
    local_config = LocalConfig(
        model=str(raw.get("model", "")).strip(),
        prompt_file=str(raw.get("prompt_file", "")).strip(),
        input_source_folder=str(raw.get("input_source_folder", "")).strip(),
        supported_media_extensions=extensions or [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"],
        output_root=str(raw.get("output_root", "outputs/booklet-ocr")).strip() or "outputs/booklet-ocr",
        raw_log_suffix=str(raw.get("raw_log_suffix", "response_raw.log")).strip() or "response_raw.log",
        yaml_suffix=str(raw.get("yaml_suffix", "lyrics.yaml")).strip() or "lyrics.yaml",
        lrc_suffix=str(raw.get("lrc_suffix", "lyrics.lrc")).strip() or "lyrics.lrc",
        pixel_threshold_for_ultra_high=max(1, int(raw.get("pixel_threshold_for_ultra_high", 6_000_000))),
        warn_file_count_over=max(1, int(raw.get("warn_file_count_over", 20))),
        small_image_media_resolution=str(
            raw.get("small_image_media_resolution", "MEDIA_RESOLUTION_HIGH")
        ).strip()
        or "MEDIA_RESOLUTION_HIGH",
        large_image_media_resolution=str(
            raw.get("large_image_media_resolution", "MEDIA_RESOLUTION_ULTRA_HIGH")
        ).strip()
        or "MEDIA_RESOLUTION_ULTRA_HIGH",
        thinking_level=raw.get("thinking_level"),
        base_url=raw.get("base_url"),
        finishing_ocr_folder=str(raw.get("finishing_ocr_folder", "")).strip() or "",
        done_folder=str(raw.get("done_folder", "")).strip() or "",
        default_lookback_minutes=max(1, int(raw.get("default_lookback_minutes", 30))),
    )
    return local_config, config_path


def load_runtime_context(local_config_path: Optional[str] = None) -> RuntimeContext:
    load_env()

    yaml_config = load_yaml_config(str(MAIN_CONFIG_PATH))
    local_config, resolved_local_config_path = load_local_config(local_config_path)
    if not local_config.model:
        raise RuntimeError(
            f"'model' is required in {resolved_local_config_path}. "
            "The model name should reference one defined under the main project's model catalog."
        )
    if not local_config.prompt_file:
        raise RuntimeError(f"'prompt_file' is required in {resolved_local_config_path}.")
    if not local_config.input_source_folder:
        raise RuntimeError(f"'input_source_folder' is required in {resolved_local_config_path}.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment or .env.")

    return RuntimeContext(
        project_root=PROJECT_ROOT,
        module_dir=MODULE_DIR,
        main_config_path=MAIN_CONFIG_PATH,
        local_config_path=resolved_local_config_path,
        yaml_config=yaml_config,
        model_configs=yaml_config.get("model", {}),
        local_config=local_config,
        pricing_table=build_pricing_table(yaml_config),
        exchange_rate=get_exchange_rate(yaml_config),
        api_key=api_key,
        prompt_path=resolve_project_path(local_config.prompt_file),
    )
