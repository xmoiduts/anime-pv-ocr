import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import yaml
from dotenv import load_dotenv


def load_yaml_config(path: str = "ocr-cli-config.yaml") -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str = "config.json") -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_env() -> None:
    """Load environment variables once for the process."""
    load_dotenv()


@dataclass
class PricingTable:
    default_input_per_million: float
    default_output_per_million: float
    per_model: Dict[str, Tuple[float, float]]

    def resolve(self, model_name: str) -> Tuple[float, float]:
        """Return (input_price, output_price) per million tokens."""
        if not model_name:
            return self.default_input_per_million, self.default_output_per_million
        model_key = model_name.lower()
        for key, value in self.per_model.items():
            if key in model_key:
                return value
        return self.default_input_per_million, self.default_output_per_million


def build_pricing_table(yaml_config: Dict) -> PricingTable:
    pricing_cfg = yaml_config.get("pricing", {})
    default_input = pricing_cfg.get("default_input_per_million", 0.10)
    default_output = pricing_cfg.get("default_output_per_million", 0.40)

    per_model: Dict[str, Tuple[float, float]] = {}
    for name, info in yaml_config.get("model", {}).items():
        price = info.get("price")
        if not price:
            continue
        input_price = price.get("input")
        output_price = price.get("output")
        if input_price is None or output_price is None:
            continue
        per_model[name.lower()] = (float(input_price), float(output_price))

    return PricingTable(
        default_input_per_million=float(default_input),
        default_output_per_million=float(default_output),
        per_model=per_model,
    )


def get_exchange_rate(yaml_config: Dict) -> float:
    fee_config = yaml_config.get("fee", {})
    return float(fee_config.get("exchange_rate", 7.2))


@dataclass
class RuntimeSettings:
    idle_exit_seconds: int = 8
    enable_post_run_chat: bool = True


def get_runtime_settings(yaml_config: Dict) -> RuntimeSettings:
    runtime_cfg = yaml_config.get("runtime", {})
    return RuntimeSettings(
        idle_exit_seconds=int(runtime_cfg.get("idle_exit_seconds", 8)),
        enable_post_run_chat=bool(runtime_cfg.get("enable_post_run_chat", True)),
    )

