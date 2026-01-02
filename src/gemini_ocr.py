import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from cli_utils import get_cli_args
from config_loader import PricingTable, build_pricing_table, get_exchange_rate, get_runtime_settings
from gemini_client import call_gemini
from post_run_chat import offer_post_run_chat
from prompt_builder import build_prompt
from task_resolver import TaskInputs, resolve_task_inputs


@dataclass
class TaskPlan:
    name: str
    prompt: str
    inputs: TaskInputs
    save_results: bool = False


class WorkflowRuntime:
    """
    Thin workflow scaffold so we can pivot to multi-step agent/workflow
    execution later. Today it runs a single step, but the structure can host
    a list of TaskPlan items or a DAG in the future.
    """

    def __init__(
        self,
        args,
        task_name: str,
        task_config: dict,
        yaml_config: dict,
        base_url: Optional[str],
        config_json: dict,
        api_key: str,
        model_info: dict,
    ):
        self.args = args
        self.task_name = task_name
        self.task_config = task_config
        self.yaml_config = yaml_config
        self.base_url = base_url
        self.config_json = config_json
        self.api_key = api_key
        self.model_info = model_info

    def build_plan(self) -> TaskPlan:
        inputs = resolve_task_inputs(self.args, self.task_name, self.task_config, self.config_json)
        prompt_content = build_prompt(
            self.args.prompt_file,
            self.task_config.get("spotter_prompt_file"),
        )
        save_results = self.task_name == "spotter"
        return TaskPlan(
            name=self.task_name,
            prompt=prompt_content,
            inputs=inputs,
            save_results=save_results,
        )

    def execute_plan(
        self,
        plan: TaskPlan,
        pricing_table: PricingTable,
        exchange_rate: float,
        gemini_generation: Optional[float],
    ) -> Optional[str]:
        response_text = call_gemini(
            self.api_key,
            self.args.model,
            plan.prompt,
            plan.inputs.image_paths,
            self.base_url,
            media_resolution=self.task_config.get("media_resolution"),
            thinking_level=self.task_config.get("thinking_level"),
            exchange_rate=exchange_rate,
            gemini_generation=gemini_generation,
            pricing_table=pricing_table,
        )

        if response_text and plan.save_results:
            self._save_spotter_results(plan.inputs.folder_path, response_text)

        return response_text

    @staticmethod
    def _extract_yaml_block(text: str) -> str:
        if "```yaml" in text:
            match = re.search(r"```yaml\n(.*?)\n```", text, re.DOTALL)
            if match:
                return match.group(1)
        if "```" in text:
            match = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
            if match:
                return match.group(1)
        return text

    def _save_spotter_results(self, folder_path: Optional[str], response_text: str) -> None:
        if not folder_path:
            return

        results_dir = Path(folder_path) / "spotter-results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"{timestamp}.yaml"

        yaml_content = self._extract_yaml_block(response_text)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"Saved spotter results to {output_file}")


def main():
    cli_data = get_cli_args()
    args = cli_data["args"]
    task_name = cli_data["task_name"]
    task_config = cli_data["task_config"]
    yaml_config = cli_data["yaml_config"]
    base_url = cli_data["base_url"]
    config_json = cli_data["config_json"]

    pricing_table = build_pricing_table(yaml_config)
    runtime_settings = get_runtime_settings(yaml_config)
    exchange_rate = get_exchange_rate(yaml_config)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file or environment variables.")
        sys.exit(1)

    model_configs = yaml_config.get("model", {})
    model_info = model_configs.get(args.model, {})
    gemini_generation = model_info.get("gemini-generation")

    if args.hello:
        print(f"Running hello world test (Task: {task_name})")
        print(f"Model: {args.model} (Generation: {gemini_generation})")
        call_gemini(
            api_key,
            args.model,
            "Hi",
            [],
            base_url,
            media_resolution=task_config.get("media_resolution"),
            thinking_level=task_config.get("thinking_level"),
            exchange_rate=exchange_rate,
            gemini_generation=gemini_generation,
            pricing_table=pricing_table,
        )
        sys.exit(0)

    runtime = WorkflowRuntime(
        args=args,
        task_name=task_name,
        task_config=task_config,
        yaml_config=yaml_config,
        base_url=base_url,
        config_json=config_json,
        api_key=api_key,
        model_info=model_info,
    )

    plan = runtime.build_plan()
    response_text = runtime.execute_plan(plan, pricing_table, exchange_rate, gemini_generation)

    offer_post_run_chat(
        runtime_settings,
        api_key,
        args.model,
        base_url,
        exchange_rate,
        gemini_generation,
        pricing_table,
        thinking_level=task_config.get("thinking_level"),
    )


if __name__ == "__main__":
    main()

