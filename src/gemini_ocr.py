import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from cli_utils import get_cli_args
from config_loader import PricingTable, build_pricing_table, get_exchange_rate, get_runtime_settings
from gemini_client import call_gemini
from post_run_chat import offer_post_run_chat
from prompt_builder import build_prompt
from task_resolver import TaskInputs, resolve_task_inputs
from task_outputs import TaskOutputHandler, get_output_handler


@dataclass
class TaskPlan:
    name: str
    prompt: str
    inputs: TaskInputs
    task_config: Dict
    output_handler: Optional[TaskOutputHandler] = None


class WorkflowRuntime:
    """
    Thin workflow scaffold so we can pivot to multi-step agent/workflow
    execution later. Today it runs one or more steps (task or pipeline),
    but the structure can host a list of TaskPlan items or a DAG in the future.
    """

    def __init__(
        self,
        args,
        yaml_config: dict,
        config_json: dict,
        api_key: str,
    ):
        self.args = args
        self.yaml_config = yaml_config
        self.config_json = config_json
        self.api_key = api_key
        self.task_configs = yaml_config.get("task", {})
        self.pipeline_configs = yaml_config.get("pipeline", {})
        self.model_configs = yaml_config.get("model", {})

    def resolve_task_sequence(self) -> List[str]:
        if self.args.pipeline:
            pipeline_cfg = self.pipeline_configs.get(self.args.pipeline)
            if not pipeline_cfg:
                available = ", ".join(self.pipeline_configs.keys()) or "none"
                print(f"Error: Pipeline '{self.args.pipeline}' not found. Available: {available}")
                sys.exit(1)
            task_sequence = pipeline_cfg.get("tasks") or []
            if not task_sequence:
                print(f"Error: Pipeline '{self.args.pipeline}' has no tasks configured.")
                sys.exit(1)
            missing = [t for t in task_sequence if t not in self.task_configs]
            if missing:
                print(
                    f"Error: Pipeline '{self.args.pipeline}' references undefined tasks: {', '.join(missing)}"
                )
                sys.exit(1)
            return task_sequence

        task_name = self.args.task or "spotter"
        if task_name not in self.task_configs and task_name != "spotter":
            print(f"Warning: Task '{task_name}' not found in config.task. Falling back to empty config.")
        return [task_name]

    def resolve_model(self, task_config: dict) -> str:
        return self.args.model or task_config.get("model") or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    def resolve_base_url(self, task_config: dict) -> Optional[str]:
        return self.args.base_url or task_config.get("base_url") or os.getenv("GEMINI_BASE_URL")

    def build_plan(self, task_name: str) -> TaskPlan:
        task_config = self.task_configs.get(task_name, {})
        if not task_config:
            print(f"Warning: Task '{task_name}' not found in config.task. Using empty config.")

        prompt_path = self.args.prompt_file or task_config.get("prompt_file") or "prompts/filter/gemini-filter-36grid.md"
        inputs = resolve_task_inputs(self.args, task_name, task_config, self.config_json)
        prompt_content = build_prompt(
            prompt_path,
            task_config.get("spotter_prompt_file"),
        )
        output_handler = get_output_handler(task_name)
        return TaskPlan(
            name=task_name,
            prompt=prompt_content,
            inputs=inputs,
            task_config=task_config,
            output_handler=output_handler,
        )

    def execute_plan(
        self,
        plan: TaskPlan,
        pricing_table: PricingTable,
        exchange_rate: float,
    ) -> Optional[str]:
        task_config = plan.task_config or {}
        model_name = self.resolve_model(task_config)
        base_url = self.resolve_base_url(task_config)
        gemini_generation = self.model_configs.get(model_name, {}).get("gemini-generation")

        response_text = call_gemini(
            self.api_key,
            model_name,
            plan.prompt,
            plan.inputs.image_paths,
            base_url,
            media_resolution=task_config.get("media_resolution"),
            thinking_level=task_config.get("thinking_level"),
            exchange_rate=exchange_rate,
            gemini_generation=gemini_generation,
            pricing_table=pricing_table,
        )

        if response_text and plan.output_handler:
            plan.output_handler(plan.inputs, response_text)

        return response_text


def main():
    cli_data = get_cli_args()
    args = cli_data["args"]
    yaml_config = cli_data["yaml_config"]
    config_json = cli_data["config_json"]

    pricing_table = build_pricing_table(yaml_config)
    runtime_settings = get_runtime_settings(yaml_config)
    exchange_rate = get_exchange_rate(yaml_config)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file or environment variables.")
        sys.exit(1)

    runtime = WorkflowRuntime(
        args=args,
        yaml_config=yaml_config,
        config_json=config_json,
        api_key=api_key,
    )

    task_sequence = runtime.resolve_task_sequence()

    if args.hello:
        hello_task = task_sequence[0]
        hello_config = runtime.task_configs.get(hello_task, {})
        hello_model = runtime.resolve_model(hello_config)
        hello_base_url = runtime.resolve_base_url(hello_config)
        hello_generation = runtime.model_configs.get(hello_model, {}).get("gemini-generation")
        print(f"Running hello world test (Task: {hello_task})")
        print(f"Model: {hello_model} (Generation: {hello_generation})")
        call_gemini(
            api_key,
            hello_model,
            "Hi",
            [],
            hello_base_url,
            media_resolution=hello_config.get("media_resolution"),
            thinking_level=hello_config.get("thinking_level"),
            exchange_rate=exchange_rate,
            gemini_generation=hello_generation,
            pricing_table=pricing_table,
        )
        sys.exit(0)

    if args.pipeline:
        print(f"Running pipeline '{args.pipeline}' -> {task_sequence}")

    last_plan: Optional[TaskPlan] = None
    for task_name in task_sequence:
        plan = runtime.build_plan(task_name)
        last_plan = plan
        runtime.execute_plan(plan, pricing_table, exchange_rate)

    if not last_plan:
        print("Error: No tasks to run.")
        sys.exit(1)
    last_config = last_plan.task_config or {}
    last_model = runtime.resolve_model(last_config)
    last_base_url = runtime.resolve_base_url(last_config)
    last_generation = runtime.model_configs.get(last_model, {}).get("gemini-generation")

    offer_post_run_chat(
        runtime_settings,
        api_key,
        last_model,
        last_base_url,
        exchange_rate,
        last_generation,
        pricing_table,
        thinking_level=last_config.get("thinking_level"),
    )


if __name__ == "__main__":
    main()

