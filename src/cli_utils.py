import os
import argparse
import yaml
import json
from dotenv import load_dotenv

def load_yaml_config(path="ocr-cli-config.yaml"):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_config(path="config.json"):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_cli_args():
    load_dotenv()
    
    # Load YAML config for pass defaults
    yaml_config = load_yaml_config()
    
    # Peek at task name to load correct defaults
    # This pre-parsing allows arguments to have dynamic defaults based on the task
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("-t", "--task", default="spotter")
    temp_args, _ = temp_parser.parse_known_args()
    task_name = temp_args.task
    
    task_config = yaml_config.get(task_name, {})
    if not task_config and task_name != "spotter":
        print(f"Warning: Task '{task_name}' not found in config. Falling back to empty config.")

    # Main Argument Parser
    parser = argparse.ArgumentParser(description="Gemini OCR for Anime PV/MV grids")
    
    parser.add_argument("-t", "--task", 
                        default="spotter", 
                        help=f"Task name in config (default: spotter). Current: {task_name}")
    
    parser.add_argument("-i", "--input-file", 
                        help="Substring of the input file/folder name")
    
    parser.add_argument("--suffix", 
                        help="Optional suffix (e.g. timestamp) for locating specific result files")

    parser.add_argument("-r", "--range", 
                        help="Range of frames (e.g., '1-300' or '255-')")
    
    parser.add_argument("-p", "--prompt-file", 
                        default=task_config.get("prompt_file", "prompts/gemini-filter-36grid.md"), 
                        help="Path to the prompt markdown file")
    
    parser.add_argument("-m", "--model", 
                        default=task_config.get("model", os.getenv("GEMINI_MODEL", "gemini-1.5-flash")), 
                        help="Gemini model name")
    
    parser.add_argument("-b", "--base-url",
                        default=None,
                        help="Custom base URL for Gemini API")
    
    parser.add_argument("--hello", 
                        action="store_true", 
                        help="Test API connection with a simple 'Hi' prompt and exit")
    
    args = parser.parse_args()
    
    # If user changed task via CLI but it's different from what we peeked
    if args.task != task_name:
        # We re-fetch task config just to return it correctly, 
        # though defaults for other args are already set based on the peeked task.
        # This is a limitation of this pattern, but acceptable.
        task_name = args.task
        task_config = yaml_config.get(task_name, {})

    # Finalize Base URL (CLI > YAML > ENV)
    base_url = args.base_url
    if base_url is None:
        base_url = task_config.get("base_url")
    if base_url is None:
        base_url = os.getenv("GEMINI_BASE_URL")
    
    # Return a consolidated config object
    return {
        "args": args,
        "task_name": task_name,
        "task_config": task_config,
        "yaml_config": yaml_config,
        "base_url": base_url,
        "config_json": load_config()
    }

