import argparse

from config_loader import load_config, load_env, load_yaml_config


def get_cli_args():
    load_env()

    yaml_config = load_yaml_config()

    parser = argparse.ArgumentParser(description="Gemini OCR for Anime PV/MV grids")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-t",
        "--task",
        default=None,
        help="Task name under config.task (default: spotter)",
    )
    group.add_argument(
        "-p",
        "--pipeline",
        default=None,
        help="Pipeline name under config.pipeline (runs tasks in order)",
    )

    parser.add_argument(
        "-i",
        "--input-file",
        help="Substring of the input file/folder name",
    )

    parser.add_argument(
        "--suffix",
        help="Optional suffix (e.g. timestamp) for locating specific result files",
    )

    parser.add_argument(
        "-r",
        "--range",
        help="Range of frames (e.g., '1-300' or '255-')",
    )

    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Override prompt markdown file for this run (applies to all tasks in a pipeline)",
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Override Gemini model name for this run",
    )

    parser.add_argument(
        "-b",
        "--base-url",
        default=None,
        help="Custom base URL for Gemini API",
    )

    parser.add_argument(
        "--hello",
        action="store_true",
        help="Test API connection with a simple 'Hi' prompt and exit",
    )

    args = parser.parse_args()

    if not args.task and not args.pipeline:
        args.task = "spotter"

    return {
        "args": args,
        "yaml_config": yaml_config,
        "config_json": load_config(),
    }

