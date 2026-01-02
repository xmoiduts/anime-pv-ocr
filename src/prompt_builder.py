import os
from pathlib import Path
from typing import Optional


def find_context_cards():
    """Look for context-cards/*.yaml in current and parent directories."""
    current_path = Path.cwd()
    for _ in range(3):
        card_dir = current_path / "context-cards"
        if card_dir.exists() and card_dir.is_dir():
            cards = list(card_dir.glob("*.yaml")) + list(card_dir.glob("*.yml"))
            if cards:
                return cards
        current_path = current_path.parent
    return []


def load_prompt_file(prompt_path: str) -> str:
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    print(f"Warning: Prompt file {prompt_path} not found. Using basic prompt.")
    return "Recognize the text in these images. Focus on lyrics and dialogue."


def build_prompt(prompt_path: str, spotter_prompt_path: Optional[str] = None) -> str:
    prompt_content = load_prompt_file(prompt_path)

    if spotter_prompt_path and os.path.exists(spotter_prompt_path):
        print(f"Adding spotter prompt from {spotter_prompt_path}")
        with open(spotter_prompt_path, "r", encoding="utf-8") as f:
            spotter_prompt_content = f.read()
            prompt_content += f"\n\n# Original Spotter Prompt\n{spotter_prompt_content}\n"

    context_cards = find_context_cards()
    if context_cards:
        print(f"Found {len(context_cards)} context cards. Adding to prompt.")
        prompt_content += "\n\n# Context Information (Background)\n"
        for card in context_cards:
            with open(card, "r", encoding="utf-8") as f:
                prompt_content += f"\n--- {card.name} ---\n{f.read()}\n"

    return prompt_content

