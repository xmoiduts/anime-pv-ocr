import sys
import time
from queue import Queue
from threading import Thread
from typing import Optional

from config_loader import RuntimeSettings, PricingTable
from gemini_client import call_gemini


def _read_input_async(queue: Queue):
    try:
        line = input()
        queue.put(line)
    except EOFError:
        queue.put(None)


def offer_post_run_chat(
    runtime_settings: RuntimeSettings,
    api_key: str,
    model_name: str,
    base_url: Optional[str],
    exchange_rate: float,
    gemini_generation: Optional[float],
    pricing_table: Optional[PricingTable],
    thinking_level: Optional[str] = None,
):
    """
    Provide a countdown-based exit. If the user types anything before timeout,
    run a single chat turn with Gemini. No conversation history is stored.
    """
    if not runtime_settings.enable_post_run_chat:
        return

    countdown = runtime_settings.idle_exit_seconds
    if countdown <= 0:
        return

    print(f"\nYou can enter a quick follow-up within {countdown}s. Press Enter to exit immediately.")
    queue: Queue = Queue()
    reader = Thread(target=_read_input_async, args=(queue,), daemon=True)
    reader.start()

    for remaining in range(countdown, 0, -1):
        if not queue.empty():
            break
        sys.stdout.write(f"\rAuto-exit in {remaining:02d}s... ")
        sys.stdout.flush()
        time.sleep(1)

    print("\r", end="")  # Clear countdown line

    if queue.empty():
        print("No follow-up received. Exiting.")
        return

    follow_up = queue.get()
    if follow_up is None or not follow_up.strip():
        print("Exit requested. Goodbye.")
        return

    print("Running one-step chat...")
    call_gemini(
        api_key,
        model_name,
        follow_up,
        [],
        base_url,
        media_resolution=None,
        thinking_level=thinking_level,
        exchange_rate=exchange_rate,
        gemini_generation=gemini_generation,
        pricing_table=pricing_table,
    )

