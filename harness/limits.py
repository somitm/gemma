"""Door control (ch-06).

Hard per-item size limits, applied before anything enters the prompt. A single
huge file or tool output can drown the window (distraction / confusion /
poisoning); clamping each item at the door is the cheapest defense.
"""

from __future__ import annotations

from harness.harness_config import CONFIG

MAX_ITEM_CHARS = CONFIG.max_item_chars  # re-export; the value lives in the editable surface


def clamp(text: str, max_chars: int = MAX_ITEM_CHARS) -> str:
    """Truncate an item to ``max_chars``, with a marker noting what was dropped."""
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return f"{text[:max_chars]}\n…[truncated {dropped} chars]"
