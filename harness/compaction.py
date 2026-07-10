"""Context management — compaction (ch-06).

When the conversation outgrows a budget, summarize the middle into one note and
keep the head and tail intact (models read the start and end most reliably —
"present is not the same as used"). A good summary preserves what the *next*
turn needs, not merely fewer words.
"""

from __future__ import annotations

import json

from model import Provider, chat

COMPACTION_PROMPT = (
    "You are a context summarizer. Compress the conversation below into a short "
    "checkpoint another model will use to continue. Preserve, verbatim, every "
    "concrete fact, code, name, decision, file path, and the current goal and "
    "next step. Drop chit-chat. Be terse but lose nothing the next turn needs."
)


def estimate_tokens(messages: list[dict]) -> int:
    """Cheap ~4-chars-per-token estimate over message contents and tool-call args.

    Tool-call arguments (e.g. a whole file body in a ``write_file`` call) are part
    of the window too — count them, or a tool-heavy turn under-measures and the
    compaction door fires late, exactly when the window is fullest.
    """
    total = 0
    for m in messages:
        total += len(str(m.get("content", "") or ""))
        for tc in m.get("tool_calls") or []:
            total += len(json.dumps(tc))
    return total // 4


def _is_tool_call_assistant(m: dict) -> bool:
    return m.get("role") == "assistant" and bool(m.get("tool_calls"))


def _clean_cut(messages: list[dict], i: int) -> bool:
    """True if splitting ``messages`` at index ``i`` keeps every tool-call group whole.

    An OpenAI-compatible payload requires each assistant ``tool_calls`` message to
    be immediately followed by its ``tool`` results. A cut is unsafe if it would
    put a ``tool`` result on the right without its assistant (``messages[i]`` is a
    tool), or leave an assistant with dangling ``tool_calls`` on the left
    (``messages[i-1]`` still expects results at ``i``).
    """
    if i <= 0 or i >= len(messages):
        return True
    if messages[i].get("role") == "tool":
        return False
    if _is_tool_call_assistant(messages[i - 1]):
        return False
    return True


def compact(
    messages: list[dict],
    *,
    keep_head: int = 2,
    keep_tail: int = 4,
    model: str | None = None,
    provider: Provider | None = None,
) -> list[dict]:
    """Summarize the middle of ``messages`` into a single note; keep head + tail.

    Head/tail boundaries are snapped to whole-turn cuts so compaction never
    orphans a tool result from its assistant ``tool_calls`` (which the API rejects
    with a 400). If snapping leaves nothing safe to summarize, the history is
    returned unchanged — better a large window this turn than a corrupt one.
    """
    if len(messages) <= keep_head + keep_tail:
        return messages

    head_end = keep_head
    while not _clean_cut(messages, head_end):
        head_end -= 1
    tail_start = len(messages) - keep_tail
    while not _clean_cut(messages, tail_start):
        tail_start -= 1

    if head_end >= tail_start:  # snapping erased the middle — nothing safe to compact
        return messages

    head = messages[:head_end]
    tail = messages[tail_start:]
    middle = messages[head_end:tail_start]

    transcript = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in middle)
    summary = chat(
        [
            {"role": "system", "content": COMPACTION_PROMPT},
            {"role": "user", "content": transcript},
        ],
        model=model,
        provider=provider,  # summarize through the same endpoint the turn uses
        max_tokens=512,
    ).content

    note = {"role": "system", "content": f"[summary of earlier conversation]\n{summary}"}
    return head + [note] + tail
