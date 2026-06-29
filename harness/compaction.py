"""Context management — compaction (ch-06).

When the conversation outgrows a budget, summarize the middle into one note and
keep the head and tail intact (models read the start and end most reliably —
"present is not the same as used"). A good summary preserves what the *next*
turn needs, not merely fewer words.
"""

from __future__ import annotations

from model import chat

COMPACTION_PROMPT = (
    "You are a context summarizer. Compress the conversation below into a short "
    "checkpoint another model will use to continue. Preserve, verbatim, every "
    "concrete fact, code, name, decision, file path, and the current goal and "
    "next step. Drop chit-chat. Be terse but lose nothing the next turn needs."
)


def estimate_tokens(messages: list[dict]) -> int:
    """Cheap ~4-chars-per-token estimate over message contents."""
    return sum(len(str(m.get("content", "") or "")) for m in messages) // 4


def compact(
    messages: list[dict],
    *,
    keep_head: int = 2,
    keep_tail: int = 4,
    model: str | None = None,
) -> list[dict]:
    """Summarize the middle of ``messages`` into a single note; keep head + tail."""
    if len(messages) <= keep_head + keep_tail:
        return messages

    head = messages[:keep_head]
    tail = messages[-keep_tail:]
    middle = messages[keep_head:-keep_tail]

    transcript = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in middle)
    summary = chat(
        [
            {"role": "system", "content": COMPACTION_PROMPT},
            {"role": "user", "content": transcript},
        ],
        model=model,
        max_tokens=512,
    ).content

    note = {"role": "system", "content": f"[summary of earlier conversation]\n{summary}"}
    return head + [note] + tail
