"""Cost (ch-13 observability support).

Token usage is only half of what a coding agent should show — the other half is
what those tokens cost. This maps a model id to a price and turns reported usage
into dollars, so the trace can display cost next to tokens. Local/unknown models are
free ($0.00), which is itself the teaching point: the *same* agent is free on a
local model and metered on a hosted one — only this table changes.

Prices are USD per 1,000,000 tokens and are illustrative — edit ``PRICES`` for
your provider. Matching is by substring, so "openai/gpt-4o-mini" matches the
"gpt-4o-mini" key.
"""

from __future__ import annotations

# (prompt, completion) USD per 1,000,000 tokens. Illustrative — edit for your provider.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "llama-3.1-70b": (0.12, 0.30),
}


def price_for(model: str | None) -> tuple[float, float]:
    """Return (prompt, completion) USD-per-1M for a model, or (0, 0) if unknown/local."""
    if not model:
        return (0.0, 0.0)
    needle = model.lower()
    for key, price in PRICES.items():
        if key in needle:
            return price
    return (0.0, 0.0)


def cost(model: str | None, prompt_tokens: int, completion_tokens: int = 0) -> float:
    """Dollar cost of a call. Local/unknown models cost 0.0."""
    p_rate, c_rate = price_for(model)
    return (prompt_tokens * p_rate + completion_tokens * c_rate) / 1_000_000


def cost_from_usage(model: str | None, usage: dict) -> float:
    """Cost from an OpenAI-style usage dict. When the prompt/completion split isn't
    reported, fall back to pricing ``total_tokens`` at the prompt rate."""
    if not usage:
        return 0.0
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    if not prompt and not completion:
        return cost(model, int(usage.get("total_tokens", 0) or 0))
    return cost(model, prompt, completion)


def format_cost(dollars: float) -> str:
    """Human display: $0.0000 (4 dp keeps sub-cent calls visible)."""
    return f"${dollars:.4f}"
