"""ch-02 — History.

Capability: the harness retains the conversation, so the model can use earlier
turns. Mocked for determinism; the live recall test is `uv run accept ch-02`.
"""

from unittest.mock import patch

import harness.agent as agent_mod
from model import LLMResponse


def test_history_is_replayed():
    captured: list[list[str]] = []

    def fake_chat(messages, **kwargs):
        captured.append([m["content"] for m in messages])
        return LLMResponse(content="ok")

    with patch.object(agent_mod, "chat", side_effect=fake_chat):
        a = agent_mod.Agent()
        a.send("Your name is Gemma.")
        a.send("What is your name?")

    # The second call sees the full history: user1, assistant1, user2.
    assert captured[1] == ["Your name is Gemma.", "ok", "What is your name?"]


def test_model_can_use_history():
    # Simulate a model that answers from whatever context it is given.
    def fake_chat(messages, **kwargs):
        joined = " ".join(m["content"] for m in messages)
        name = "Gemma" if "Gemma" in joined else "unknown"
        return LLMResponse(content=f"Your name is {name}.")

    with patch.object(agent_mod, "chat", side_effect=fake_chat):
        a = agent_mod.Agent()
        a.send("Your name is Gemma.")
        out = a.send("What is your name?")

    assert "Gemma" in out
