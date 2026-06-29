"""The model seam has a working second implementation.

The provider seam is only honest once it has more than one implementation. This
drives a real ``Agent`` end-to-end against the ``fake`` provider — no network, no
monkeypatching of ``chat`` — and asserts the fake responder's reply comes back.
Deterministic and fully offline.
"""

from __future__ import annotations

from harness.agent import Agent
from model import fake


def test_agent_runs_offline_through_fake_provider():
    a = Agent(provider=fake(scripted=lambda msgs: "PONG"))
    assert a.send("ping") == "PONG"
