"""Live acceptance gate: run the real agent against a real model and assert the
chapter's actual capability. REQUIRED before a code commit.

  uv run accept ch-02
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, os.getcwd())
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: uv run accept ch-NN")
        return 2
    chapter = argv[0]

    from tasks.checks import ACCEPTANCE

    check = ACCEPTANCE.get(chapter)
    if check is None:
        print(f"no live acceptance check registered for '{chapter}'")
        return 2

    # Resolve through the provider so the banner reports what the run actually
    # targets — including values that live only in .env (loaded by from_env).
    from model.provider import Provider

    cfg = Provider.from_env()
    print(f"== live acceptance: {chapter}  (model={cfg.model} @ {cfg.base_url}) ==", flush=True)
    try:
        ok = bool(check())
    except Exception as exc:  # noqa: BLE001
        print(f"ACCEPT ERROR: {exc}")
        return 1
    print("\nACCEPT OK" if ok else "\nACCEPT FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
