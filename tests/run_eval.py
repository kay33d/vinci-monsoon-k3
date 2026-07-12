"""Eval harness — runs the FULL router over mock_tasks.json.

Two modes:
  * DEFAULT (offline): Fireworks is MOCKED — no API key, no network, no GGUF.
    This is what CI's build gate and local `make test` use.
  * EVAL_LIVE=1: uses the REAL FireworksClient, reading FIREWORKS_API_KEY /
    FIREWORKS_BASE_URL / ALLOWED_MODELS from the environment. Used by the
    manual CI integration job to validate PLUMBING (auth, base URL, fallback
    behaviour) — it does NOT validate the real routing map, since the model
    IDs available before launch day are placeholders.

Usage:  python tests/run_eval.py            # offline, mocked
        EVAL_LIVE=1 python tests/run_eval.py  # live client

The summary printed at the end is non-sensitive: counts and booleans only,
never env var values.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A plausible-looking allowed list purely for exercising role resolution
# offline. No-op when the environment already provides ALLOWED_MODELS.
os.environ.setdefault(
    "ALLOWED_MODELS",
    "accounts/fireworks/models/minimax-m3,"
    "accounts/fireworks/models/kimi-k2p7-code",
)

from src.local_models.loader import get_local_model      # noqa: E402
from src.router.dispatch import Router                   # noqa: E402

MOCK_TASKS = ROOT / "tests" / "mock_tasks.json"


class MockFireworks:
    """Stub Fireworks client: canned answer + simulated token accounting.

    SLOW_MOCK=1 makes each call sleep ~2s (simulating real API latency) and
    the FIRST minimax call fail (exercising the alternate-model path), so the
    per-task timing/budget logic can be observed offline.
    """

    def __init__(self):
        self.total_tokens = 0
        self.calls = 0
        self.last_finish_reason = None
        self.slow = os.environ.get("SLOW_MOCK") == "1"
        self._failed_once = False

    @property
    def configured(self) -> bool:
        return True

    def chat(self, model, system, user, max_tokens=512, timeout=25.0,
             reasoning_effort="none") -> str:
        import time as _time

        from src.api_clients.fireworks import FireworksError

        if self.slow:
            _time.sleep(min(2.0, timeout))
            if "minimax" in model and not self._failed_once:
                self._failed_once = True
                raise FireworksError("simulated transient failure (SLOW_MOCK)")
        # Rough token simulation: ~1 token per 4 chars of input + 60 output.
        self.total_tokens += (len(system) + len(user)) // 4 + 60
        self.calls += 1
        self.last_finish_reason = "stop"
        return f"[mock answer from {model.split('/')[-1]}]"


def run_offline_eval(output_path: Path | None = None, live: bool | None = None) -> list[dict]:
    if live is None:
        live = os.environ.get("EVAL_LIVE") == "1"

    with open(MOCK_TASKS, "r", encoding="utf-8") as fh:
        tasks = json.load(fh)

    if live:
        from src.api_clients.fireworks import FireworksClient

        client = FireworksClient()
        # Booleans only — never print the values themselves.
        print(f"live mode: fireworks_configured={str(client.configured).lower()}")
    else:
        client = MockFireworks()
    router = Router(get_local_model(), client)

    results = []
    rows = defaultdict(lambda: {"local": 0, "remote": 0, "model": "-"})
    route_counts = {"local": 0, "remote": 0, "local_fallback": 0}
    errors: list[str] = []
    timing_rows: list[tuple] = []
    for task in tasks:
        answer, meta = router.route(task["prompt"])
        results.append({"task_id": task["task_id"], "answer": answer})
        t = meta.get("timing", {})
        timing_rows.append((
            task["task_id"], meta["decision"]["intent"], meta.get("route"),
            t.get("primary_secs", 0), t.get("alternate_fired", False),
            t.get("alternate_secs", 0), t.get("total_secs", 0),
        ))
        route = meta.get("route", "local")
        route_counts[route] = route_counts.get(route, 0) + 1
        if meta.get("error"):
            # Truncated failure reason (HTTP status etc.) — contains no secrets.
            errors.append(f"{task['task_id']}: {meta['error'][:120]}")
        cat = meta["decision"]["intent"]
        if route == "remote":
            rows[cat]["remote"] += 1
            rows[cat]["model"] = meta["model"].split("/")[-1]
        else:
            rows[cat]["local"] += 1

    # --- report ---------------------------------------------------------
    print(f"\nmode: {'LIVE' if live else 'mock'} | local backend: {router.local.backend}")
    print(f"{'category':<20} {'local':>5} {'remote':>6}  remote model")
    print("-" * 60)
    for cat in sorted(rows):
        r = rows[cat]
        print(f"{cat:<20} {r['local']:>5} {r['remote']:>6}  {r['model']}")
    print("-" * 60)
    real_success = live and route_counts["remote"] > 0
    print(
        f"SUMMARY mode={'live' if live else 'mock'} tasks={len(results)} "
        f"local={route_counts['local']} rule={route_counts.get('local_rule', 0)} "
        f"model={route_counts.get('local_model', 0)} "
        f"remote={route_counts['remote']} "
        f"fallback={route_counts['local_fallback']} "
        f"calls={client.calls} tokens={client.total_tokens} "
        f"real_fireworks_success={str(real_success).lower()}"
    )
    for line in errors:
        print(f"remote failure -> handled fallback: {line}")

    # --- per-task timing table -------------------------------------------
    print(f"\n{'task_id':<15} {'category':<19} {'route':<15} "
          f"{'primary_s':>9} {'alt?':>5} {'alt_s':>6} {'total_s':>8}")
    print("-" * 82)
    for tid, cat, route, p, alt, alts, tot in timing_rows:
        print(f"{tid:<15} {cat:<19} {route:<15} {p:>9} "
              f"{'yes' if alt else 'no':>5} {alts:>6} {tot:>8}")
    slowest = sorted(timing_rows, key=lambda r: r[6], reverse=True)[:3]
    print("3 slowest:", ", ".join(f"{r[0]} ({r[6]}s)" for r in slowest))
    print(f"alternate fired on {sum(1 for r in timing_rows if r[4])} task(s)\n")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=1)
    return results


if __name__ == "__main__":
    run_offline_eval(ROOT / "tests" / "out_results.json")
