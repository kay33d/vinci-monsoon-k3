"""Container entrypoint for the Track 1 hybrid routing agent.

Contract with the grading harness:
  * read  /input/tasks.json   (list of {"task_id", "prompt"})
  * write /output/results.json (list of {"task_id", "answer"}) before exiting
  * exit 0 on success, non-zero on failure
  * total runtime < 10 min, per-request < 30 s, ready < 60 s

ALL-REMOTE + PARALLEL: tasks are dispatched concurrently from a
ThreadPoolExecutor (MAX_WORKERS, default 5); every task is answered by an
allowed Fireworks model. A single GLOBAL HARD DEADLINE (500s from process
launch) bounds the run: any task not returned by then gets a deterministic
fallback answer so results.json is always complete and schema-valid.

INPUT_PATH / OUTPUT_PATH env vars exist only for local development; the
defaults match the harness contract.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

# Process-launch timestamp, taken before any import-heavy work we control:
# on the 2-vCPU grading box, everything counts against the 10-minute wall
# clock, so the global deadline below is anchored HERE.
CONTAINER_START_TS = time.time()

# Make repo-root imports work no matter where the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.api_clients.fireworks import FireworksClient          # noqa: E402
from src.local_models.loader import get_local_model            # noqa: E402
from src.router.dispatch import Router                         # noqa: E402
from src.router.formatting import format_answer                # noqa: E402

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Global hard deadline, anchored at PROCESS LAUNCH. 500s leaves 100s of
# headroom under the 600s harness limit for result collection, the flush of
# the three output files, and any still-running worker HTTP call (each is
# bounded by remote_timeout_seconds plus one retry).
HARD_DEADLINE_SECONDS = float(os.environ.get("HARD_DEADLINE_SECONDS", "500"))
MAX_WORKERS = max(1, int(os.environ.get("MAX_WORKERS", "5")))


def _category_token_breakdown(diag_rows: list) -> dict:
    """Per-category token sums and means — shows the biggest spend buckets."""
    cats: dict = {}
    for row in diag_rows:
        c = cats.setdefault(row.get("detected_category", "?"), {
            "tasks": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0,
        })
        c["tasks"] += 1
        c["prompt_tokens"] += row.get("prompt_tokens", 0)
        c["completion_tokens"] += row.get("completion_tokens", 0)
        c["total_tokens"] += row.get("prompt_tokens", 0) + row.get("completion_tokens", 0)
    for c in cats.values():
        c["mean_total"] = round(c["total_tokens"] / c["tasks"], 1) if c["tasks"] else 0
    return cats


def _write_outputs(results: list, diag_rows: list, full_rows: list,
                   runinfo: dict) -> None:
    """Write results.json (clean harness schema), diag.json, AND
    results_full.json (full prompt+answer per task, for offline judging).

    Shared flush path: called on the normal exit AND from the crash handler,
    so diagnostics survive even a partially-failed run.
    """
    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    from datetime import datetime, timezone
    diag = {
        "startup_secs": runinfo.get("startup_secs"),
        "total_elapsed_secs": round(time.time() - CONTAINER_START_TS, 2),
        # Wall-clock evidence for the organizers: when OUR PROCESS actually
        # began and ended. If the platform reports TIMEOUT with these two
        # stamps minutes apart, the overrun happened outside the container.
        "container_start_utc": datetime.fromtimestamp(
            CONTAINER_START_TS, tz=timezone.utc).isoformat(),
        "container_end_utc": datetime.now(tz=timezone.utc).isoformat(),
        "max_workers": MAX_WORKERS,
        "hard_deadline_secs": HARD_DEADLINE_SECONDS,
        "deadline_forced_tasks": runinfo.get("deadline_forced_tasks", 0),
        # Tasks answered by the zero-token lanes — each one is a task that
        # consumed no Fireworks tokens at all.
        "local_rule_tasks": sum(
            1 for r in diag_rows if r.get("route") == "local_rule"),
        "local_model_tasks": sum(
            1 for r in diag_rows if r.get("route") == "local_model"),
        # GGUF lazy-load evidence: 0.0/false when the model lane never fired.
        "model_load_secs": runinfo.get("model_load_secs", 0.0),
        "model_loaded": runinfo.get("model_loaded", False),
        # Token accounting (ranking metric): totals as reported by the API
        # client, plus a per-category breakdown for targeting reductions.
        "total_prompt_tokens": runinfo.get("total_prompt_tokens", 0),
        "total_completion_tokens": runinfo.get("total_completion_tokens", 0),
        "total_tokens": runinfo.get("total_tokens", 0),
        "category_tokens": _category_token_breakdown(diag_rows),
        "tasks": diag_rows,
    }
    with open(out.parent / "diag.json", "w", encoding="utf-8") as fh:
        json.dump(diag, fh, ensure_ascii=False, indent=1)
    with open(out.parent / "results_full.json", "w", encoding="utf-8") as fh:
        json.dump(full_rows, fh, ensure_ascii=False, indent=1)


def _fallback_answer(prompt: str, router: Router) -> str:
    """Deterministic answer used when a task fails or misses the deadline."""
    try:
        return router._local_answer(prompt)
    except Exception:
        return "Unable to produce an answer for this task."


def main() -> int:
    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as fh:
            tasks = json.load(fh)
        assert isinstance(tasks, list)
    except Exception as exc:
        print(f"FATAL: cannot read tasks from {INPUT_PATH}: {exc}", file=sys.stderr)
        return 1

    local_model = get_local_model()
    router = Router(local_model, FireworksClient())
    # Resolved role -> model map (from runtime ALLOWED_MODELS, never hardcoded)
    print(f"resolved model map: {router.resolved_map()}", flush=True)
    print(
        f"hybrid mode: heuristic classifier | rule lane "
        f"{'ON' if router.local_answers_enabled else 'OFF'} "
        f"({','.join(sorted(router.local_answer_categories)) or '-'}) | "
        f"model lane {'ON' if router.local_model_enabled else 'OFF'} "
        f"({','.join(sorted(router.local_model_categories)) or '-'}, "
        f"backend={local_model.backend}) | rest -> Fireworks "
        f"(max_workers={MAX_WORKERS}, deadline={HARD_DEADLINE_SECONDS:.0f}s)",
        flush=True,
    )

    first_task_ts = time.time()
    runinfo = {
        "startup_secs": round(first_task_ts - CONTAINER_START_TS, 2),
        "deadline_forced_tasks": 0,
    }
    print(f"STARTUP startup_secs={runinfo['startup_secs']} "
          f"max_workers={MAX_WORKERS}", file=sys.stderr, flush=True)

    results = []
    diag_rows = []
    full_rows = []
    try:
        _route_all(tasks, router, results, diag_rows, full_rows, runinfo)
    finally:
        # Same flush path for success and crash: all three output files are
        # emitted no matter what happened mid-run.
        fw = router.fireworks
        runinfo["total_prompt_tokens"] = getattr(fw, "total_prompt_tokens", 0)
        runinfo["total_completion_tokens"] = getattr(fw, "total_completion_tokens", 0)
        runinfo["total_tokens"] = getattr(fw, "total_tokens", 0)
        runinfo["model_load_secs"] = getattr(local_model, "load_secs", 0.0)
        runinfo["model_loaded"] = bool(getattr(local_model, "model_loaded", False))
        _write_outputs(results, diag_rows, full_rows, runinfo)

    elapsed = time.time() - CONTAINER_START_TS
    print(
        f"done: {len(results)} tasks in {elapsed:.1f}s | "
        f"fireworks calls={router.fireworks.calls} tokens={router.fireworks.total_tokens}",
        flush=True,
    )
    return 0


def _route_all(tasks, router, results, diag_rows, full_rows, runinfo):
    deadline_ts = CONTAINER_START_TS + HARD_DEADLINE_SECONDS
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = [executor.submit(router.route, task.get("prompt", ""))
               for task in tasks]
    try:
        for task, fut in zip(tasks, futures):
            task_id = task.get("task_id", "")
            prompt = task.get("prompt", "")
            try:
                remaining = deadline_ts - time.time()
                # A finished future returns instantly even past the deadline;
                # an unfinished one past the deadline raises immediately.
                answer, meta = fut.result(timeout=max(0.0, remaining))
            except FutureTimeout:
                runinfo["deadline_forced_tasks"] += 1
                if runinfo["deadline_forced_tasks"] == 1:
                    print(
                        f"GLOBAL DEADLINE hit at "
                        f"{time.time() - CONTAINER_START_TS:.0f}s: remaining "
                        f"unfinished tasks get deterministic fallback answers",
                        flush=True,
                    )
                answer, meta = _fallback_answer(prompt, router), {"route": "deadline_fallback"}
            except Exception as exc:  # one bad task must never sink the run
                answer, meta = _fallback_answer(prompt, router), {"route": "error", "error": str(exc)}
            cat = meta.get("decision", {}).get("intent", "?")
            if isinstance(answer, str) and answer.strip():
                answer = format_answer(cat, answer, prompt)
            if not isinstance(answer, str) or not answer.strip():
                answer = "No answer available."
            results.append({"task_id": task_id, "answer": answer})
            print(f"[{task_id}] route={meta.get('route')} model={meta.get('model', '-')}", flush=True)
            # Per-task diagnostic (stderr + /output/diag.json, non-sensitive):
            # lets a failed grading run be diagnosed instead of tuning blind.
            t = meta.get("timing", {})
            diag_rows.append({
                "task_id": task_id,
                "detected_category": cat,
                "route": meta.get("route"),
                "model_used": meta.get("model", "-"),
                "finish_reason": meta.get("finish_reason", "-"),
                "answer_len": len(answer),
                "truncated": bool(meta.get("truncated")),
                "primary_call_secs": t.get("primary_secs", 0),
                "alternate_fired": bool(t.get("alternate_fired")),
                "total_task_secs": t.get("total_secs", 0),
                "prompt_tokens": meta.get("prompt_tokens", 0),
                "completion_tokens": meta.get("completion_tokens", 0),
            })
            # Full record (prompt + final answer) for offline answer
            # inspection and judging; never read by the grading harness.
            full_rows.append({
                "task_id": task_id,
                "category": cat,
                "route": meta.get("route"),
                "model_used": meta.get("model", "-"),
                "prompt": prompt,
                "answer": answer,
                "finish_reason": meta.get("finish_reason", "-"),
                "truncated": bool(meta.get("truncated")),
                "total_task_secs": t.get("total_secs", 0),
            })
            print(
                f"DIAG {task_id} | {cat} | {meta.get('route')} | "
                f"{meta.get('model', '-')} | finish={meta.get('finish_reason', '-')} | "
                f"answer_len={len(answer)} | "
                f"truncated={'yes' if meta.get('truncated') else 'no'} | "
                f"task_secs={t.get('total_secs', 0)} "
                f"(primary={t.get('primary_secs', 0)}s, "
                f"alt={'yes ' + str(t.get('alternate_secs', 0)) + 's' if t.get('alternate_fired') else 'no'}) | "
                f"tokens={meta.get('prompt_tokens', 0)}p+{meta.get('completion_tokens', 0)}c",
                file=sys.stderr, flush=True,
            )
    finally:
        # Don't wait for stragglers: queued futures are cancelled, running
        # HTTP calls are bounded by their request timeout, and the harness
        # only needs results.json (written by our caller's finally-flush).
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    sys.exit(main())
