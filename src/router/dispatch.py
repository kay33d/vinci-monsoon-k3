"""Stage 2 — routing decision + category->role->model resolution.

REMOTE-FIRST WITH A ZERO-TOKEN LOCAL RULE LANE. Decision flow per task
(thread-safe; the entrypoint calls ``route`` from a ThreadPoolExecutor):
  1. classify with the deterministic keyword heuristic (0 tokens, instant)
  2. LOCAL RULE lane (config ``local_answers``): provably-easy tasks —
     clearly one-sided sentiment, pure-arithmetic math — are answered
     deterministically for 0 Fireworks tokens (the ranking metric). The
     lane returns None for anything ambiguous, which falls through to:
  3. resolve category -> role -> concrete model ID from runtime
     ALLOWED_MODELS (never hardcoded; graceful fallback to first allowed)
  4. call the primary model ONCE (client retries transients internally,
     4xx permanent). EMPTY content only -> one attempt on the OTHER allowed
     model. If everything fails -> deterministic non-empty fallback answer.
There is no per-task budget any more: each HTTP call is bounded by
remote_timeout_seconds, and the GLOBAL 500s deadline in entrypoint.py is the
run-level guard.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import yaml

from config.prompts import REMOTE_SYSTEM, remote_user_prompt
from src.api_clients.fireworks import EmptyCompletion, FireworksClient, FireworksError
from src.local_models.loader import LocalModel
from src.router.classifier import classify_task
from src.router.local_answers import try_local_answer

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "routing_map.yaml"

# The stage-1 classifier emits categorical confidence; map it to a score so
# the config threshold (local_confidence_threshold) can gate it numerically.
_CONFIDENCE_SCORE = {"high": 0.95, "low": 0.50}

# --- Aggressive escalation cues (kept for diag visibility) ------------------
_CODE_CUES = re.compile(
    r"(def |class |return\b|function\b|=>|;|\{|\}|import |print\(|"
    r"\bpython\b|\bjavascript\b|\bjava\b|\bc\+\+\b|\bsql\b|\brust\b|\bbug\b)",
    re.IGNORECASE,
)
_MATH_CUES = re.compile(
    r"(\d+\s*[+\-*/^=]\s*\d+|%|\bpercent|\bcalculate\b|\bhow many\b|\baverage\b|"
    r"\bsum\b|\btotal\b|\bprofit\b|\bratio\b|\brate\b|\bprojection\b)",
    re.IGNORECASE,
)
_REASONING_CUES = re.compile(
    r"(step[- ]by[- ]step|explain your reasoning|deduce|puzzle|constraint|"
    r"each own|who owns|what is the order)",
    re.IGNORECASE,
)


def load_config(path: Optional[Path] = None) -> dict:
    with open(path or _CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def allowed_models() -> list[str]:
    """Parse ALLOWED_MODELS from the environment (never hardcoded)."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


class Router:
    def __init__(self, local_model: LocalModel, fireworks: FireworksClient,
                 config: Optional[dict] = None):
        self.local = local_model
        self.fireworks = fireworks
        self.cfg = config or load_config()
        self.allowed = allowed_models()
        self.limits = self.cfg.get("limits", {})
        self.thresholds = self.cfg.get("thresholds", {})
        # Heuristic backend (always, since the GGUF was removed) forces
        # EVERY task remote — local text is only the last-resort fallback.
        # (The LOCAL RULE lane below is independent of this: it is exact by
        # construction, not a weak-model answer.)
        self.force_all_remote = getattr(local_model, "backend", "") == "heuristic"
        # Zero-token rule lane: config-driven, env-overridable for A/B runs
        # (LOCAL_ANSWERS=0 disables, =1 forces on, unset -> config value).
        la_cfg = self.cfg.get("local_answers", {})
        env_flag = os.environ.get("LOCAL_ANSWERS", "").strip().lower()
        if env_flag in ("0", "false", "off"):
            self.local_answers_enabled = False
        elif env_flag in ("1", "true", "on"):
            self.local_answers_enabled = True
        else:
            self.local_answers_enabled = bool(la_cfg.get("enabled", False))
        self.local_answer_categories = set(la_cfg.get("categories", []))

    # ------------------------------------------------------------------ #
    # role -> concrete allowed model ID                                   #
    # ------------------------------------------------------------------ #
    def resolve_role(self, role: str) -> Optional[str]:
        """Resolve a role to a model ID present in ALLOWED_MODELS.

        Hint substrings are matched case-insensitively; graceful fallback to
        the first allowed model; None only if ALLOWED_MODELS is empty.
        """
        if not self.allowed:
            return None
        for hint in self.cfg["role_model_hints"].get(role, []):
            for model_id in self.allowed:
                if hint.lower() in model_id.lower():
                    return model_id
        return self.allowed[0]

    def resolve_model(self, category: str) -> Optional[str]:
        return self.resolve_role(self.cfg["category_roles"].get(category, "general"))

    def resolved_map(self) -> dict:
        """role -> model ID map, for the startup log."""
        roles = sorted(set(self.cfg["category_roles"].values()))
        return {role: self.resolve_role(role) for role in roles}

    # ------------------------------------------------------------------ #
    # escalation decision (vestigial: force_all_remote is always true)    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def aggressive_override(prompt: str) -> Optional[str]:
        """Return the cue type detected in the prompt (diag only)."""
        if _CODE_CUES.search(prompt):
            return "code_cue"
        if _MATH_CUES.search(prompt):
            return "math_cue"
        if _REASONING_CUES.search(prompt):
            return "reasoning_cue"
        return None

    def should_answer_locally(self, decision: dict, prompt: str) -> bool:
        if self.force_all_remote:
            return False
        policy_cfg = self.cfg.get("escalation_policy", {})
        policy = policy_cfg.get("overrides", {}).get(
            decision["intent"], policy_cfg.get("default", "strict")
        )
        if policy == "always":
            return False
        if decision["difficulty"] != "shallow":
            return False
        conf = _CONFIDENCE_SCORE.get(decision.get("confidence"), 0.0)
        if conf <= self.thresholds.get("local_confidence_threshold", 0.90):
            return False
        return True

    # ------------------------------------------------------------------ #
    # full pipeline for one task (runs inside a worker thread)            #
    # ------------------------------------------------------------------ #
    def route(self, task_prompt: str) -> tuple[str, dict]:
        """Return (answer, meta). meta records how the task was routed,
        including per-call wall-clock timing.

        Remote attempts: primary model once (the client already retries
        transient failures internally with backoff). ONLY an EMPTY
        completion triggers one attempt on the other allowed model —
        empties are model-specific (hidden reasoning), so the sibling
        model usually rescues the task. Any other failure goes straight
        to the deterministic fallback: no unbounded retry chains.
        """
        t_start = time.time()
        timing = {"primary_secs": 0.0, "alternate_fired": False,
                  "alternate_secs": 0.0, "total_secs": 0.0}
        decision = classify_task(
            task_prompt, self.local,
            max_chars=self.limits.get("classify_prompt_chars", 1500),
            max_tokens=self.limits.get("classify_max_tokens", 64),
        )
        category = decision["intent"]
        meta = {
            "decision": decision, "route": "local", "model": "local",
            "finish_reason": "-", "truncated": False,
            "escalation_cue": self.aggressive_override(task_prompt) or "-",
            "timing": timing,
            "prompt_tokens": 0, "completion_tokens": 0,
        }

        def _add_usage() -> None:
            """Accumulate this task's token spend across ALL its attempts —
            an empty completion still burned tokens and still scores."""
            pt, ct = getattr(self.fireworks, "last_usage", (0, 0))
            meta["prompt_tokens"] += pt
            meta["completion_tokens"] += ct

        def _finish_local(route: str = "local", err: Optional[str] = None) -> tuple[str, dict]:
            if err:
                meta.update(route=route, model="local", error=err)
            answer = self._local_answer(task_prompt)
            timing["total_secs"] = round(time.time() - t_start, 2)
            return answer, meta

        # LOCAL RULE lane: exact deterministic answer for 0 tokens, or None
        # to escalate. Tried BEFORE any remote resolution so a hit spends
        # neither tokens nor network time.
        if self.local_answers_enabled and category in self.local_answer_categories:
            rule_answer = try_local_answer(category, task_prompt)
            if rule_answer:
                meta.update(route="local_rule", model="rule")
                timing["total_secs"] = round(time.time() - t_start, 2)
                return rule_answer, meta

        if self.should_answer_locally(decision, task_prompt):
            return _finish_local()

        primary = self.resolve_model(category)
        if primary is None:  # ALLOWED_MODELS empty: fallback is all we have
            return _finish_local()

        role = self.cfg["category_roles"].get(category, "general")
        max_tokens = self.limits.get("remote_max_tokens_by_role", {}).get(
            role, self.limits.get("remote_max_tokens", 512)
        )
        req_timeout = self.limits.get("remote_timeout_seconds", 12)
        # Primary, then (on EMPTY content only) the other allowed model.
        alternate = next((m for m in self.allowed if m != primary), None)
        attempts = [primary]
        last_err: Optional[Exception] = None
        for idx, model_id in enumerate(attempts):
            call_t0 = time.time()
            # reasoning_effort for minimax on the reasoning role comes from
            # config (A/B knob: "none" vs "low"); everything else stays
            # "none" for speed and guaranteed non-empty content.
            effort = (self.thresholds.get("reasoning_role_effort", "low")
                      if (role == "reasoning" and "minimax" in model_id.lower())
                      else "none")
            try:
                answer = self.fireworks.chat(
                    model=model_id,
                    system=REMOTE_SYSTEM,
                    user=remote_user_prompt(category, task_prompt),
                    max_tokens=max_tokens,
                    timeout=req_timeout,
                    reasoning_effort=effort,
                )
                call_secs = round(time.time() - call_t0, 2)
                _add_usage()
                if idx == 0:
                    timing["primary_secs"] = call_secs
                else:
                    timing["alternate_fired"] = True
                    timing["alternate_secs"] = call_secs
                finish = getattr(self.fireworks, "last_finish_reason", None)
                meta.update(
                    route="remote", model=model_id,
                    finish_reason=finish or "?",
                    truncated=finish == "length",
                )
                timing["total_secs"] = round(time.time() - t_start, 2)
                return answer, meta
            except FireworksError as exc:
                call_secs = round(time.time() - call_t0, 2)
                _add_usage()
                if idx == 0:
                    timing["primary_secs"] = call_secs
                else:
                    timing["alternate_fired"] = True
                    timing["alternate_secs"] = call_secs
                last_err = exc
                # Empty completion is the ONE case worth a sibling-model
                # attempt; everything else (auth, 4xx, exhausted retries)
                # would fail there too.
                if isinstance(exc, EmptyCompletion) and alternate is not None \
                        and len(attempts) == 1:
                    attempts.append(alternate)

        # Remote attempts failed — degrade to the deterministic fallback.
        return _finish_local("local_fallback", str(last_err) if last_err else "remote unavailable")

    def _local_answer(self, task_prompt: str) -> str:
        return self.local.generate(
            task_prompt, max_tokens=self.limits.get("local_max_tokens", 300)
        )
