"""Stage 2 — routing decision + category->role->model resolution.

HYBRID: TWO ZERO-TOKEN LOCAL LANES, THEN REMOTE. Decision flow per task
(thread-safe; the entrypoint calls ``route`` from a ThreadPoolExecutor):
  1. classify with the deterministic keyword heuristic (0 tokens, instant)
  2. LOCAL RULE lane (config ``local_answers``): provably-easy tasks —
     clearly one-sided sentiment, pure-arithmetic math — are answered
     deterministically for 0 Fireworks tokens (the ranking metric).
  3. LOCAL MODEL lane (config ``local_model``): the bundled quantized Qwen
     GGUF answers the categories it is historically reliable at (sentiment,
     summarization by default) — also 0 Fireworks tokens. Guarded by hard
     time gates (cumulative generation budget + a latest-start cutoff) so a
     slow grading CPU can never push the run past the global deadline, and
     by per-category output validation: an answer that fails validation
     falls through to remote exactly as if the lane didn't exist.
  4. resolve category -> role -> concrete model ID from runtime
     ALLOWED_MODELS (never hardcoded; graceful fallback to first allowed)
  5. call the primary model ONCE (client retries transients internally,
     4xx permanent). EMPTY content only -> one attempt on the OTHER allowed
     model. If everything fails -> deterministic non-empty fallback answer.
Local generation is serialized behind the model lock while remote tasks run
in parallel around it; each HTTP call is bounded by remote_timeout_seconds,
and the GLOBAL 500s deadline in entrypoint.py is the run-level guard.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import yaml

from config.prompts import LOCAL_ANSWER_SYSTEM, REMOTE_SYSTEM, remote_user_prompt
from src.api_clients.fireworks import EmptyCompletion, FireworksClient, FireworksError
from src.local_models.loader import LocalModel
from src.router.classifier import classify_task
from src.router.local_answers import try_local_answer

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "routing_map.yaml"

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


def _env_flag(name: str, default: bool) -> bool:
    """0/false/off disables, 1/true/on enables, unset -> config default."""
    val = os.environ.get(name, "").strip().lower()
    if val in ("0", "false", "off"):
        return False
    if val in ("1", "true", "on"):
        return True
    return default


# Local-model output validation: an answer that fails its category check is
# discarded and the task escalates to Fireworks — the lane can only ever
# REPLACE a remote call with an equally-acceptable answer, never degrade one.
_SENTIMENT_LABEL = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.IGNORECASE)
_REFUSAL = re.compile(r"\b(i can'?t|i cannot|as an ai|i'?m sorry|i am sorry)\b", re.IGNORECASE)

# "exactly two sentences" / "exactly 3 bullet points": small local models
# sometimes miscount (measured: 1.5B gave 4 bullets for "exactly three").
# The judge grades format compliance, so a count mismatch must escalate.
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_EXACT_COUNT = re.compile(
    r"exactly\s+(\d+|" + "|".join(_WORD_NUM) + r")\s+"
    r"(sentence|bullet|point|line|word)", re.IGNORECASE)
_BULLET_LINE = re.compile(r"^\s*([-*•]|\d+[.)])\s+\S")


def _requested_count_ok(prompt: str, text: str) -> bool:
    m = _EXACT_COUNT.search(prompt)
    if not m:
        return True
    want = int(m.group(1)) if m.group(1).isdigit() else _WORD_NUM[m.group(1).lower()]
    unit = m.group(2).lower()
    if unit == "word":
        return len(text.split()) == want
    if unit in ("bullet", "point", "line"):
        got = sum(1 for ln in text.splitlines() if _BULLET_LINE.match(ln))
    else:  # sentences: split at terminator + following capital/quote
        got = len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])",
                                       text.strip()) if s.strip()])
    return got == want


def _valid_local_answer(category: str, text: str, prompt: str = "") -> bool:
    if not text or len(text.strip()) < 3 or _REFUSAL.search(text[:120]):
        return False
    if category == "sentiment":
        return bool(_SENTIMENT_LABEL.search(text[:200]))
    if category == "summarization":
        return len(text.split()) >= 5 and _requested_count_ok(prompt, text)
    return True


class Router:
    def __init__(self, local_model: LocalModel, fireworks: FireworksClient,
                 config: Optional[dict] = None):
        self.local = local_model
        self.fireworks = fireworks
        self.cfg = config or load_config()
        self.allowed = allowed_models()
        self.limits = self.cfg.get("limits", {})
        self.thresholds = self.cfg.get("thresholds", {})
        # Without a usable GGUF (backend == "heuristic") every task that the
        # rule lane doesn't catch goes remote — deterministic fallback text
        # is only for total remote failure. (The RULE lane is independent:
        # it is exact by construction, not a weak-model answer.)
        self.force_all_remote = getattr(local_model, "backend", "") == "heuristic"
        # Zero-token rule lane: config-driven, env-overridable for A/B runs
        # (LOCAL_ANSWERS=0 disables, =1 forces on, unset -> config value).
        la_cfg = self.cfg.get("local_answers", {})
        self.local_answers_enabled = _env_flag("LOCAL_ANSWERS",
                                               bool(la_cfg.get("enabled", False)))
        self.local_answer_categories = set(la_cfg.get("categories", []))
        # LOCAL MODEL lane (quantized GGUF). All knobs from config; the env
        # kill switch LOCAL_MODEL=0/1 exists for A/B integration runs.
        lm_cfg = self.cfg.get("local_model", {})
        self.local_model_enabled = (
            _env_flag("LOCAL_MODEL", bool(lm_cfg.get("enabled", False)))
            and getattr(local_model, "backend", "") == "gguf-lazy"
        )
        self.local_model_categories = set(lm_cfg.get("categories", []))
        self.lm_max_prompt_chars = int(lm_cfg.get("max_prompt_chars", 2600))
        self.lm_max_tokens = lm_cfg.get("max_tokens", {}) or {}
        self.lm_time_budget = float(lm_cfg.get("time_budget_secs", 240))
        self.lm_latest_start = float(lm_cfg.get("latest_start_secs", 350))
        # Router construction happens right after process launch, so this
        # anchor approximates container start for the model lane's gates.
        self._t0 = time.time()
        self._lm_spent = 0.0                    # cumulative GGUF generation secs
        self._lm_spent_lock = threading.Lock()

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

        # LOCAL MODEL lane: quantized GGUF answer for 0 tokens. Validation
        # failure or any time-gate refusal falls through to remote.
        if self._local_model_eligible(category, task_prompt):
            lm_answer = self._try_local_model(category, task_prompt, timing)
            if lm_answer:
                meta.update(route="local_model",
                            model=os.path.basename(self.local.model_path))
                timing["total_secs"] = round(time.time() - t_start, 2)
                return lm_answer, meta

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

    # ------------------------------------------------------------------ #
    # LOCAL MODEL lane                                                    #
    # ------------------------------------------------------------------ #
    def _local_model_eligible(self, category: str, prompt: str) -> bool:
        """Hard gates that keep the GGUF lane provably inside the deadline:
        the lane refuses long prompts (CPU prompt-eval is the slow part),
        refuses once the cumulative generation budget is spent, and refuses
        to START a generation late in the run."""
        if not self.local_model_enabled or category not in self.local_model_categories:
            return False
        if len(prompt) > self.lm_max_prompt_chars:
            return False
        elapsed = time.time() - self._t0
        if elapsed > self.lm_latest_start:
            return False
        with self._lm_spent_lock:
            return self._lm_spent < self.lm_time_budget

    def _try_local_model(self, category: str, prompt: str,
                         timing: dict) -> Optional[str]:
        max_tokens = int(self.lm_max_tokens.get(category, 160))
        t0 = time.time()
        text = self.local.llm_answer(
            system=LOCAL_ANSWER_SYSTEM,
            user=remote_user_prompt(category, prompt),
            max_tokens=max_tokens,
        )
        gen_secs = round(time.time() - t0, 2)
        timing["primary_secs"] = gen_secs
        with self._lm_spent_lock:
            self._lm_spent += gen_secs
        if text and _valid_local_answer(category, text, prompt):
            return text
        return None  # invalid / empty -> remote path takes over

    def _local_answer(self, task_prompt: str) -> str:
        return self.local.generate(
            task_prompt, max_tokens=self.limits.get("local_max_tokens", 300)
        )
