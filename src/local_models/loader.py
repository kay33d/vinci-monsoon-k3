"""Deterministic keyword classifier + last-resort fallback answers.

ALL-REMOTE ARCHITECTURE: the bundled GGUF and llama-cpp-python were removed
(image pull time was the prime suspect for platform-side grading timeouts,
and the small-GGUF classifier was unusably inaccurate). Stage-1 category
detection is now a pure keyword heuristic — zero tokens, zero latency, no
weights — and EVERY task is answered by an allowed Fireworks model
(``force_all_remote`` in the router fires because backend == "heuristic").

The category only selects the role→model lane and the output-format hint;
each task prompt carries its own instruction, so a rare misdetection sends
the task to the other (still strong) allowed model rather than breaking it.

``generate`` remains only as the deterministic, non-empty fallback used when
every remote attempt fails — results.json must never contain a blank answer.
"""

from __future__ import annotations

import re
from typing import Optional

# --- category cue patterns (checked in order; first hit wins) --------------
# Debug/codegen before math/logic: code prompts often contain digits too.
_CODE_SIGNAL = re.compile(r"(def |```|function\b|\bcode\b|\bclass \w+|=>|\breturn\b)", re.IGNORECASE)
_DEBUG_CUES = re.compile(r"\b(bug|fix|error|incorrect|wrong|doesn'?t work|broken|traceback|exception)\b", re.IGNORECASE)
_CODEGEN_CUES = re.compile(r"\b(write|implement|create|build)\b.{0,60}\b(function|code|program|script|class|method)\b", re.IGNORECASE)
_MATH_CUES = re.compile(
    r"\b(how many|how much|how long|calculate|compute|percent|remain|total|sum|cost|costs|"
    r"average|speed|rate|profit|revenue|interest|discount|price|at what time|"
    r"fills?|empties|grows|drops|per (minute|hour|second|day|year))\b|%|\d\s*[+\-*/^=]\s*\d",
    re.IGNORECASE,
)
_LOGIC_CUES = re.compile(
    r"\b(puzzle|constraints?|deduce|logic|logical|syllogism|true or false|"
    r"who owns|which day|what is the order|order of the|"
    r"immediately (left|right)|each (own|owns|speak|speaks|sit|sits|has|have))\b",
    re.IGNORECASE,
)


class LocalModel:
    """Heuristic-only backend. Keeps the pre-all-remote interface so the
    router, entrypoint, and eval harness are unchanged: ``backend`` is always
    "heuristic", which trips the router's force_all_remote path."""

    def __init__(self, model_path: Optional[str] = None):
        self.backend = "heuristic"
        self.load_secs = 0.0      # no weights are ever loaded
        self.model_loaded = False

    # ------------------------------------------------------------------ #
    # Stage 1: classification (0 tokens, deterministic)                   #
    # ------------------------------------------------------------------ #
    def classify(self, task_prompt: str, max_tokens: int = 64) -> dict:
        """Return {"intent","difficulty","confidence"} — always valid."""
        p = task_prompt.lower()
        if "sentiment" in p:
            return {"intent": "sentiment", "difficulty": "shallow", "confidence": "high"}
        if "summar" in p or "tl;dr" in p:
            return {"intent": "summarization", "difficulty": "shallow", "confidence": "high"}
        if "entit" in p:
            return {"intent": "ner", "difficulty": "shallow", "confidence": "high"}
        if _CODE_SIGNAL.search(task_prompt):
            if _DEBUG_CUES.search(task_prompt):
                return {"intent": "code_debugging", "difficulty": "deep", "confidence": "high"}
            if _CODEGEN_CUES.search(task_prompt):
                return {"intent": "code_generation", "difficulty": "deep", "confidence": "high"}
        if re.search(r"\d", task_prompt) and _MATH_CUES.search(task_prompt):
            return {"intent": "math_reasoning", "difficulty": "deep", "confidence": "high"}
        if _LOGIC_CUES.search(task_prompt):
            return {"intent": "logical_reasoning", "difficulty": "deep", "confidence": "high"}
        return {"intent": "factual_knowledge", "difficulty": "shallow", "confidence": "low"}

    # ------------------------------------------------------------------ #
    # Deterministic fallback answer (remote failed / deadline hit)        #
    # ------------------------------------------------------------------ #
    def generate(self, task_prompt: str, max_tokens: int = 300) -> str:
        """Minimal non-empty answer — results.json must never be blank."""
        head = re.sub(r"\s+", " ", task_prompt).strip()[:160]
        return f"Best-effort response (remote unavailable) to: {head}"


_singleton: Optional[LocalModel] = None


def get_local_model() -> LocalModel:
    global _singleton
    if _singleton is None:
        _singleton = LocalModel()
    return _singleton
