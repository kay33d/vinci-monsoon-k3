"""Deterministic keyword classifier + optional quantized-GGUF answer backend.

HYBRID ARCHITECTURE (three lanes, cheapest first):
  1. Stage-1 category detection is a pure keyword heuristic — zero tokens,
     zero latency, no weights. An earlier round tried the bundled GGUF AS
     THE CLASSIFIER and it was measurably inaccurate at 0.5B, so
     classification NEVER uses the model regardless of its size.
  2. A bundled quantized Qwen GGUF (llama-cpp-python, CPU) answers the
     easy categories the router sends it (config ``local_model``) for ZERO
     Fireworks tokens — a different, lower-stakes job than classifying
     (only 2 categories, and dispatch.py's validation gate discards any
     answer that fails a format check, so a weak completion costs a
     Fireworks call, never a wrong final answer). Current default is
     Qwen2.5-0.5B (image-size constrained: 1.5B pushed the compressed
     image to 1.2 GB, which timed out on the grading platform; 0.5B keeps
     it under the ~530 MB size that is confirmed to pull successfully).
     The model is LAZY-LOADED on first use — startup on the 2-vCPU grading
     box counts against the 10-minute wall clock — and generation is
     serialized behind a lock (llama.cpp instances are not thread-safe)
     while remote tasks proceed in parallel around it.
  3. Everything else goes to an allowed Fireworks model.

``generate`` remains the deterministic, instant, non-empty fallback used
when every other path fails — it deliberately never touches the GGUF, so
deadline/crash fallbacks cost zero seconds. When the GGUF or llama_cpp is
absent (dev boxes, CI's offline gate), backend == "heuristic" and the
model lane simply never opens.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading
import time
from typing import Optional

DEFAULT_MODEL_PATH = "/models/model.gguf"

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
    """Keyword classifier + (optionally) a lazy llama.cpp answer backend."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.environ.get("LOCAL_MODEL_PATH", DEFAULT_MODEL_PATH)
        self._llm = None
        self._load_attempted = False
        self._lock = threading.Lock()   # llama.cpp is NOT thread-safe
        self.load_secs = 0.0            # wall clock of the actual GGUF load
        # Cheap capability probe only — NO weights load at startup (ready<60s;
        # and if the model lane never fires, the load never happens at all).
        self.backend = "gguf-lazy" if self._llama_available() else "heuristic"

    def _llama_available(self) -> bool:
        """Model file exists and llama_cpp is importable — without importing."""
        return (
            os.path.isfile(self.model_path)
            and importlib.util.find_spec("llama_cpp") is not None
        )

    @property
    def model_loaded(self) -> bool:
        return self._llm is not None

    def _ensure_loaded(self) -> None:
        """LAZY TRIGGER: first llm_answer() call loads the weights."""
        with self._lock:
            if self._llm is not None or self._load_attempted:
                return
            self._load_attempted = True
            t0 = time.time()
            try:
                from llama_cpp import Llama

                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=3072,
                    n_threads=os.cpu_count() or 2,
                    n_gpu_layers=0,  # grading VM is CPU-only
                    verbose=False,
                )
                self.load_secs = round(time.time() - t0, 2)
                print(f"GGUF lazy-loaded in {self.load_secs}s",
                      file=sys.stderr, flush=True)
            except Exception as exc:
                # Load failure degrades to remote-for-everything — a running
                # agent with remote answers beats a crashed container.
                self._llm = None
                self.backend = "heuristic"
                print(f"GGUF load failed ({exc}); model lane disabled",
                      file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ #
    # Stage 1: classification (0 tokens, deterministic — NEVER the GGUF)  #
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
    # Stage 2a: GGUF answering (0 Fireworks tokens; serialized)           #
    # ------------------------------------------------------------------ #
    def llm_answer(self, system: str, user: str, max_tokens: int = 180) -> Optional[str]:
        """One local completion, or None if the model can't produce one.

        None (never an exception) so the router falls through to the normal
        remote path. Generation holds the instance lock: local tasks queue
        behind each other while remote tasks run in parallel around them.
        """
        if self.backend != "gguf-lazy":
            return None
        self._ensure_loaded()
        if self._llm is None:
            return None
        try:
            with self._lock:
                out = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
            text = (out["choices"][0]["message"].get("content") or "").strip()
            return text or None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Deterministic fallback answer (remote failed / deadline hit)        #
    # Deliberately NEVER the GGUF: fallbacks must cost zero seconds.      #
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
