"""Unit tests for the LOCAL MODEL (quantized GGUF) lane.

The lane's contract mirrors the rule lane's: it may only ever REPLACE a
remote call with a validated answer — any weak output, exhausted time
budget, or late start falls through to Fireworks. These tests fake the GGUF
backend (no weights needed) and pin the gates, the validation, and the lane
ordering against both the rule lane and the remote path.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_models.loader import LocalModel  # noqa: E402
from src.router.dispatch import Router, _valid_local_answer  # noqa: E402
from tests.run_eval import MockFireworks  # noqa: E402

MIXED_SENTIMENT = ("Classify the sentiment of this review: The battery life "
                   "is great, but the screen scratches too easily.")
CLEAR_SENTIMENT = ("Classify the sentiment of this review: Absolutely love "
                   "this blender — the build quality is excellent and the "
                   "smoothies come out perfect every time.")
SUMMARIZATION = ("Summarize the following in exactly one sentence: The new "
                 "library opened downtown last month. It offers free coding "
                 "classes and extended weekend hours.")


class FakeGGUF(LocalModel):
    """Pretends the GGUF backend is available and returns canned answers."""

    def __init__(self, answer="mixed — praises battery but criticizes the screen."):
        super().__init__(model_path="/nonexistent/fake.gguf")
        self.backend = "gguf-lazy"
        self.canned = answer
        self.calls = 0

    def llm_answer(self, system, user, max_tokens=180):
        self.calls += 1
        return self.canned


def make_router(local=None, **cfg_overrides):
    from src.router.dispatch import load_config

    cfg = load_config()
    for key, val in cfg_overrides.items():
        section, opt = key.split(".", 1)
        cfg.setdefault(section, {})[opt] = val
    return Router(local or FakeGGUF(), MockFireworks(), config=cfg)


# --------------------------------------------------------------------------- #
# validation                                                                   #
# --------------------------------------------------------------------------- #

def test_validation_rules():
    assert _valid_local_answer("sentiment", "Mixed — good food, slow service.")
    assert not _valid_local_answer("sentiment", "The review praises the food.")
    assert not _valid_local_answer("sentiment", "I'm sorry, I cannot classify this.")
    assert _valid_local_answer("summarization", "The library opened and is popular.")
    assert not _valid_local_answer("summarization", "Popular.")
    assert not _valid_local_answer("summarization", "")


def test_exact_count_validation():
    # The measured live failure: "exactly three bullet points" -> 4 bullets.
    prompt3 = "Summarize the passage in exactly three bullet points."
    four = "- flexibility\n- work-life balance improves\n- culture challenges\n- office rethink"
    three = "- flexibility gains for employees\n- culture challenges persist\n- offices become social hubs"
    assert not _valid_local_answer("summarization", four, prompt3)
    assert _valid_local_answer("summarization", three, prompt3)

    prompt2 = "Summarize the following passage in exactly two sentences: ..."
    two = "ML is widely used in healthcare. However, concerns about bias persist."
    one = "ML is widely used in healthcare despite concerns."
    assert _valid_local_answer("summarization", two, prompt2)
    assert not _valid_local_answer("summarization", one, prompt2)

    # No "exactly N" in the prompt -> no count constraint.
    assert _valid_local_answer("summarization", four,
                               "Summarize the passage briefly.")


# --------------------------------------------------------------------------- #
# lane behaviour                                                               #
# --------------------------------------------------------------------------- #

def test_model_lane_answers_mixed_sentiment():
    router = make_router()
    answer, meta = router.route(MIXED_SENTIMENT)
    assert meta["route"] == "local_model"
    assert router.fireworks.calls == 0
    assert "mixed" in answer.lower()


def test_rule_lane_wins_over_model_lane():
    local = FakeGGUF()
    router = make_router(local)
    _, meta = router.route(CLEAR_SENTIMENT)
    assert meta["route"] == "local_rule"
    assert local.calls == 0  # GGUF never touched for rule-lane hits


def test_summarization_goes_to_model_lane():
    local = FakeGGUF(answer="The new downtown library offers classes and is popular.")
    router = make_router(local)
    _, meta = router.route(SUMMARIZATION)
    assert meta["route"] == "local_model"


def test_invalid_output_escalates_to_remote():
    local = FakeGGUF(answer="The review praises the battery.")  # no label word
    router = make_router(local)
    _, meta = router.route(MIXED_SENTIMENT)
    assert meta["route"] == "remote"
    assert local.calls == 1  # lane tried, failed validation, escalated
    assert router.fireworks.calls == 1


def test_math_never_uses_model_lane():
    local = FakeGGUF()
    router = make_router(local)
    _, meta = router.route(
        "A store has 240 items. It sells 15% on Monday and 60 more on "
        "Tuesday. How many items remain?")
    assert meta["route"] == "remote"
    assert local.calls == 0


def test_time_budget_gate():
    router = make_router()
    router._lm_spent = router.lm_time_budget + 1
    _, meta = router.route(MIXED_SENTIMENT)
    assert meta["route"] == "remote"


def test_latest_start_gate():
    router = make_router()
    router._t0 -= router.lm_latest_start + 10  # pretend we're late in the run
    _, meta = router.route(MIXED_SENTIMENT)
    assert meta["route"] == "remote"


def test_long_prompt_gate():
    local = FakeGGUF(answer="A summary of the very long passage in one sentence.")
    router = make_router(local)
    long_prompt = "Summarize the following in one sentence: " + "word " * 1500
    _, meta = router.route(long_prompt)
    assert meta["route"] == "remote"
    assert local.calls == 0


def test_env_kill_switch(monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL", "0")
    router = make_router()
    assert not router.local_model_enabled
    _, meta = router.route(MIXED_SENTIMENT)
    assert meta["route"] == "remote"


def test_heuristic_backend_disables_lane():
    """Offline boxes / CI gate: no GGUF -> lane closed, behaviour unchanged."""
    local = LocalModel(model_path="/nonexistent/nope.gguf")
    assert local.backend == "heuristic"
    router = make_router(local)
    assert not router.local_model_enabled
    _, meta = router.route(MIXED_SENTIMENT)
    assert meta["route"] == "remote"
