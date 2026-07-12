"""Unit tests for the zero-token LOCAL RULE lane.

The lane's contract: when it fires the answer is essentially guaranteed
correct; when in ANY doubt it returns None and the task goes remote. These
tests pin both sides — the hits AND the refusals — because a rule that fires
on a mixed review or a word problem would cost accuracy-gate points.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.router.local_answers import try_local_answer  # noqa: E402


# --------------------------------------------------------------------------- #
# math: fires only on pure-arithmetic asks, and is then exactly right          #
# --------------------------------------------------------------------------- #

def test_math_percent_of():
    ans = try_local_answer("math_reasoning", "What is 15% of 240?")
    assert ans is not None and ans.endswith("Answer: 36")


def test_math_plain_expression():
    ans = try_local_answer("math_reasoning", "Calculate 128 * 46 - 200.")
    assert ans is not None and ans.endswith("Answer: 5688")


def test_math_parentheses_and_power():
    ans = try_local_answer("math_reasoning", "Compute (3 + 5) * 2^3")
    assert ans is not None and ans.endswith("Answer: 64")


def test_math_division_non_integer():
    ans = try_local_answer("math_reasoning", "What is 7 / 2?")
    assert ans is not None and ans.endswith("Answer: 3.5")


def test_math_word_problem_refused():
    prompt = ("A store has 240 items. It sells 15% on Monday and 60 more on "
              "Tuesday. How many items remain?")
    assert try_local_answer("math_reasoning", prompt) is None


def test_math_multi_question_refused():
    # Trailing extra sentence breaks the anchor — must go remote.
    assert try_local_answer("math_reasoning",
                            "What is 2 + 2? Show your work.") is None


def test_math_division_by_zero_refused():
    assert try_local_answer("math_reasoning", "Calculate 5 / 0") is None


def test_math_huge_power_refused():
    assert try_local_answer("math_reasoning", "Calculate 9999 ^ 9999") is None


def test_math_single_number_refused():
    assert try_local_answer("math_reasoning", "What is 42?") is None


# --------------------------------------------------------------------------- #
# sentiment: fires only on clearly one-sided reviews                           #
# --------------------------------------------------------------------------- #

def test_sentiment_clear_positive():
    prompt = ("Classify the sentiment of this review: Absolutely love this "
              "blender — the build quality is excellent and the smoothies "
              "come out perfect every time.")
    assert try_local_answer("sentiment", prompt) == "Positive"


def test_sentiment_clear_negative_with_justification():
    prompt = ("Classify the sentiment of this review and briefly explain why: "
              "Terrible experience — the package arrived damaged and the "
              "seller was rude and unhelpful.")
    ans = try_local_answer("sentiment", prompt)
    assert ans is not None
    assert ans.startswith("Negative")
    assert "'terrible'" in ans.lower() or "'damaged'" in ans.lower()


def test_sentiment_mixed_contrast_refused():
    # mock-05: positive AND negative sides joined by "but" — judge territory.
    prompt = ("Classify the sentiment of this review: The battery life is "
              "great, but the screen scratches too easily.")
    assert try_local_answer("sentiment", prompt) is None


def test_sentiment_trap_refused():
    # trap-sent-01: late delivery + glowing food review.
    prompt = ("Classify the sentiment of this review: I waited 45 minutes "
              "past the delivery estimate, but the pizza arrived hot and was "
              "honestly the best I've had in years.")
    assert try_local_answer("sentiment", prompt) is None


def test_sentiment_negation_refused():
    prompt = ("Classify the sentiment of this review: This is not great and "
              "definitely not the excellent product they advertised.")
    assert try_local_answer("sentiment", prompt) is None


def test_sentiment_single_cue_refused():
    # One lone cue is too weak a signal to skip the LLM.
    prompt = "Classify the sentiment of this review: The pizza was great."
    assert try_local_answer("sentiment", prompt) is None


# --------------------------------------------------------------------------- #
# other categories never touch the lane                                        #
# --------------------------------------------------------------------------- #

def test_other_categories_refused():
    for cat in ("factual_knowledge", "summarization", "ner",
                "code_debugging", "logical_reasoning", "code_generation"):
        assert try_local_answer(cat, "What is 15% of 240?") is None


# --------------------------------------------------------------------------- #
# router integration: a rule hit spends ZERO Fireworks calls/tokens            #
# --------------------------------------------------------------------------- #

def test_router_rule_lane_skips_fireworks():
    from src.local_models.loader import get_local_model
    from src.router.dispatch import Router
    from tests.run_eval import MockFireworks

    client = MockFireworks()
    router = Router(get_local_model(), client)
    assert router.local_answers_enabled

    answer, meta = router.route("What is 15% of 240?")
    assert meta["route"] == "local_rule"
    assert answer.endswith("Answer: 36")
    assert client.calls == 0 and client.total_tokens == 0
    assert meta["prompt_tokens"] == 0 and meta["completion_tokens"] == 0

    # An ambiguous task still goes remote through the same router.
    _, meta2 = router.route(
        "Classify the sentiment of this review: The battery life is great, "
        "but the screen scratches too easily.")
    assert meta2["route"] == "remote"
    assert client.calls == 1


def test_router_env_kill_switch(monkeypatch):
    from src.local_models.loader import get_local_model
    from src.router.dispatch import Router
    from tests.run_eval import MockFireworks

    monkeypatch.setenv("LOCAL_ANSWERS", "0")
    router = Router(get_local_model(), MockFireworks())
    assert not router.local_answers_enabled
    _, meta = router.route("What is 15% of 240?")
    assert meta["route"] == "remote"


def test_easy_mocks_hit_rule_lane():
    """The four easy-* mock tasks must all resolve through the rule lane."""
    from src.local_models.loader import get_local_model
    from src.router.dispatch import Router
    from tests.run_eval import MockFireworks

    with open(ROOT / "tests" / "mock_tasks.json", encoding="utf-8") as fh:
        tasks = {t["task_id"]: t["prompt"] for t in json.load(fh)}
    router = Router(get_local_model(), MockFireworks())
    for tid in ("easy-math-01", "easy-math-02", "easy-sent-01", "easy-sent-02"):
        _, meta = router.route(tasks[tid])
        assert meta["route"] == "local_rule", f"{tid} did not hit the rule lane"
