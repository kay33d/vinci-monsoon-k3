"""Zero-token LOCAL RULE lane — deterministic answers for provably-easy tasks.

Token efficiency play: every task answered here costs 0 Fireworks tokens
(the ranking metric once the 80% accuracy gate is passed). The lane is
DELIBERATELY CONSERVATIVE because accuracy headroom above the gate is thin:
it fires only when a rule-based answer is essentially guaranteed to pass an
LLM judge, and returns None for everything else so the router escalates to
Fireworks exactly as before.

Coverage (by design, nothing more):
  * sentiment  — clearly one-sided reviews only. ANY negation word, contrast
    conjunction ("but", "however", ...), or a single cue of the opposite
    polarity sends the task remote: mixed/trap reviews are where judges dock
    points, so rules never touch them.
  * math       — pure-arithmetic asks only ("What is 15% of 240?",
    "Calculate 128 * 46 - 200"). The WHOLE prompt must be a single anchored
    arithmetic question; word problems (units, rates, multi-step stories)
    never match and go remote. When the rule fires the result is computed
    exactly, so it can't be wrong.

NER and summarization are deliberately excluded: rule-based NER measurably
dropped DATE/ORG entities in the judged rounds, and extractive summaries are
a judge-risk with no reliable confidence signal.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Optional

# --------------------------------------------------------------------------- #
# sentiment                                                                    #
# --------------------------------------------------------------------------- #
# Label words themselves ("positive", "negative", "neutral") are deliberately
# NOT cues: they appear in the instruction ("classify as Positive/Negative"),
# not in the review.
_POSITIVE_CUES = frozenset("""
    great excellent amazing awesome fantastic wonderful love loved loves
    perfect best superb brilliant outstanding delightful impressed impressive
    happy satisfied recommend recommended flawless incredible smooth reliable
    pleasant comfortable beautiful sturdy durable helpful friendly delicious
    gorgeous seamless
""".split())

_NEGATIVE_CUES = frozenset("""
    terrible awful horrible bad worst poor disappointing disappointed broken
    broke defective useless waste refund late damaged dented missing scratched
    scratches cracked faulty annoying frustrating frustrated unusable regret
    avoid rude unhelpful leaked overpriced noisy flimsy garbage junk
    malfunction malfunctioning unacceptable slow
""".split())

# Any of these anywhere in the review => polarity may be flipped or mixed =>
# the rule refuses and the task goes remote.
_NEGATIONS = re.compile(
    r"\b(not|no|never|none|hardly|barely|isn'?t|wasn'?t|aren'?t|weren'?t|"
    r"don'?t|doesn'?t|didn'?t|can'?t|cannot|couldn'?t|won'?t|wouldn'?t|"
    r"shouldn'?t|lacks?|without)\b",
    re.IGNORECASE,
)
_CONTRAST = re.compile(
    r"\b(but|however|although|though|yet|except|unfortunately|despite|"
    r"whereas|while)\b|on the other hand",
    re.IGNORECASE,
)

# Same trigger the formatter uses: if the task asks for a justification, a
# bare label would LOSE judge points, so the rule answer includes one.
_WANTS_JUSTIFICATION = re.compile(r"justify|explain|why|reason|because", re.IGNORECASE)


def _sentiment_answer(prompt: str) -> Optional[str]:
    # The review text usually follows the instruction after a colon; scan
    # only that part so instruction wording can't contribute cues.
    review = prompt.split(":", 1)[1] if ":" in prompt else prompt
    if _NEGATIONS.search(review) or _CONTRAST.search(review):
        return None  # possibly flipped or mixed — judges live here; go remote
    words = re.findall(r"[a-z']+", review.lower())
    pos_hits = [w for w in words if w in _POSITIVE_CUES]
    neg_hits = [w for w in words if w in _NEGATIVE_CUES]
    # Fire only on a strong one-sided signal: >=2 cues, ZERO opposite cues.
    if len(pos_hits) >= 2 and not neg_hits:
        label, hits = "Positive", pos_hits
    elif len(neg_hits) >= 2 and not pos_hits:
        label, hits = "Negative", neg_hits
    else:
        return None
    if _WANTS_JUSTIFICATION.search(prompt):
        cited = ", ".join(f"'{w}'" for w in dict.fromkeys(hits[:3]))
        opposite = "negative" if label == "Positive" else "positive"
        return (f"{label} — the review uses clearly {label.lower()} language "
                f"such as {cited} and contains no {opposite} remarks.")
    return label


# --------------------------------------------------------------------------- #
# math                                                                         #
# --------------------------------------------------------------------------- #
# Anchored templates: the ENTIRE prompt must be one arithmetic question.
# Anything with surrounding story text ("A store has 240 items...") fails the
# anchor and goes remote — word problems are never attempted locally.
_LEAD = r"(?:what is|what's|calculate|compute|evaluate|find|solve|how much is)\s+"
_NUM = r"[\d,]+(?:\.\d+)?"
_PERCENT_Q = re.compile(
    rf"^{_LEAD}({_NUM})\s*(?:%|percent)\s+of\s+({_NUM})\s*[?.!]*$", re.IGNORECASE)
_EXPR_Q = re.compile(
    rf"^{_LEAD}(?:the\s+(?:value|result)\s+of\s+)?([\d\s.,+\-*/^()]+?)\s*[?.!=]*$",
    re.IGNORECASE,
)

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_node(node):
    """Whitelist-only AST arithmetic: numbers and + - * / ** and unary sign."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and (abs(left) > 1000 or abs(right) > 16):
            raise ValueError("power out of safe range")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported node {type(node).__name__}")


def _fmt_number(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{round(x, 6):g}"


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _math_answer(prompt: str) -> Optional[str]:
    p = re.sub(r"\s+", " ", prompt).strip()

    m = _PERCENT_Q.match(p)
    if m:
        pct, base = _to_float(m.group(1)), _to_float(m.group(2))
        value = _fmt_number(pct * base / 100.0)
        return f"{_fmt_number(pct)}% of {_fmt_number(base)} = {value}\nAnswer: {value}"

    m = _EXPR_Q.match(p)
    if m:
        expr = m.group(1).strip()
        # Must be an actual computation: >=2 numbers joined by an operator.
        if not re.search(r"\d\s*[+\-*/^]", expr) or len(re.findall(r"\d+", expr)) < 2:
            return None
        # Digit-grouping commas only; a comma used any other way disqualifies.
        cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
        if "," in cleaned:
            return None
        cleaned = cleaned.replace("^", "**")
        try:
            result = _eval_node(ast.parse(cleaned, mode="eval"))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            return None
        if not isinstance(result, (int, float)) or abs(result) > 1e15:
            return None
        value = _fmt_number(float(result))
        return f"{expr} = {value}\nAnswer: {value}"

    return None


# --------------------------------------------------------------------------- #
# public entry point                                                           #
# --------------------------------------------------------------------------- #
_ANSWERERS = {
    "sentiment": _sentiment_answer,
    "math_reasoning": _math_answer,
}


def try_local_answer(category: str, prompt: str) -> Optional[str]:
    """Return a deterministic zero-token answer, or None to escalate remote.

    Never raises: any internal surprise means "not confident" — the caller
    falls through to the normal Fireworks path.
    """
    answerer = _ANSWERERS.get(category)
    if answerer is None:
        return None
    try:
        return answerer(prompt)
    except Exception:
        return None
