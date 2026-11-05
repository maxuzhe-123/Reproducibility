import re
import textwrap
FENCE = re.compile(r"```(?:python)?[^\n]*\n(.*?)```", re.DOTALL|re.IGNORECASE)
# Leftover opening code fence (e.g., the model output ```python without closing it)
FENCE_OPEN = re.compile(r"^```(?:python)?[^\n]*\n", re.IGNORECASE)

def extract_code(text: str) -> str:
    # Prefer extracting the ```python...``` block
    m = FENCE.search(text)
    if m:
        code = m.group(1)
    else:
        # Remove a possibly leftover opening ``` marker
        text = FENCE_OPEN.sub("", text.lstrip())
        if "def " in text:
            code = text[text.find("def "):]
        else:
            code = text
    # CoT format ([BEGIN]...[DONE]): the content before [BEGIN] is the problem/thinking echoed by the model,
    # and the real answer comes after [BEGIN]. Discard everything up to and including the last [BEGIN] marker.
    # Otherwise, body-only completion would leave a "[BEGIN]" line, causing indentation/syntax errors after prepending the def signature.
    if "[BEGIN]" in code:
        code = code[code.rindex("[BEGIN]") + len("[BEGIN]"):]
    # Truncate common end markers
    for stop in ("Stop.", "[DONE]", "[END]", "```"):
        if stop in code:
            code = code[:code.index(stop)]
    # Use dedent to remove common indentation (strip() only removes the first line's indentation and would misalign the relative indentation of later lines)
    return textwrap.dedent(code).strip()


# ============================================================
# LLM-as-a-Judge verdict parsing (CodeJudgeBench, pairwise)
# ============================================================
# Identical to PairwiseModel.extract_response by the CodeJudgeBench authors:
#   - verdict markers [[A>B]] / boxed{A>B} → A is better; [[B>A]] / boxed{B>A} → B is better
#   - if the text contains </think>, only the part after it is used
#   - take the last occurrence of each pattern (rfind); the later one wins
#   - if neither matches (both rfind are -1, equal) → return "" (treated as an invalid verdict/wrong)
#   - if generation did not finish normally due to length etc. (finish_reason != 'stop') → return ""
CODEJUDGE_RESPONSE_PATTERNS_A = ["[[A>B]]", "boxed{A>B}"]
CODEJUDGE_RESPONSE_PATTERNS_B = ["[[B>A]]", "boxed{B>A}"]


def parse_codejudge_verdict(text: str, finish_reason: str = "stop") -> str:
    """Parse a pairwise verdict, returning 'A' / 'B' / '' (empty string means no valid verdict).

    Replicates the original logic verbatim (including the finish_reason fallback, </think> truncation, and rfind position comparison).
    Note: lm.generate in this repository does not return finish_reason, so 'stop' is assumed by default;
    if the output is truncated by max_new_tokens before a verdict is given, '' is naturally returned because no marker is found.
    """
    if finish_reason != "stop":
        return ""
    if not text:
        return ""

    if "</think>" in text:
        text = text.split("</think>")[1]

    pred_a = max(text.rfind(p) for p in CODEJUDGE_RESPONSE_PATTERNS_A)
    pred_b = max(text.rfind(p) for p in CODEJUDGE_RESPONSE_PATTERNS_B)

    if pred_a > pred_b:
        return "A"
    elif pred_a < pred_b:
        return "B"
    return ""
