"""
text_metrics.py
---------------
Weighted N-Gram Match (BLEU_weight) for code reproducibility.

Formula follows Section 3.1 of the paper:
  - Python keywords have weight 5; all other tokens have weight 1
  - only unigrams are computed (N=1, w_1=1), consistent with the paper's setting
  - BLEU_weight = BP · exp( Σ_{n=1}^{N} w_n · log p_n )
                = BP · p_1          (degenerates when N=1)

Public interface:
  bleu_weight(candidate, references)     -> float  single candidate vs reference set
  pairwise_bleu_weight(codes)            -> float  mean pairwise similarity among k generations
  summarize_bleu(scores)                 -> (mean, std)
"""

from __future__ import annotations

import keyword
import math
import re
from collections import Counter
from itertools import combinations
from typing import List, Tuple

import numpy as np


# ── Python keyword set (weight 5); all other tokens have weight 1 ──────────────────────
_PYTHON_KEYWORDS: set = set(keyword.kwlist)   # if, for, while, return, def ...
_KEYWORD_W  = 5.0
_DEFAULT_W  = 1.0


# ── Tokenizer ─────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r'[A-Za-z_]\w*|[0-9]+(?:\.[0-9]+)?|[^\w\s]')

def _tokenize(code: str) -> List[str]:
    """
    Code-aware tokenization: extract identifiers / numbers / operators and ignore pure whitespace.
    More accurate than str.split() for unigram statistics on code.
    """
    return _TOKEN_RE.findall(code)


def _token_weight(token: str) -> float:
    return _KEYWORD_W if token in _PYTHON_KEYWORDS else _DEFAULT_W


# ── Core computation ───────────────────────────────────────────────────────────────
def _weighted_counter(tokens: List[str]) -> Counter:
    """Return weighted token counts (Counter value = occurrence count × weight)."""
    c: Counter = Counter()
    for t in tokens:
        c[t] += _token_weight(t)
    return c


def bleu_weight(candidate: str, references: List[str]) -> float:
    """
    Compute the BLEU_weight (unigram) of a single candidate code against a reference set.

    Args:
        candidate:  a generated code string
        references: list of reference code strings (usually [canonical_solution])

    Returns:
        float in [0, 1]
    """
    cand_tokens = _tokenize(candidate)
    if not cand_tokens:
        return 0.0

    # Reference set: maximum weighted count of each token (used for clipping)
    ref_max: Counter = Counter()
    for ref in references:
        ref_wc = _weighted_counter(_tokenize(ref))
        for tok, w in ref_wc.items():
            ref_max[tok] = max(ref_max[tok], w)

    # Weighted counts of the candidate
    cand_wc = _weighted_counter(cand_tokens)

    numerator   = sum(min(w, ref_max[tok]) for tok, w in cand_wc.items())
    denominator = sum(cand_wc.values())

    if denominator == 0:
        return 0.0

    p1 = numerator / denominator

    # Brevity Penalty
    c = len(cand_tokens)
    # Effective reference length: the reference closest to the candidate length (standard BLEU practice)
    ref_lens = [len(_tokenize(r)) for r in references]
    r = min(ref_lens, key=lambda rl: (abs(rl - c), rl))
    bp = 1.0 if c >= r else math.exp(1.0 - r / c)

    return bp * p1


def pairwise_bleu_weight(codes: List[str]) -> float:
    """
    Compute the mean pairwise BLEU_weight of the k generated code samples, measuring internal textual consistency
    (high → outputs are similar across generations, strong reproducibility).

    Each pair (i, j) is computed in both directions and averaged:
        sim(i,j) = (bleu(i→j) + bleu(j→i)) / 2

    Returns:
        float: nan if fewer than 2 parseable code samples
    """
    valid = [c for c in codes if c and c.strip()]
    if len(valid) < 2:
        return float("nan")

    sims = []
    for i, j in combinations(range(len(valid)), 2):
        fwd = bleu_weight(valid[i], [valid[j]])
        bwd = bleu_weight(valid[j], [valid[i]])
        sims.append((fwd + bwd) / 2.0)

    return float(np.mean(sims)) if sims else float("nan")


def bleu_weight_vs_ref(codes: List[str], reference: str) -> Tuple[float, float]:
    """
    Compute the BLEU_weight of each of the k generated code samples against the reference code (canonical_solution),
    returning (mean, std). Measures the average level and stability of generation quality.

    Returns:
        (mean, std): mean and standard deviation
    """
    scores = [bleu_weight(c, [reference]) for c in codes if c and c.strip()]
    if not scores:
        return float("nan"), float("nan")
    arr = np.array(scores, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize_bleu(scores: List[float]) -> Tuple[float, float]:
    """
    Cross-task summary: returns (mean, std), skipping nan values.
    """
    valid = [s for s in scores if not math.isnan(s)]
    if not valid:
        return float("nan"), float("nan")
    arr = np.array(valid, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))
