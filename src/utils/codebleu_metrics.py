# utils/codebleu_metrics.py
"""
Self-CodeBLEU (pairwise CodeBLEU among multiple generations)

This metric mirrors SDI / RDR: it is a *reproducibility* metric computed
across the k generations produced for the SAME prompt — it does NOT compare
against a ground-truth / reference solution.

For one example:
    - Input: multiple generated code strings for the same prompt
    - Step 1: remove duplicate generations (identical text collapses)
    - Step 2: if fewer than 2 unique generations remain → the model is
              perfectly consistent → CodeBLEU(x) = 1.0
    - Step 3: for each unordered pair (u_i, u_j) compute a symmetric CodeBLEU
              score = mean( CodeBLEU(ref=u_i, hyp=u_j),
                            CodeBLEU(ref=u_j, hyp=u_i) )
              (CodeBLEU is direction-sensitive, so we symmetrize for a
               well-defined self-similarity)
    - Step 4: CodeBLEU(x) = average over all pairs

Higher self-CodeBLEU  ->  more reproducible / consistent outputs.

The official `codebleu` package is used (n-gram + weighted n-gram + AST
syntax match + dataflow match). Install with:
    pip install codebleu tree-sitter tree-sitter-python
"""

from __future__ import annotations
from itertools import combinations
from typing import List, Tuple, Dict
import numpy as np

# Standard CodeBLEU component weights:
# (n-gram, weighted n-gram, syntax AST, dataflow)
_DEFAULT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
_LANG = "python"

# All components returned by calc_codebleu: overall score + four components
# Same order as in the codebleu package, for direct unpacking and storage.
CODEBLEU_KEYS = (
    "codebleu",                    # Overall score (weighted sum of the four components)
    "ngram_match_score",           # BLEU n-gram match
    "weighted_ngram_match_score",  # Keyword-weighted n-gram match
    "syntax_match_score",          # AST syntax subtree match
    "dataflow_match_score",        # Data-flow match
)


def _pair_codebleu_detailed(a: str, b: str, weights=_DEFAULT_WEIGHTS) -> Dict[str, float]:
    """
    Symmetric CodeBLEU between two code strings, returning ALL five scores
    (codebleu + 4 components). On parse/eval failure returns 0.0 for every score.
    """
    from codebleu import calc_codebleu  # lazy import: only needed when metric runs

    try:
        r1 = calc_codebleu([a], [b], lang=_LANG, weights=weights)
        r2 = calc_codebleu([b], [a], lang=_LANG, weights=weights)
        # CodeBLEU is directional → average both directions to obtain a well-defined self-similarity
        return {kk: float((r1[kk] + r2[kk]) / 2.0) for kk in CODEBLEU_KEYS}
    except Exception:
        # Empty / unparseable code → treat the pair as fully dissimilar
        return {kk: 0.0 for kk in CODEBLEU_KEYS}


def compute_codebleu_detailed(pred_codes: List[str], weights=_DEFAULT_WEIGHTS) -> Dict[str, float]:
    """
    Compute self-CodeBLEU for a single example, returning ALL five scores.

    Returns:
        dict: {codebleu, ngram_match_score, weighted_ngram_match_score,
               syntax_match_score, dataflow_match_score}, each averaged over pairs.
    """
    # 1) No deduplication (consistent with the current SDI/RDR implementation)
    uniq = [c for c in pred_codes if c is not None]
    m = len(uniq)

    # 2) Fewer than 2 outputs → fully consistent → maximal reproducibility → all scores are 1.0
    if m < 2:
        return {kk: 1.0 for kk in CODEBLEU_KEYS}

    # 3) Compute symmetric CodeBLEU pairwise (including all components)
    pair_scores = [
        _pair_codebleu_detailed(uniq[i], uniq[j], weights=weights)
        for i, j in combinations(range(m), 2)
    ]
    if not pair_scores:
        return {kk: 1.0 for kk in CODEBLEU_KEYS}

    # 4) Average each component over all pairs
    return {kk: float(np.mean([ps[kk] for ps in pair_scores])) for kk in CODEBLEU_KEYS}


def compute_codebleu(pred_codes: List[str], weights=_DEFAULT_WEIGHTS) -> float:
    """
    Compute self-CodeBLEU (overall scalar score) for a single example.
    Kept for backward compatibility: returns the 'codebleu' scalar from the detailed result.
    """
    return compute_codebleu_detailed(pred_codes, weights=weights)["codebleu"]


def summarize_codebleu(cb_list: List[float]) -> Tuple[float, float]:
    """
    Dataset-level aggregation for self-CodeBLEU (overall scalar score).
    Returns (mean, std).
    """
    if not cb_list:
        return 0.0, 0.0
    arr = np.asarray(cb_list, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize_codebleu_components(
    detail_list: List[Dict[str, float]],
    keys=CODEBLEU_KEYS,
) -> Dict[str, Tuple[float, float]]:
    """
    Dataset-level aggregation for each CodeBLEU component.

    Args:
        detail_list: list of dicts returned by compute_codebleu_detailed(...) for each task
    Returns:
        dict: {component_name: (mean, std)}, covering codebleu and the four components.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for kk in keys:
        vals = [d[kk] for d in detail_list if isinstance(d, dict) and kk in d]
        if vals:
            arr = np.asarray(vals, dtype=float)
            out[kk] = (float(arr.mean()), float(arr.std(ddof=0)))
        else:
            out[kk] = (0.0, 0.0)
    return out
