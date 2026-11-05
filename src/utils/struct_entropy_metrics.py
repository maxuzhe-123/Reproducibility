"""
struct_entropy_metrics.py
--------------------------
Faithful rewrite of the author's reference implementation.

Core logic is transcribed directly from the author's code:
  extract_subtrees / extract_subtrees_withvalue
  _normalize_vector / kullback_leibler_divergence / jensen_shannon_divergence
  compute_cte_jsd_struct  →  _compute_jsd
  compute_sce_norm_struct →  _compute_sce   (bug in original fixed: sce_norm was
                                             never assigned before return; formula
                                             taken from compute_sce_norm_value)

API compatibility note
----------------------
The installed tree-sitter-language-pack ships tree-sitter ≥ 0.24, whose Python
bindings expose Node attributes as *methods* (.kind(), .child_count(), …) rather
than properties (.type, .children, …) used in the author's code.  Thin wrappers
(_nkind, _nchildren, _ncc, _nbytes) bridge the difference transparently.

JSD uses natural log (np.log) — exactly as in the author's
kullback_leibler_divergence — so 1-JSD is in [1-ln2, 1] ≈ [0.307, 1].
SCE uses log base-2 (math.log2) — exactly as in the author's formula.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import List, Tuple

import numpy as np
from tree_sitter_language_pack import get_parser

# ── Default hyperparameters (match author's defaults) ────────────────────────
DEFAULT_DEPTH: int   = 30        # author uses max_depth=30
DEFAULT_EPS:   float = 1e-10     # author uses eps=1e-10
DEFAULT_LANG:  str   = "python"  # HumanEval / HumanEval-X are Python


# ═════════════════════════════════════════════════════════════════════════════
# tree-sitter version-compatibility helpers
# The installed binding exposes everything as callable methods, not properties.
# ═════════════════════════════════════════════════════════════════════════════

def _nkind(node) -> str:
    """node.type  in old API  →  node.kind() in new API."""
    k = node.kind
    return k() if callable(k) else k

def _ncc(node) -> int:
    """node.child_count  (property → method)."""
    cc = node.child_count
    return cc() if callable(cc) else cc

def _nchildren(node) -> list:
    """node.children  (property → iterated via .child(i))."""
    if hasattr(node, "children"):
        ch = node.children
        return ch() if callable(ch) else ch
    return [node.child(i) for i in range(_ncc(node))]

def _nbytes(node) -> tuple:
    """Return (start_byte, end_byte) regardless of API version."""
    sb = node.start_byte
    eb = node.end_byte
    return (sb() if callable(sb) else sb,
            eb() if callable(eb) else eb)

def _parse_root(code: str, lang: str):
    """Parse *code* string and return the root Node."""
    parser = get_parser(lang)
    tree   = parser.parse(code)       # new API accepts str directly
    rn     = tree.root_node
    return rn() if callable(rn) else rn


# ═════════════════════════════════════════════════════════════════════════════
# Author's core helpers  (copied faithfully)
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Author's _normalize_vector.
    Adds ε=1e-10 to every element, then normalises to sum-to-1.
    """
    if np.any(v < 0):
        raise ValueError("Vector elements must be non-negative.")
    epsilon = 1e-10
    v = v + epsilon
    return v / np.sum(v)


def _kullback_leibler_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Author's kullback_leibler_divergence.
    Uses np.log (natural logarithm) — matches author's code exactly.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("Input arrays must have the same shape.")
    if not np.isclose(np.sum(p), 1.0) or not np.isclose(np.sum(q), 1.0):
        p = _normalize_vector(p)
        q = _normalize_vector(q)
    return float(np.sum(p * np.log(p / q)))


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Author's jensen_shannon_divergence.
    Internally normalises both vectors (adds ε, renormalises), then computes
    JSD via two KL divergences.  Uses natural log.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("Input arrays must have the same shape.")
    p_norm = _normalize_vector(p.copy())
    q_norm = _normalize_vector(q.copy())
    m = 0.5 * (p_norm + q_norm)
    kl_pm  = _kullback_leibler_divergence(p_norm, m)
    kl_qm  = _kullback_leibler_divergence(q_norm, m)
    return 0.5 * (kl_pm + kl_qm)


# ═════════════════════════════════════════════════════════════════════════════
# Author's subtree extraction  (structure-only)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_subtrees(root_node, max_depth: int) -> Counter:
    """
    Author's extract_subtrees.
    DFS up to max_depth; each node represented as
        (node_type, (child1_type, child2_type, ...))
    """
    counter: Counter = Counter()

    def dfs(node, depth: int) -> None:
        if depth > max_depth:
            return
        children   = _nchildren(node)
        repr_tuple = (_nkind(node), tuple(_nkind(c) for c in children))
        counter[repr_tuple] += 1
        for child in children:
            dfs(child, depth + 1)

    dfs(root_node, 1)
    return counter


# ═════════════════════════════════════════════════════════════════════════════
# Author's subtree extraction  (structure + leaf value)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_subtrees_withvalue(
    root_node,
    source_code: bytes,
    max_depth: int,
) -> Counter:
    """
    Author's extract_subtrees_withvalue.
    Leaf nodes (child_count == 0): (node_type, (leaf_text,))
    Internal nodes               : (node_type, (child1_type, ...))
    """
    counter: Counter = Counter()

    def dfs(node, depth: int) -> None:
        if depth > max_depth:
            return
        node_kind = _nkind(node)

        if _ncc(node) == 0:                        # leaf
            start, end = _nbytes(node)
            leaf_text  = source_code[start:end].decode("utf8", errors="ignore")
            counter[(node_kind, (leaf_text,))] += 1
            return

        children   = _nchildren(node)
        repr_tuple = (node_kind, tuple(_nkind(c) for c in children))
        counter[repr_tuple] += 1
        for child in children:
            dfs(child, depth + 1)

    dfs(root_node, 1)
    return counter


# ═════════════════════════════════════════════════════════════════════════════
# Author's metric computation  (factored out from compute_* top-level fns)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_sce(
    freqA: Counter,
    freqB: Counter,
    eps:   float = DEFAULT_EPS,
) -> float:
    """
    Factored from author's compute_sce_norm_struct / compute_sce_norm_value.

    S_CE(P, Q) = H(Q) / H(P, Q)
    where H(P,Q) = -Σ P(u) log₂ Q(u)  and  H(Q) = -Σ Q(u) log₂ Q(u).

    Smoothing (author's Phase-2 SCE):
        P(u) = c_A(u)/n_A  if c_A(u) > 0,  else  ε
        Q(u) = c_B(u)/n_B  if c_B(u) > 0,  else  ε
    (no renormalisation — ε contributions negligible)

    Bug fix: compute_sce_norm_struct returned sce_norm without ever defining
    it; formula taken from the correct compute_sce_norm_value.
    """
    nA = sum(freqA.values())
    nB = sum(freqB.values())
    if nA == 0 or nB == 0:
        return float("nan")

    U = set(freqA) | set(freqB)
    P = [freqA[s] / nA if freqA[s] > 0 else eps for s in U]
    Q = [freqB[s] / nB if freqB[s] > 0 else eps for s in U]

    cross_entropy = -sum(p * math.log2(q) for p, q in zip(P, Q))  # H(P,Q)
    entropy_Q     = -sum(q * math.log2(q) for q in Q)              # H(Q)

    return entropy_Q / cross_entropy if cross_entropy > 0 else 0.0


def _compute_jsd(
    freqA: Counter,
    freqB: Counter,
    eps:   float = DEFAULT_EPS,
) -> float:
    """
    Factored from author's compute_cte_jsd_struct / compute_cte_jsd_value.

    Returns 1 - JSD(P, Q).

    Smoothing (author's Phase-2 JSD):
        P_vec(u) = c_A(u)/n_A                       (no smoothing on P)
        Q_vec(u) = max(c_B(u)/n_B, ε), renormalised

    Then _jensen_shannon_divergence internally adds ε to both and renormalises
    again (author's _normalize_vector).

    Uses np.log (natural log) — matches author's kullback_leibler_divergence.
    So 1-JSD ∈ [1-ln(2), 1] ≈ [0.307, 1].
    """
    nA = sum(freqA.values())
    nB = sum(freqB.values())
    if nA == 0 or nB == 0:
        return float("nan")

    labels = sorted(set(freqA) | set(freqB))            # author uses sorted
    P_vec  = np.array([freqA.get(s, 0) / nA for s in labels], dtype=float)
    Q_vec  = np.array([freqB.get(s, 0) / nB for s in labels], dtype=float)

    Q_vec  = np.maximum(Q_vec, eps)
    Q_vec /= Q_vec.sum()                                 # renormalise Q

    jsd_val = _jensen_shannon_divergence(P_vec, Q_vec)
    if np.isnan(jsd_val):                                # author's NaN fallback
        jsd_val = 1.0

    return 1.0 - jsd_val


# ═════════════════════════════════════════════════════════════════════════════
# Public interface — Phase 1 multiset extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_subtree_multiset(
    code:  str,
    depth: int = DEFAULT_DEPTH,
    lang:  str = DEFAULT_LANG,
    mode:  str = "structure",    # "structure" | "value"
) -> Counter:
    """
    Parse *code* and return the multiset of subtree symbols (Phase 1).

    mode="structure" → author's extract_subtrees       (σ_struct)
    mode="value"     → author's extract_subtrees_withvalue (σ_value)
    Returns empty Counter on parse failure.
    """
    if not code or not code.strip():
        return Counter()
    try:
        root = _parse_root(code, lang)
        if mode == "value":
            return _extract_subtrees_withvalue(root, code.encode("utf8"), depth)
        return _extract_subtrees(root, depth)
    except Exception:
        return Counter()


# ═════════════════════════════════════════════════════════════════════════════
# Public interface — pairwise reproducibility metrics
# ═════════════════════════════════════════════════════════════════════════════

def pairwise_jsd(
    codes: List[str],
    depth: int = DEFAULT_DEPTH,
    lang:  str = DEFAULT_LANG,
    mode:  str = "structure",
) -> float:
    """
    Mean 1-JSD across all C(k, 2) pairs of generated codes.
    Higher → more structurally consistent → more reproducible.
    Returns nan if fewer than 2 parseable codes.
    """
    multisets = [extract_subtree_multiset(c, depth, lang, mode)
                 for c in codes if c and c.strip()]
    valid = [ms for ms in multisets if sum(ms.values()) > 0]
    if len(valid) < 2:
        return float("nan")

    scores = []
    for i, j in combinations(range(len(valid)), 2):
        s = _compute_jsd(valid[i], valid[j])
        if not math.isnan(s):
            scores.append(s)

    return float(np.mean(scores)) if scores else float("nan")


def pairwise_sce(
    codes: List[str],
    depth: int   = DEFAULT_DEPTH,
    lang:  str   = DEFAULT_LANG,
    mode:  str   = "structure",
    eps:   float = DEFAULT_EPS,
) -> float:
    """
    Mean pairwise SCE across all C(k, 2) pairs.
    SCE is asymmetric → symmetrised as ½[SCE(A,B) + SCE(B,A)].
    Returns nan if fewer than 2 parseable codes.
    """
    multisets = [extract_subtree_multiset(c, depth, lang, mode)
                 for c in codes if c and c.strip()]
    valid = [ms for ms in multisets if sum(ms.values()) > 0]
    if len(valid) < 2:
        return float("nan")

    scores = []
    for i, j in combinations(range(len(valid)), 2):
        s_ij = _compute_sce(valid[i], valid[j], eps)
        s_ji = _compute_sce(valid[j], valid[i], eps)
        if not (math.isnan(s_ij) or math.isnan(s_ji)):
            scores.append(0.5 * (s_ij + s_ji))

    return float(np.mean(scores)) if scores else float("nan")


# ═════════════════════════════════════════════════════════════════════════════
# Public interface — vs-reference metrics
# ═════════════════════════════════════════════════════════════════════════════

def jsd_vs_ref(
    codes:     List[str],
    reference: str,
    depth:     int = DEFAULT_DEPTH,
    lang:      str = DEFAULT_LANG,
    mode:      str = "structure",
) -> Tuple[float, float]:
    """1-JSD between each generated code and the reference. Returns (mean, std)."""
    ref_ms = extract_subtree_multiset(reference, depth, lang, mode)
    if sum(ref_ms.values()) == 0:
        return float("nan"), float("nan")

    scores: List[float] = []
    for c in codes:
        ms = extract_subtree_multiset(c, depth, lang, mode)
        if sum(ms.values()) == 0:
            continue
        s = _compute_jsd(ref_ms, ms)
        if not math.isnan(s):
            scores.append(s)

    if not scores:
        return float("nan"), float("nan")
    arr = np.array(scores, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def sce_vs_ref(
    codes:     List[str],
    reference: str,
    depth:     int   = DEFAULT_DEPTH,
    lang:      str   = DEFAULT_LANG,
    mode:      str   = "structure",
    eps:       float = DEFAULT_EPS,
) -> Tuple[float, float]:
    """S_CE(P_ref, Q_gen) for each generated code vs. reference. Returns (mean, std)."""
    ref_ms = extract_subtree_multiset(reference, depth, lang, mode)
    if sum(ref_ms.values()) == 0:
        return float("nan"), float("nan")

    scores: List[float] = []
    for c in codes:
        ms = extract_subtree_multiset(c, depth, lang, mode)
        if sum(ms.values()) == 0:
            continue
        s = _compute_sce(ref_ms, ms, eps)
        if not math.isnan(s):
            scores.append(s)

    if not scores:
        return float("nan"), float("nan")
    arr = np.array(scores, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


# ═════════════════════════════════════════════════════════════════════════════
# Aggregation helper
# ═════════════════════════════════════════════════════════════════════════════

def summarize_struct_entropy(scores: List[float]) -> Tuple[float, float]:
    """Aggregate per-task scores → (mean, std), skipping nan."""
    valid = [s for s in scores if not math.isnan(s)]
    if not valid:
        return float("nan"), float("nan")
    arr = np.array(valid, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))
