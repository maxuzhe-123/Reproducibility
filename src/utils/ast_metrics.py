"""
ast_metrics.py
--------------
AST-based structure reproducibility metrics for code generation.

Borrows the AST traversal idea from the reference code (ast.If / ast.Compare feature extraction)
and extends it into structural similarity metrics for reproducibility research:

① extract_ast_profile(code)
   Extract the structural profile of the code:
   - node_counts   : Counter of all AST node types
   - control_flow  : flat sequence of control-flow nodes (preserving order)
   - if_features   : list of (variable name, comparison operator) for all if-compare conditions
     (directly borrowed from find_if_else_features in the reference code)

② ast_cosine_similarity(profile_a, profile_b)
   Cosine similarity based on node_counts ∈ [0, 1]

③ pairwise_ast_similarity(codes)
   Mean pairwise AST similarity among the k generations (structural consistency)

④ ast_similarity_vs_ref(codes, reference)
   AST similarity of each generation vs the reference code → (mean, std)

⑤ summarize_ast(scores)
   Cross-task summary → (mean, std)

The public interface is consistent with text_metrics.py.
"""

from __future__ import annotations

import ast
import math
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Control-flow node types (used to extract structural sequences) ────────────────────────────────────
_CONTROL_FLOW_TYPES = (
    ast.If, ast.For, ast.While, ast.Try,
    ast.With, ast.Match,           # Python 3.10+ match
    ast.ExceptHandler,
    ast.AsyncFor, ast.AsyncWith,
)

# ── Mapping of comparison operator names ─────────────────────────────────────────────────────
_CMP_OP_NAMES: Dict[type, str] = {
    ast.Eq:    "==",
    ast.NotEq: "!=",
    ast.Lt:    "<",
    ast.LtE:   "<=",
    ast.Gt:    ">",
    ast.GtE:   ">=",
    ast.Is:    "is",
    ast.IsNot: "is not",
    ast.In:    "in",
    ast.NotIn: "not in",
}


# ── Internal AST traversal functions ──────────────────────────────────────────────────────

def _collect_node_counts(tree: ast.AST) -> Counter:
    """Traverse the AST and count occurrences of each node type."""
    return Counter(type(node).__name__ for node in ast.walk(tree))


def _collect_control_flow_sequence(tree: ast.AST) -> List[str]:
    """
    Depth-first traversal that extracts the sequence of control-flow node names in order of appearance.
    For example, ['If', 'For', 'If', 'While'] represents one structural pattern.
    """
    seq: List[str] = []

    def _visit(node: ast.AST):
        if isinstance(node, _CONTROL_FLOW_TYPES):
            seq.append(type(node).__name__)
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree)
    return seq


def _collect_if_features(tree: ast.AST) -> List[Dict[str, Any]]:
    """
    Following find_if_else_features_and_values in the reference code:
    traverse all ast.If nodes and extract condition features of the form `if <variable> <comparator> <value>`.

    Returns a list whose elements are:
        {
            "var":   variable name (str),
            "op":    comparison operator (str),
            "value": AST dump of the right-hand value (str),
        }
    """
    features: List[Dict[str, Any]] = []

    def _visit(node: ast.AST):
        if isinstance(node, ast.If):
            test = node.test
            # Of the form `if var op value` (single comparison)
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and len(test.ops) == 1
                and len(test.comparators) == 1
            ):
                var_name = test.left.id
                op_name  = _CMP_OP_NAMES.get(type(test.ops[0]), "?")
                val_dump = ast.dump(test.comparators[0])
                features.append({
                    "var":   var_name,
                    "op":    op_name,
                    "value": val_dump,
                })
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree)
    return features


# ── Public interface ───────────────────────────────────────────────────────────────

def extract_ast_profile(code: str) -> Optional[Dict[str, Any]]:
    """
    Parse the code and extract its structural profile. Returns None if parsing fails (SyntaxError, etc.).

    Returns:
        {
            "node_counts":      Counter,       # node type counts
            "control_flow_seq": List[str],     # ordered sequence of control-flow nodes
            "if_features":      List[dict],    # if-compare condition features
            "parse_ok":         bool,
        }
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return None

    return {
        "node_counts":      _collect_node_counts(tree),
        "control_flow_seq": _collect_control_flow_sequence(tree),
        "if_features":      _collect_if_features(tree),
        "parse_ok":         True,
    }


def ast_cosine_similarity(profile_a: Dict, profile_b: Dict) -> float:
    """
    Cosine similarity based on node_counts.
    The closer the node type distributions of the two profiles, the higher the score (more reproducible).

    Returns:
        float ∈ [0, 1]; returns 0.0 if either profile is None
    """
    if profile_a is None or profile_b is None:
        return 0.0

    ca: Counter = profile_a["node_counts"]
    cb: Counter = profile_b["node_counts"]

    if not ca or not cb:
        return 0.0

    keys = set(ca) | set(cb)
    dot   = sum(ca.get(k, 0) * cb.get(k, 0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def control_flow_jaccard(profile_a: Dict, profile_b: Dict) -> float:
    """
    Jaccard similarity of the sets of control-flow node types (ignoring order and counts).
    For example, if both functions use [If, For], the Jaccard is 1.0.

    Returns:
        float ∈ [0, 1]
    """
    if profile_a is None or profile_b is None:
        return 0.0
    sa = set(profile_a["control_flow_seq"])
    sb = set(profile_b["control_flow_seq"])
    if not sa and not sb:
        return 1.0          # Neither has control flow; the structures are identical
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pairwise_ast_similarity(codes: List[str]) -> float:
    """
    Compute the mean pairwise AST cosine similarity of the k generated code samples.
    - High score (→1): the code structure is consistent across generations, strong reproducibility
    - Low score (→0): large structural differences between generations

    Returns:
        float: nan if fewer than 2 parseable code samples
    """
    profiles = [extract_ast_profile(c) for c in codes if c and c.strip()]
    valid = [p for p in profiles if p is not None]
    if len(valid) < 2:
        return float("nan")

    sims = [
        ast_cosine_similarity(valid[i], valid[j])
        for i, j in combinations(range(len(valid)), 2)
    ]
    return float(np.mean(sims)) if sims else float("nan")


def ast_similarity_vs_ref(codes: List[str], reference: str) -> Tuple[float, float]:
    """
    AST cosine similarity of each generated code sample vs the reference code (canonical_solution),
    returning (mean, std).

    Returns:
        (mean, std): (nan, nan) if no parseable code
    """
    ref_profile = extract_ast_profile(reference)
    if ref_profile is None:
        return float("nan"), float("nan")

    scores = [
        ast_cosine_similarity(extract_ast_profile(c), ref_profile)
        for c in codes
        if c and c.strip()
    ]
    if not scores:
        return float("nan"), float("nan")
    arr = np.array(scores, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize_ast(scores: List[float]) -> Tuple[float, float]:
    """
    Summarize AST similarity across tasks, returning (mean, std) and skipping nan.
    """
    valid = [s for s in scores if not math.isnan(s)]
    if not valid:
        return float("nan"), float("nan")
    arr = np.array(valid, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))
