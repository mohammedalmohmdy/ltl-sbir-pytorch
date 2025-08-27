
import numpy as np
from typing import List

def average_precision(relevance: List[int]) -> float:
    """AP for a single query given a binary relevance list sorted by score descending."""
    if not relevance:
        return 0.0
    hits, cum_prec = 0, 0.0
    for i, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            cum_prec += hits / i
    return 0.0 if hits == 0 else cum_prec / hits

def mean_average_precision(all_relevances: List[List[int]]) -> float:
    """Mean of AP over all queries."""
    if not all_relevances:
        return 0.0
    return float(np.mean([average_precision(r) for r in all_relevances]))

def cmc_curve(all_relevances: List[List[int]], max_rank: int = 100) -> List[float]:
    """CMC: fraction of queries having at least one relevant item within top‑r."""
    cmc = np.zeros(max_rank, dtype=np.float32)
    Q = len(all_relevances)
    for rel in all_relevances:
        found_at = None
        for i, v in enumerate(rel[:max_rank]):
            if v == 1:
                found_at = i; break
        if found_at is not None:
            cmc[found_at:] += 1.0
    return [0.0] * max_rank if Q == 0 else list((cmc / Q).tolist())

def topk_binary_relevance(sorted_labels, query_label: int, k: int = 100) -> List[int]:
    """Utility to build a binary relevance list of length k from ranked labels."""
    return [1 if lbl == query_label else 0 for lbl in sorted_labels[:k]]
