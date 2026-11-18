from typing import List, Tuple, Dict, Any


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Combines multiple ranked lists of (item_id, score) using Reciprocal Rank Fusion.
    RRF score for item d = sum_i(1 / (k + rank_i(d))).
    Returns fused list of (item_id, fused_score) sorted descending by score.
    """
    rrf_scores: Dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked_list, start=1):
            if item_id not in rrf_scores:
                rrf_scores[item_id] = 0.0
            rrf_scores[item_id] += 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
