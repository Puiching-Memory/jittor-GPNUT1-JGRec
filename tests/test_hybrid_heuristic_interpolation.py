import numpy as np

from jgrec.core.types import Interaction, InteractionTable, TestQuery
from jgrec.rankers.hybrid_heuristic.interpolation import (
    InterpolationScorer,
    InterpolationWeights,
    fit_weights_on_validation,
)


def _interactions() -> InteractionTable:
    return InteractionTable.from_events(
        [
            Interaction(src=1, dst=10, time=1),
            Interaction(src=1, dst=10, time=2),
            Interaction(src=1, dst=20, time=3),
            Interaction(src=2, dst=10, time=4),
            Interaction(src=2, dst=30, time=5),
        ]
    )


def test_interpolation_edgebank_prefers_repeated_pair():
    scorer = InterpolationScorer(InterpolationWeights(alpha=1.0, beta=0.0, gamma=0.0))
    scorer.fit(_interactions())
    scores = scorer.scores_for_queries([TestQuery(src=1, time=6, candidates=(10, 20, 99))])[0]
    # src=1 与 dst=10 重复交互，EdgeBank 应给候选 10 最高分
    assert scores[0] > scores[1]
    assert scores[1] > scores[2]


def test_interpolation_popularity_ranks_popular_dst_higher():
    scorer = InterpolationScorer(InterpolationWeights(alpha=0.0, beta=1.0, gamma=0.0))
    scorer.fit(_interactions())
    scores = scorer.scores_for_queries([TestQuery(src=9, time=6, candidates=(10, 30, 20))])[0]
    # dst=10 总入边最多(3)，流行度最高
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_interpolation_scores_are_row_normalized():
    scorer = InterpolationScorer(InterpolationWeights(alpha=1.0, beta=1.0, gamma=1.0))
    scorer.fit(_interactions())
    scores = scorer.scores_for_queries([TestQuery(src=1, time=6, candidates=(10, 20, 99))])[0]
    assert scores.min() >= 0.0
    assert scores.max() <= 3.0 + 1e-6


def test_fit_weights_returns_finite_mrr_improvement():
    scorer = InterpolationScorer()
    scorer.fit(_interactions())
    queries = [TestQuery(src=1, time=6, candidates=(10, 20, 99))]
    targets = np.array([0])
    weights = fit_weights_on_validation(scorer, queries, targets, grid_step=0.5)
    total = weights.alpha + weights.beta + weights.gamma
    assert total > 0.0
    # 拟合后正例（候选10）应排第一
    scores = scorer.scores_for_queries(queries)[0]
    assert int(np.argmax(scores)) == 0
