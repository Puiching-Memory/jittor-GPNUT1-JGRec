from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import jittor as jt
from jgrec.rankers.craft.ranker import CRAFTBaselineRanker
from jgrec.rankers.hybrid.ranker import TemporalHybridRanker
from jgrec.rankers.temporal_graph.ranker import TemporalGraphRanker


class _TinyModel(jt.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = jt.array([[1.0, 2.0], [3.0, 4.0]])
        self.bias = jt.array([0.5, -0.5])


def test_temporal_graph_save_load_checkpoint() -> None:
    ranker = TemporalGraphRanker()
    ranker.model = _TinyModel()
    before = {k: np.array(v) for k, v in ranker.model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tg.npz"
        ranker.save_checkpoint(path)
        ranker.model.weight = jt.array([[0.0, 0.0], [0.0, 0.0]])
        ranker.load_checkpoint(path)
        after = {k: np.array(v) for k, v in ranker.model.state_dict().items()}

    for key in before:
        np.testing.assert_allclose(before[key], after[key])


def test_craft_save_load_checkpoint() -> None:
    ranker = CRAFTBaselineRanker()
    ranker.model = _TinyModel()
    before = {k: np.array(v) for k, v in ranker.model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "craft.npz"
        ranker.save_checkpoint(path)
        ranker.model.weight = jt.array([[0.0, 0.0], [0.0, 0.0]])
        ranker.load_checkpoint(path)
        after = {k: np.array(v) for k, v in ranker.model.state_dict().items()}

    for key in before:
        np.testing.assert_allclose(before[key], after[key])


def test_hybrid_save_load_checkpoint() -> None:
    ranker = TemporalHybridRanker()
    ranker.fusion = _TinyModel()
    before = {k: np.array(v) for k, v in ranker.fusion.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hybrid.npz"
        ranker.save_checkpoint(path)
        ranker.fusion.weight = jt.array([[0.0, 0.0], [0.0, 0.0]])
        ranker.load_checkpoint(path)
        after = {k: np.array(v) for k, v in ranker.fusion.state_dict().items()}

    for key in before:
        np.testing.assert_allclose(before[key], after[key])


def test_temporal_graph_save_checkpoint_without_model_raises() -> None:
    ranker = TemporalGraphRanker()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError, match="not fitted"):
            ranker.save_checkpoint(Path(tmp) / "tg.npz")


def test_craft_save_checkpoint_without_model_raises() -> None:
    ranker = CRAFTBaselineRanker()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError, match="not fitted"):
            ranker.save_checkpoint(Path(tmp) / "craft.npz")


def test_hybrid_save_checkpoint_without_fusion_raises() -> None:
    ranker = TemporalHybridRanker()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError, match="not fitted"):
            ranker.save_checkpoint(Path(tmp) / "hybrid.npz")
