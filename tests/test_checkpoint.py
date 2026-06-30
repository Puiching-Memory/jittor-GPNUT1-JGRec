from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import jittor as jt
from jgrec.checkpoint import get_model_state, load_model_state, save_model_state, set_model_state


class _TinyModel(jt.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = jt.array([[1.0, 2.0], [3.0, 4.0]])
        self.bias = jt.array([0.5, -0.5])


class _NestedModel(jt.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = _TinyModel()
        self.scale = jt.array(2.0)


def test_get_model_state_returns_numpy_arrays() -> None:
    model = _TinyModel()
    state = get_model_state(model)
    assert set(state.keys()) == {"weight", "bias"}
    assert isinstance(state["weight"], np.ndarray)
    assert state["weight"].dtype == np.float32
    np.testing.assert_allclose(state["weight"], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(state["bias"], [0.5, -0.5])


def test_save_and_load_model_state_roundtrip() -> None:
    model = _NestedModel()
    state_before = get_model_state(model)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.npz"
        save_model_state(path, state_before)
        assert path.exists()

        state_loaded = load_model_state(path)
        assert set(state_loaded.keys()) == set(state_before.keys())
        for key in state_before:
            np.testing.assert_allclose(state_before[key], state_loaded[key])


def test_set_model_state_overwrites_parameters() -> None:
    model = _TinyModel()
    state_before = get_model_state(model)

    model.weight = jt.array([[0.0, 0.0], [0.0, 0.0]])
    model.bias = jt.array([0.0, 0.0])
    set_model_state(model, state_before)

    state_after = get_model_state(model)
    for key in state_before:
        np.testing.assert_allclose(state_before[key], state_after[key])


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (3, 4),
        (2, 3, 4),
    ],
)
def test_checkpoint_handles_various_parameter_shapes(shape: tuple[int, ...]) -> None:
    class _ShapeModel(jt.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = jt.array(np.random.randn(*shape).astype(np.float32))

    model = _ShapeModel()
    state = get_model_state(model)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shape.npz"
        save_model_state(path, state)
        loaded = load_model_state(path)
    np.testing.assert_allclose(state["param"], loaded["param"])
