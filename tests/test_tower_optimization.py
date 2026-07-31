from __future__ import annotations

import pytest

from jgrec.rankers.common.optimization import (
    set_tower_optimizer_learning_rate,
    tower_epoch_learning_rate,
)
from jgrec.rankers.hybrid.config import TrainingConfig


def test_cosine_tower_learning_rate_reaches_frozen_endpoints() -> None:
    initial_lr = 1e-3

    assert tower_epoch_learning_rate(
        initial_lr=initial_lr,
        epoch=1,
        total_epochs=5,
        schedule="cosine",
        min_lr_ratio=0.1,
    ) == pytest.approx(initial_lr)
    assert tower_epoch_learning_rate(
        initial_lr=initial_lr,
        epoch=3,
        total_epochs=5,
        schedule="cosine",
        min_lr_ratio=0.1,
    ) == pytest.approx(initial_lr * 0.55)
    assert tower_epoch_learning_rate(
        initial_lr=initial_lr,
        epoch=5,
        total_epochs=5,
        schedule="cosine",
        min_lr_ratio=0.1,
    ) == pytest.approx(initial_lr * 0.1)


def test_constant_and_single_epoch_tower_learning_rates_preserve_initial_lr() -> None:
    assert tower_epoch_learning_rate(
        initial_lr=2e-3,
        epoch=4,
        total_epochs=8,
        schedule="constant",
        min_lr_ratio=0.1,
    ) == pytest.approx(2e-3)
    assert tower_epoch_learning_rate(
        initial_lr=2e-3,
        epoch=1,
        total_epochs=1,
        schedule="cosine",
        min_lr_ratio=0.1,
    ) == pytest.approx(2e-3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "initial_lr": 0.0,
                "epoch": 1,
                "total_epochs": 2,
                "schedule": "cosine",
                "min_lr_ratio": 0.1,
            },
            "initial_lr",
        ),
        (
            {
                "initial_lr": 1e-3,
                "epoch": 0,
                "total_epochs": 2,
                "schedule": "cosine",
                "min_lr_ratio": 0.1,
            },
            "epoch",
        ),
        (
            {
                "initial_lr": 1e-3,
                "epoch": 1,
                "total_epochs": 2,
                "schedule": "mystery",
                "min_lr_ratio": 0.1,
            },
            "schedule",
        ),
        (
            {
                "initial_lr": 1e-3,
                "epoch": 1,
                "total_epochs": 2,
                "schedule": "cosine",
                "min_lr_ratio": 1.1,
            },
            "min_lr_ratio",
        ),
    ],
)
def test_tower_learning_rate_rejects_invalid_contract(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tower_epoch_learning_rate(**kwargs)


def test_training_config_maps_independent_optimizer_settings_to_each_tower() -> None:
    config = TrainingConfig(
        lr=9e-3,
        weight_decay=9e-2,
        gnn_lr=1e-3,
        gnn_lr_schedule="cosine",
        gnn_min_lr_ratio=0.11,
        gnn_weight_decay=1e-4,
        seq_lr=2e-3,
        seq_lr_schedule="cosine",
        seq_min_lr_ratio=0.12,
        seq_weight_decay=2e-4,
        two_tower_lr=3e-3,
        two_tower_lr_schedule="cosine",
        two_tower_min_lr_ratio=0.13,
        two_tower_weight_decay=3e-4,
        source_profile_lr=4e-3,
        source_profile_lr_schedule="cosine",
        source_profile_min_lr_ratio=0.14,
        source_profile_weight_decay=4e-4,
    )

    graph = config.graph_config()
    sequence = config.sequence_config()
    two_tower = config.two_tower_config()
    source_profile = config.source_profile_config()

    assert graph.lr_schedule == "cosine"
    assert (graph.lr, graph.min_lr_ratio, graph.weight_decay) == pytest.approx(
        (1e-3, 0.11, 1e-4)
    )
    assert sequence.lr_schedule == "cosine"
    assert (
        sequence.lr,
        sequence.min_lr_ratio,
        sequence.weight_decay,
    ) == pytest.approx((2e-3, 0.12, 2e-4))
    assert two_tower.lr_schedule == "cosine"
    assert (
        two_tower.lr,
        two_tower.min_lr_ratio,
        two_tower.weight_decay,
    ) == pytest.approx((3e-3, 0.13, 3e-4))
    assert source_profile.lr_schedule == "cosine"
    assert (
        source_profile.lr,
        source_profile.min_lr_ratio,
        source_profile.weight_decay,
    ) == pytest.approx((4e-3, 0.14, 4e-4))


def test_legacy_training_config_without_tower_optimizer_fields_keeps_old_defaults() -> None:
    legacy = TrainingConfig(lr=2e-3, weight_decay=0.0, gnn_lr=3e-3)
    for field in (
        "gnn_lr_schedule",
        "gnn_min_lr_ratio",
        "gnn_weight_decay",
        "seq_lr",
        "seq_lr_schedule",
        "seq_min_lr_ratio",
        "seq_weight_decay",
        "two_tower_lr",
        "two_tower_lr_schedule",
        "two_tower_min_lr_ratio",
        "two_tower_weight_decay",
        "source_profile_lr",
        "source_profile_lr_schedule",
        "source_profile_min_lr_ratio",
        "source_profile_weight_decay",
    ):
        object.__delattr__(legacy, field)

    graph = legacy.graph_config()
    sequence = legacy.sequence_config()
    two_tower = legacy.two_tower_config()
    source_profile = legacy.source_profile_config()

    assert graph.lr_schedule == "constant"
    assert (graph.lr, graph.min_lr_ratio, graph.weight_decay) == pytest.approx(
        (3e-3, 0.0, 0.0)
    )
    for tower in (sequence, two_tower, source_profile):
        assert tower.lr_schedule == "constant"
        assert (
            tower.lr,
            tower.min_lr_ratio,
            tower.weight_decay,
        ) == pytest.approx((2e-3, 0.0, 0.0))


def test_tower_optimizer_learning_rate_is_updated_for_the_current_epoch() -> None:
    class FakeOptimizer:
        lr = 9.0

    optimizer = FakeOptimizer()

    applied = set_tower_optimizer_learning_rate(
        optimizer,
        initial_lr=1e-3,
        epoch=3,
        total_epochs=5,
        schedule="cosine",
        min_lr_ratio=0.1,
    )

    assert applied == pytest.approx(5.5e-4)
    assert optimizer.lr == pytest.approx(applied)


def test_training_config_maps_two_tower_in_batch_negative_settings() -> None:
    tower = TrainingConfig(
        two_tower_in_batch_negatives=True,
        two_tower_in_batch_negative_weight=0.75,
        two_tower_in_batch_temperature=0.2,
    ).two_tower_config()

    assert tower.in_batch_negatives is True
    assert tower.in_batch_negative_weight == pytest.approx(0.75)
    assert tower.in_batch_temperature == pytest.approx(0.2)
