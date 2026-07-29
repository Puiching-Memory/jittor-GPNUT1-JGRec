from __future__ import annotations

from pathlib import Path

import pytest

from jgrec.contest_checkpoint import (
    ContestCheckpointWriter,
    compose_checkpoint_datasets,
    load_checkpoint_dataset,
    load_checkpoint_metadata,
)


def _raise_if_unpickled():
    raise AssertionError("unrelated dataset state was unpickled")


class _ExplodesWhenUnpickled:
    def __reduce__(self):
        return _raise_if_unpickled, ()


def test_writer_publishes_complete_checkpoint_and_loads_each_dataset(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "checkpoint1.pkl"
    writer = ContestCheckpointWriter(
        checkpoint_path,
        model_name="hybrid",
        expected_datasets=("dataset1", "dataset2"),
        metadata={"run_name": "champion"},
    )

    writer.add_dataset("dataset1", {"weight": 1})
    assert not checkpoint_path.exists()

    writer.add_dataset("dataset2", {"weight": 2})
    writer.finalize()

    assert checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".pkl.tmp").exists()
    assert load_checkpoint_metadata(checkpoint_path) == {
        "format": "jgrec-contest-checkpoint",
        "version": 1,
        "model_name": "hybrid",
        "datasets": ("dataset1", "dataset2"),
        "run_name": "champion",
    }
    assert load_checkpoint_dataset(checkpoint_path, "dataset1") == {"weight": 1}
    assert load_checkpoint_dataset(checkpoint_path, "dataset2") == {"weight": 2}


def test_writer_refuses_to_publish_when_a_dataset_is_missing(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint1.pkl"
    writer = ContestCheckpointWriter(
        checkpoint_path,
        model_name="hybrid",
        expected_datasets=("dataset1", "dataset2"),
    )
    writer.add_dataset("dataset1", {"weight": 1})

    with pytest.raises(RuntimeError, match="missing datasets: dataset2"):
        writer.finalize()

    assert not checkpoint_path.exists()


def test_checkpoint_loader_rejects_unknown_dataset(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint1.pkl"
    writer = ContestCheckpointWriter(
        checkpoint_path,
        model_name="hybrid",
        expected_datasets=("dataset1",),
    )
    writer.add_dataset("dataset1", {"weight": 1})
    writer.finalize()

    with pytest.raises(KeyError, match="dataset9"):
        load_checkpoint_dataset(checkpoint_path, "dataset9")


def test_writer_resumes_partial_checkpoint_across_training_runs(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint1.pkl"
    first_run = ContestCheckpointWriter(
        checkpoint_path,
        model_name="hybrid",
        expected_datasets=("dataset1", "dataset2"),
    )
    first_run.add_dataset("dataset2", {"weight": 2})
    first_run.close_partial()

    assert not checkpoint_path.exists()
    assert checkpoint_path.with_suffix(".pkl.tmp").exists()

    second_run = ContestCheckpointWriter(
        checkpoint_path,
        model_name="hybrid",
        expected_datasets=("dataset1", "dataset2"),
    )
    assert second_run.written_datasets == ("dataset2",)
    second_run.add_dataset("dataset1", {"weight": 1})
    second_run.finalize()

    assert load_checkpoint_dataset(checkpoint_path, "dataset1") == {"weight": 1}
    assert load_checkpoint_dataset(checkpoint_path, "dataset2") == {"weight": 2}


def test_loader_does_not_materialize_unrelated_dataset_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint1.pkl"
    writer = ContestCheckpointWriter(
        checkpoint_path,
        model_name="hybrid",
        expected_datasets=("dataset1", "dataset2"),
    )
    writer.add_dataset("dataset1", {"weight": 1})
    writer.add_dataset("dataset2", _ExplodesWhenUnpickled())
    writer.finalize()

    assert load_checkpoint_dataset(checkpoint_path, "dataset1") == {"weight": 1}


def test_writer_discards_interrupted_tail_before_resuming(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint1.pkl"
    first_run = ContestCheckpointWriter(
        checkpoint_path,
        model_name="hybrid",
        expected_datasets=("dataset1", "dataset2"),
    )
    first_run.add_dataset("dataset1", {"weight": 1})
    first_run.close_partial()
    with checkpoint_path.with_suffix(".pkl.tmp").open("ab") as partial:
        partial.write(b"\x80")

    second_run = ContestCheckpointWriter(
        checkpoint_path,
        model_name="hybrid",
        expected_datasets=("dataset1", "dataset2"),
    )
    second_run.add_dataset("dataset2", {"weight": 2})
    second_run.finalize()

    assert load_checkpoint_dataset(checkpoint_path, "dataset2") == {"weight": 2}


def test_compose_checkpoint_keeps_champion_dataset1_and_replaces_dataset2_from_partial(
    tmp_path: Path,
) -> None:
    champion_path = tmp_path / "champion.pkl"
    champion = ContestCheckpointWriter(
        champion_path,
        model_name="hybrid",
        expected_datasets=("dataset1", "dataset2"),
        metadata={"run_name": "champion"},
    )
    champion.add_dataset("dataset1", {"weight": 11, "marker": "unchanged"})
    champion.add_dataset("dataset2", {"weight": 22, "marker": "old"})
    champion.finalize()

    candidate_path = tmp_path / "candidate.pkl"
    candidate = ContestCheckpointWriter(
        candidate_path,
        model_name="hybrid",
        expected_datasets=("dataset1", "dataset2"),
        metadata={"run_name": "candidate"},
    )
    candidate.add_dataset("dataset2", {"weight": 33, "marker": "new"})
    candidate.close_partial()

    output_path = tmp_path / "composed.pkl"
    compose_checkpoint_datasets(
        output_path,
        base_checkpoint=champion_path,
        replacements={
            "dataset2": candidate_path.with_suffix(".pkl.tmp"),
        },
    )

    assert load_checkpoint_dataset(output_path, "dataset1") == {
        "weight": 11,
        "marker": "unchanged",
    }
    assert load_checkpoint_dataset(output_path, "dataset2") == {
        "weight": 33,
        "marker": "new",
    }
    metadata = load_checkpoint_metadata(output_path)
    assert metadata["run_name"] == "champion"
    assert metadata["composed_replacements"] == ("dataset2",)
