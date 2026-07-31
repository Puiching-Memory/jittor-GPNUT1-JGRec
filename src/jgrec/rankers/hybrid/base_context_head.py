from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_STATE_PREFIX = "state__"


@dataclass(frozen=True)
class BaseContextHeadArtifact:
    context_transform_version: int
    hidden_dim: int
    feature_indices: tuple[int, ...]
    mean: np.ndarray
    std: np.ndarray
    state: dict[str, np.ndarray]
    best_val_ap: float
    best_val_mrr: float
    candidate_name: str
    fit_rows: tuple[int, int]
    tune_rows: tuple[int, int]


def save_base_context_head(
    path: Path,
    artifact: BaseContextHeadArtifact,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    validated = _validated_artifact(artifact)
    payload: dict[str, np.ndarray] = {
        "context_transform_version": np.asarray(
            [validated.context_transform_version],
            dtype=np.int32,
        ),
        "hidden_dim": np.asarray(
            [validated.hidden_dim],
            dtype=np.int32,
        ),
        "feature_indices": np.asarray(
            validated.feature_indices,
            dtype=np.int32,
        ),
        "mean": validated.mean,
        "std": validated.std,
        "best_val_ap": np.asarray(
            [validated.best_val_ap],
            dtype=np.float64,
        ),
        "best_val_mrr": np.asarray(
            [validated.best_val_mrr],
            dtype=np.float64,
        ),
        "candidate_name": np.asarray([validated.candidate_name]),
        "fit_rows": np.asarray(validated.fit_rows, dtype=np.int64),
        "tune_rows": np.asarray(validated.tune_rows, dtype=np.int64),
    }
    payload.update(
        {
            f"{_STATE_PREFIX}{key}": value
            for key, value in validated.state.items()
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)


def load_base_context_head(
    path: Path,
    *,
    expected_context_transform_version: int | None = None,
) -> BaseContextHeadArtifact:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        state = {
            key.removeprefix(_STATE_PREFIX): np.asarray(
                payload[key],
                dtype=np.float32,
            ).copy()
            for key in payload.files
            if key.startswith(_STATE_PREFIX)
        }
        artifact = BaseContextHeadArtifact(
            context_transform_version=int(
                payload["context_transform_version"][0]
            ),
            hidden_dim=int(payload["hidden_dim"][0]),
            feature_indices=tuple(
                int(value) for value in payload["feature_indices"]
            ),
            mean=np.asarray(payload["mean"], dtype=np.float32).copy(),
            std=np.asarray(payload["std"], dtype=np.float32).copy(),
            state=state,
            best_val_ap=float(payload["best_val_ap"][0]),
            best_val_mrr=float(payload["best_val_mrr"][0]),
            candidate_name=str(payload["candidate_name"][0]),
            fit_rows=tuple(int(value) for value in payload["fit_rows"]),
            tune_rows=tuple(int(value) for value in payload["tune_rows"]),
        )
    validated = _validated_artifact(artifact)
    if (
        expected_context_transform_version is not None
        and validated.context_transform_version
        != int(expected_context_transform_version)
    ):
        raise ValueError(
            "base context head has an unexpected context transform version"
        )
    return validated


def _validated_artifact(
    artifact: BaseContextHeadArtifact,
) -> BaseContextHeadArtifact:
    version = int(artifact.context_transform_version)
    if version not in {0, 1}:
        raise ValueError("base context head supports only v0 or v1")
    hidden_dim = int(artifact.hidden_dim)
    if hidden_dim <= 0:
        raise ValueError("base context head hidden_dim must be positive")
    indices = tuple(int(value) for value in artifact.feature_indices)
    if not indices or any(value < 0 for value in indices):
        raise ValueError("base context head feature indices are invalid")
    if len(set(indices)) != len(indices):
        raise ValueError("base context head feature indices must be unique")

    mean = np.asarray(artifact.mean, dtype=np.float32)
    std = np.asarray(artifact.std, dtype=np.float32)
    expected_width = len(indices) * (3 if version == 1 else 1)
    if mean.shape != (expected_width,) or std.shape != (expected_width,):
        channel_label = (
            "three context channels" if version == 1 else "one raw channel"
        )
        raise ValueError(
            f"base context head normalizer must contain {channel_label}"
        )
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
        or np.any(std <= 0.0)
    ):
        raise ValueError("base context head normalizer is invalid")

    state = {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in artifact.state.items()
    }
    first_layer = state.get("linear1.weight")
    if (
        first_layer is None
        or first_layer.shape != (hidden_dim, expected_width)
    ):
        raise ValueError(
            "base context head first layer differs from its input contract"
        )
    if not all(np.all(np.isfinite(value)) for value in state.values()):
        raise ValueError("base context head state must be finite")

    fit_rows = tuple(int(value) for value in artifact.fit_rows)
    tune_rows = tuple(int(value) for value in artifact.tune_rows)
    if (
        len(fit_rows) != 2
        or len(tune_rows) != 2
        or not fit_rows[0] < fit_rows[1] == tune_rows[0] < tune_rows[1]
    ):
        raise ValueError("base context head fit/tune rows are not causal")
    candidate_name = str(artifact.candidate_name).strip()
    if not candidate_name:
        raise ValueError("base context head candidate name is required")
    best_values = (
        float(artifact.best_val_ap),
        float(artifact.best_val_mrr),
    )
    if not all(np.isfinite(value) for value in best_values):
        raise ValueError("base context head validation metrics must be finite")
    return BaseContextHeadArtifact(
        context_transform_version=version,
        hidden_dim=hidden_dim,
        feature_indices=indices,
        mean=mean.copy(),
        std=std.copy(),
        state={key: value.copy() for key, value in state.items()},
        best_val_ap=best_values[0],
        best_val_mrr=best_values[1],
        candidate_name=candidate_name,
        fit_rows=(fit_rows[0], fit_rows[1]),
        tune_rows=(tune_rows[0], tune_rows[1]),
    )
