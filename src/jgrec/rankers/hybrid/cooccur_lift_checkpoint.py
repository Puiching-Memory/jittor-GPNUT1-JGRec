from __future__ import annotations

import hashlib
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from jgrec.core.types import TestQueryArray
from jgrec.rankers.hybrid.cooccur_lift import (
    BASE_FEATURE_COUNT,
    CONTEXT_FEATURE_COUNT,
    INTEGRATION_ID,
)

FINGERPRINT_BYTES = 32
LIFT_FEATURE_COUNT = 2
CHECKPOINT_FIELD = "cooccur_lift_auxiliary_state"
LOCKED_WEIGHT = 0.5
ONLINE_PROMOTION_STATUS = "online_score_passed_before_checkpoint_wiring"


def validate_online_promotion_receipt(
    receipt: Mapping[str, object],
) -> None:
    """Reject promotion authority that differs from the frozen contract."""
    if int(receipt.get("schema_version", 0)) != 1:
        raise ValueError("unsupported online-promotion receipt schema")
    if receipt.get("status") != ONLINE_PROMOTION_STATUS:
        raise ValueError("online-promotion receipt status differs")
    if receipt.get("integration_id") != INTEGRATION_ID:
        raise ValueError("online-promotion integration_id differs")
    score = float(receipt.get("online_score", float("-inf")))
    threshold = float(
        receipt.get("promotion_threshold", float("inf"))
    )
    if (
        not np.isfinite(score)
        or not np.isfinite(threshold)
        or score <= threshold
        or receipt.get("threshold_comparison") != "strictly_greater"
    ):
        raise ValueError("online score does not clear the frozen threshold")
    delta = float(
        receipt.get("delta_online_minus_threshold", float("nan"))
    )
    if not np.isfinite(delta) or abs(delta - (score - threshold)) > 1e-15:
        raise ValueError("online-promotion score delta differs")
    if float(receipt.get("selected_weight", -1.0)) != LOCKED_WEIGHT:
        raise ValueError("online-promotion weight differs")
    required_authority = {
        "checkpoint_wiring_authorized": True,
        "double_replay_required": True,
        "weight_rescan_authorized": False,
        "formula_change_authorized": False,
        "model_retraining_authorized": False,
    }
    for key, expected in required_authority.items():
        if receipt.get(key) is not expected:
            raise ValueError(f"online-promotion authority differs: {key}")
    for key in (
        "candidate_zip_sha256",
        "candidate_report_sha256",
        "selection_lock_sha256",
        "external_report_sha256",
        "source_checkpoint_sha256",
        "auxiliary_model_sha256",
    ):
        value = receipt.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"online-promotion receipt has invalid {key}")


def fingerprint_queries(queries: TestQueryArray) -> np.ndarray:
    """Bind each query's source, time, and ordered candidate IDs."""
    if not isinstance(queries, TestQueryArray):
        raise TypeError("queries must be a TestQueryArray")
    output = np.empty(
        (len(queries), FINGERPRINT_BYTES),
        dtype=np.uint8,
    )
    for row in range(len(queries)):
        digest = hashlib.sha256()
        digest.update(
            struct.pack(
                "<qqq",
                int(queries.src[row]),
                int(queries.time[row]),
                int(queries.candidate_count),
            )
        )
        candidates = np.asarray(
            queries.candidates[row],
            dtype="<i8",
        )
        digest.update(candidates.tobytes(order="C"))
        output[row] = np.frombuffer(digest.digest(), dtype=np.uint8)
    return output


@dataclass
class CausalLiftFeatureStore:
    query_fingerprints: np.ndarray
    lift_features: np.ndarray
    _row_by_fingerprint: dict[bytes, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.query_fingerprints = np.asarray(
            self.query_fingerprints,
            dtype=np.uint8,
        )
        self.lift_features = np.asarray(
            self.lift_features,
            dtype=np.float32,
        )
        if (
            self.query_fingerprints.ndim != 2
            or self.query_fingerprints.shape[1] != FINGERPRINT_BYTES
        ):
            raise ValueError(
                "query fingerprints must have shape (rows, 32)"
            )
        if (
            self.lift_features.ndim != 3
            or self.lift_features.shape[0]
            != self.query_fingerprints.shape[0]
            or self.lift_features.shape[2] != LIFT_FEATURE_COUNT
        ):
            raise ValueError(
                "lift feature shape must be (rows, candidates, 2)"
            )
        if not np.all(np.isfinite(self.lift_features)):
            raise ValueError("lift features contain non-finite values")
        self._build_lookup()

    @classmethod
    def from_queries(
        cls,
        queries: TestQueryArray,
        lift_features: np.ndarray,
    ) -> CausalLiftFeatureStore:
        lift = np.asarray(lift_features)
        expected = (
            len(queries),
            queries.candidate_count,
            LIFT_FEATURE_COUNT,
        )
        if lift.shape != expected:
            raise ValueError(
                f"lift feature shape is {lift.shape}, expected {expected}"
            )
        return cls(
            query_fingerprints=fingerprint_queries(queries),
            lift_features=np.asarray(lift, dtype=np.float32),
        )

    def lookup(self, queries: TestQueryArray) -> np.ndarray:
        if queries.candidate_count != self.lift_features.shape[1]:
            raise ValueError(
                "query candidate count differs from lift feature store"
            )
        fingerprints = fingerprint_queries(queries)
        rows = np.empty(len(queries), dtype=np.intp)
        for index, fingerprint in enumerate(fingerprints):
            key = fingerprint.tobytes()
            try:
                rows[index] = self._row_by_fingerprint[key]
            except KeyError as error:
                raise KeyError(
                    "query fingerprint is not present in lift feature store"
                ) from error
        return self.lift_features[rows]

    def __getstate__(self) -> dict[str, np.ndarray]:
        return {
            "query_fingerprints": self.query_fingerprints,
            "lift_features": self.lift_features,
        }

    def __setstate__(self, state: dict[str, np.ndarray]) -> None:
        self.query_fingerprints = state["query_fingerprints"]
        self.lift_features = state["lift_features"]
        self.__post_init__()

    def _build_lookup(self) -> None:
        lookup: dict[bytes, int] = {}
        for row, fingerprint in enumerate(self.query_fingerprints):
            key = fingerprint.tobytes()
            existing = lookup.get(key)
            if existing is not None:
                if not np.array_equal(
                    self.lift_features[existing],
                    self.lift_features[row],
                ):
                    raise ValueError(
                        "query fingerprint collision has conflicting lift "
                        "features"
                    )
                continue
            lookup[key] = row
        self._row_by_fingerprint = lookup


@dataclass(frozen=True)
class CooccurLiftAuxiliaryState:
    integration_id: str
    weight: float
    hidden_dim: int
    gnn_short_column: int
    model_state: dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    feature_indices: tuple[int, ...]
    feature_store: CausalLiftFeatureStore
    provenance: dict[str, str]

    def __post_init__(self) -> None:
        if self.integration_id != INTEGRATION_ID:
            raise ValueError("cooccur-lift integration_id differs")
        if float(self.weight) != LOCKED_WEIGHT:
            raise ValueError("cooccur-lift auxiliary weight must be 0.50")
        if int(self.hidden_dim) <= 0:
            raise ValueError("cooccur-lift hidden_dim must be positive")
        if not 0 <= int(self.gnn_short_column) < BASE_FEATURE_COUNT:
            raise ValueError("cooccur-lift gnn_short column is invalid")
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        expected_indices = tuple(range(CONTEXT_FEATURE_COUNT))
        if (
            mean.shape != (CONTEXT_FEATURE_COUNT,)
            or std.shape != (CONTEXT_FEATURE_COUNT,)
            or self.feature_indices != expected_indices
        ):
            raise ValueError(
                "cooccur-lift auxiliary requires the complete 195-column "
                "context"
            )
        if (
            not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(std))
            or np.any(std <= 0.0)
        ):
            raise ValueError(
                "cooccur-lift normalizer must be finite with positive std"
            )
        if not self.model_state:
            raise ValueError("cooccur-lift model state must not be empty")
        for value in self.model_state.values():
            array = np.asarray(value)
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    "cooccur-lift model state contains non-finite values"
                )
        if not isinstance(self.feature_store, CausalLiftFeatureStore):
            raise TypeError(
                "feature_store must be a CausalLiftFeatureStore"
            )
        if not self.provenance.get("selection_lock_sha256"):
            raise ValueError(
                "cooccur-lift provenance must bind the selection lock"
            )


def install_cooccur_lift_auxiliary_state(
    source: dict[str, object],
    state: CooccurLiftAuxiliaryState,
) -> dict[str, object]:
    if source.get(CHECKPOINT_FIELD) is not None:
        raise ValueError("checkpoint already has a cooccur-lift auxiliary")
    feature_names = tuple(str(name) for name in source["feature_names"])
    if len(feature_names) != BASE_FEATURE_COUNT:
        raise ValueError(
            "cooccur-lift source checkpoint must have 63 base features"
        )
    if feature_names[state.gnn_short_column] != "gnn_short":
        raise ValueError(
            "cooccur-lift gnn_short column differs from source checkpoint"
        )
    candidate = dict(source)
    candidate[CHECKPOINT_FIELD] = state
    return candidate


def build_cooccur_lift_auxiliary_model(
    state: CooccurLiftAuxiliaryState,
):
    from jgrec.rankers.hybrid.fusion import (  # noqa: PLC0415
        build_fusion_from_state,
    )

    return build_fusion_from_state(
        input_dim=CONTEXT_FEATURE_COUNT,
        hidden_dim=state.hidden_dim,
        state=state.model_state,
    )


def predict_cooccur_lift_auxiliary_probabilities(
    state: CooccurLiftAuxiliaryState,
    model,
    base_features: np.ndarray,
    queries: TestQueryArray,
) -> np.ndarray:
    from jgrec.rankers.hybrid.cooccur_lift import (  # noqa: PLC0415
        CooccurLiftAugmentedView,
    )
    from jgrec.rankers.hybrid.fusion import (  # noqa: PLC0415
        predict_logits,
    )
    from jgrec.rankers.hybrid.setwise import (  # noqa: PLC0415
        SetwiseFeatureView,
    )

    base = np.asarray(base_features, dtype=np.float32)
    if base.shape != (
        len(queries),
        queries.candidate_count,
        BASE_FEATURE_COUNT,
    ):
        raise ValueError(
            "base features differ from the cooccur-lift 63-column contract"
        )
    lift = state.feature_store.lookup(queries)
    augmented = CooccurLiftAugmentedView(
        base,
        short_none_scores=base[..., state.gnn_short_column],
        gnn_short_column=state.gnn_short_column,
        lift_features=lift,
    )
    context = np.asarray(
        SetwiseFeatureView(augmented, transform_version=1)[:],
        dtype=np.float32,
    )
    _synchronize_and_clean_jittor()
    try:
        logits = predict_logits(
            model,
            context,
            state.mean,
            state.std,
        )
        values = np.asarray(logits, dtype=np.float64)
    finally:
        _synchronize_and_clean_jittor()
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _synchronize_and_clean_jittor() -> None:
    jt = sys.modules.get("jittor")
    if jt is None:
        return
    sync_all = getattr(jt, "sync_all", None)
    if callable(sync_all):
        sync_all()
    clean = getattr(jt, "clean", None)
    if callable(clean):
        clean()
