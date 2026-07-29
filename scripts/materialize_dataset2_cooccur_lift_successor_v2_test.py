from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.cooccur_lift_successor_external import (
    authorize_successor_package,
    short_window_support_from_availability,
    validate_successor_external_setup,
)
from jgrec.core.io import read_interactions, read_test_queries
from jgrec.rankers.hybrid.cooccur_lift_native import (
    materialize_compact_cooccur_lift,
)
from jgrec.rankers.hybrid.cooccur_lift_successor import (
    CooccurLiftGapAwareView,
)
from jgrec.rankers.hybrid.fusion import predict_logits
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView
from jgrec.rankers.registry import create_ranker
from materialize_dataset2_cooccur_lift_test import (
    _load_auxiliary_model,
    _softmax,
)

EXPECTED_ROWS = 153_420
CANDIDATE_COUNT = 100


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize accepted gap-aware v2 test probabilities with "
            "support derived from deployed feature availability."
        )
    )
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--external-report", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--auxiliary-model", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    setup = validate_successor_external_setup(
        candidate_config_path=args.candidate_config,
        selection_lock_path=args.selection_lock,
    )
    external = _read_json(args.external_report)
    authorization = authorize_successor_package(
        external_report=external,
        external_report_sha256=_sha256(args.external_report),
        expected_selection_lock_sha256=setup.selection_lock_sha256,
        expected_candidate_id=setup.candidate_id,
        expected_config_sha256=setup.config_sha256,
    )

    interactions = read_interactions(args.train_csv).sort_by_time()
    queries = read_test_queries(args.test_csv)
    if (
        len(queries) != EXPECTED_ROWS
        or queries.candidate_count != CANDIDATE_COUNT
        or not np.all(queries.time[1:] >= queries.time[:-1])
    ):
        raise ValueError("Dataset2 test query contract differs")
    history_end = int(interactions.time[-1])
    availability = np.minimum(
        np.asarray(queries.time, dtype=np.int64),
        history_end,
    )
    support = short_window_support_from_availability(
        queries.time,
        availability,
        short_window_seconds=setup.short_window_seconds,
    )
    collapsed_rows = int(np.count_nonzero(support == 0.0))
    if collapsed_rows != 61_109:
        raise ValueError("deployed short-window collapse count differs")

    args.output_dir.mkdir(parents=True)
    started = time.time()
    lift_path = args.output_dir / "test-lift-features.npy"
    native = materialize_compact_cooccur_lift(
        interactions=interactions,
        sources=queries.src,
        candidates=queries.candidates,
        destinations=queries.candidates[:, 0],
        event_time=queries.time,
        availability_time=availability,
        short_window=float(setup.short_window_seconds),
        lift_path=lift_path,
        positive_popularity_path=(
            args.output_dir / "test-positive-popularity.npy"
        ),
        progress_path=args.output_dir / "test-materialization-progress.json",
        work_dir=args.output_dir / "native",
    )
    if native["collapsed_short_rows"] != collapsed_rows:
        raise ValueError("native and support-indicator collapse counts differ")
    lift = np.load(lift_path, mmap_mode="r", allow_pickle=False)

    checkpoint = load_checkpoint_dataset(args.source_checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in checkpoint["feature_names"])
    gnn_column = feature_names.index("gnn_short")
    ranker = create_ranker("hybrid", None)
    ranker.hydrate(checkpoint)
    del checkpoint
    model, result = _load_auxiliary_model(args.auxiliary_model)
    if len(result.mean) != 198:
        raise ValueError("gap-aware v2 model context schema differs")
    jt.flags.use_cuda = 1

    probabilities_path = args.output_dir / "test-auxiliary-probabilities.npy"
    probabilities = np.lib.format.open_memmap(
        probabilities_path,
        mode="w+",
        dtype=np.float64,
        shape=queries.candidates.shape,
    )
    score_order = np.argsort(queries.src, kind="stable")
    for start in range(0, len(queries), args.batch_size):
        stop = min(start + args.batch_size, len(queries))
        rows = score_order[start:stop]
        batch_queries = queries[rows]
        base = ranker.impl.encoder.features_for_query_array(batch_queries)
        view = SetwiseFeatureView(
            CooccurLiftGapAwareView(
                base,
                short_none_scores=base[..., gnn_column],
                gnn_short_column=gnn_column,
                lift_features=lift[rows],
                short_window_supported=support[rows],
            ),
            transform_version=1,
        )
        selected = np.asarray(view[:], dtype=np.float32)
        if result.feature_indices != tuple(range(selected.shape[-1])):
            selected = selected[..., result.feature_indices]
        logits = predict_logits(
            model,
            selected,
            result.mean,
            result.std,
        )
        probabilities[rows] = _softmax(logits)
        if stop % 10_000 < args.batch_size or stop == len(queries):
            print(
                json.dumps(
                    {
                        "status": "scoring_test",
                        "completed_rows": stop,
                        "total_rows": len(queries),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del base, view, selected, logits
        if stop % 16_384 < args.batch_size:
            gc.collect()
            jt.sync_all()
            jt.clean()
    probabilities.flush()
    del probabilities

    persisted = np.load(
        probabilities_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if not np.all(np.isfinite(persisted)):
        raise ValueError("test auxiliary probabilities are non-finite")
    row_sum_error = float(
        np.max(np.abs(np.asarray(persisted).sum(axis=1) - 1.0))
    )
    if row_sum_error > 5e-6:
        raise ValueError("test auxiliary probabilities are not normalized")
    report = {
        "schema_version": 1,
        "protocol": "cooccur_lift_successor_v2_test_materialization_v1",
        "status": "complete_online_candidate_materialization",
        "candidate_id": setup.candidate_id,
        "candidate_config_sha256": setup.config_sha256,
        "selection_lock_sha256": setup.selection_lock_sha256,
        "selected_weight": setup.selected_weight,
        "external_authorization": authorization,
        "external_report": str(args.external_report.resolve()),
        "external_report_sha256": _sha256(args.external_report),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "auxiliary_model": str(args.auxiliary_model.resolve()),
        "auxiliary_model_sha256": _sha256(args.auxiliary_model),
        "test_csv": str(args.test_csv.resolve()),
        "test_csv_sha256": _sha256(args.test_csv),
        "test_candidate_fingerprint": hashlib.sha256(
            np.ascontiguousarray(queries.candidates).tobytes(order="C")
        ).hexdigest(),
        "probabilities": str(probabilities_path.resolve()),
        "probabilities_sha256": _sha256(probabilities_path),
        "shape": list(persisted.shape),
        "maximum_row_sum_error": row_sum_error,
        "short_window_support": {
            "supported_rows": int(len(support) - collapsed_rows),
            "collapsed_rows": collapsed_rows,
            "total_rows": len(support),
            "collapsed_fraction": collapsed_rows / len(support),
            "unique_values": [0, 1],
            "availability_rule": "min(query_time, train_history_end)",
            "boundary": "strictly_less_than_short_window_seconds",
        },
        "native_materializer": native,
        "feature_scoring_query_order": "source_grouped",
        "feature_scoring_output_order": "original_test_csv_row_order",
        "scoring_device": "cuda",
        "production_checkpoint_modified": False,
        "external_effect_size_used": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "test-materialization-report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
