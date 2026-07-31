from __future__ import annotations

import argparse
import builtins
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.cuda import require_jittor_cuda
from jgrec.core.io import discover_datasets, read_test_queries
from jgrec.rankers.hybrid.candidate_set_transformer import (
    CandidateSetEnsembleCheckpoint,
    load_candidate_set_checkpoint,
    predict_candidate_set_logits,
)
from jgrec.rankers.hybrid.oof_models import (
    PureJittorOOFStackingCheckpoint,
    load_candidate_set_mlp_checkpoint,
    predict_candidate_set_mlp_logits,
    snapshot_pure_jittor_oof_stacking,
)
from jgrec.rankers.hybrid.oof_stacking import (
    STABLE_EXPERT_LOGIT_FEATURE_VERSION,
    stable_expert_logit_features,
)
from jgrec.rankers.registry import create_ranker

EXPERT_NAMES = ("cst_main", "cst_residual", "setwise_mlp")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate pure-Jittor full-expert Dataset2 test logits for a "
            "rejected OOF experiment without creating a submission package."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--base-experiment-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--stable-experiment-dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import jittor as jt  # noqa: PLC0415

    require_jittor_cuda(jt)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    evaluation_path = args.stable_experiment_dir / "evaluation-report.json"
    evaluation = _read_json(evaluation_path)
    if evaluation.get("status") != "rejected":
        raise ValueError("diagnostic test-logit generation requires a rejected gate")
    if int(evaluation["stable_feature_version"]) != STABLE_EXPERT_LOGIT_FEATURE_VERSION:
        raise ValueError("stable feature version differs from evaluation")

    cst_pairs = tuple(
        load_candidate_set_checkpoint(args.base_experiment_dir / "full-experts" / f"{name}.npz")
        for name in EXPERT_NAMES[:2]
    )
    setwise_pair = load_candidate_set_mlp_checkpoint(args.base_experiment_dir / "full-experts" / "setwise_mlp.npz")
    meta_pair = load_candidate_set_mlp_checkpoint(args.stable_experiment_dir / "meta-stacking-mlp.npz")
    stacking = PureJittorOOFStackingCheckpoint(
        expert_names=EXPERT_NAMES,
        cst_experts=CandidateSetEnsembleCheckpoint(
            models=tuple(pair[0] for pair in cst_pairs),
            results=tuple(pair[1] for pair in cst_pairs),
            weights=(0.5, 0.5),
        ),
        setwise_mlp=setwise_pair,
        meta_mlp=meta_pair,
        meta_weight=float(evaluation["meta_weight"]),
    )
    stacking_state = snapshot_pure_jittor_oof_stacking(stacking)

    dataset2_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset2",
    )
    dataset2_state.update(
        {
            "fusion_state": None,
            "fusion_result": None,
            "lgbm_result": None,
            "segment_gate_result": None,
            "setwise_fusion_state": None,
            "setwise_fusion_result": None,
            "time_ramp_setwise_fusion_state": None,
            "time_ramp_setwise_result": None,
            "time_ramp_config": None,
            "conservative_window_fusion_states": {},
            "conservative_window_results": {},
            "conservative_window_hidden_dims": {},
            "conservative_window_config": None,
            "multi_interest_proxy_state": None,
            "candidate_set_ensemble_state": None,
            "oof_stacking_state": stacking_state,
        }
    )
    ranker = create_ranker("hybrid", None)
    original_import = builtins.__import__

    def block_external_ml(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {"lightgbm", "sklearn"}:
            raise RuntimeError(f"forbidden Dataset2 inference import: {name}")
        if name in {"fusion", "fusion_lgbm"}:
            raise RuntimeError(f"legacy Dataset2 fusion import during hydrate: {name}")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = block_external_ml
    try:
        ranker.hydrate(dataset2_state)
        datasets = {dataset.name: dataset for dataset in discover_datasets(args.data_dir)}
        queries = read_test_queries(datasets["dataset2"].test_path)
        prediction_order = ranker.prediction_order(queries)
        if prediction_order is None:
            prediction_order = np.arange(len(queries), dtype=np.int64)
        else:
            prediction_order = np.asarray(prediction_order, dtype=np.int64)
        raw_path = args.output_dir / "dataset2-test-expert-logits.npy"
        selected_path = args.output_dir / "dataset2-test-stacking-scores.npy"
        raw = np.lib.format.open_memmap(
            raw_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(EXPERT_NAMES), len(queries), queries.candidate_count),
        )
        selected = np.lib.format.open_memmap(
            selected_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(queries), queries.candidate_count),
        )
        for start in range(0, len(queries), args.batch_size):
            stop = min(start + args.batch_size, len(queries))
            row_indices = prediction_order[start:stop]
            batch_features = ranker.impl.encoder.features_for_queries(queries[row_indices])
            batch_logits = [
                predict_candidate_set_logits(
                    model,
                    batch_features,
                    mean=result.mean,
                    std=result.std,
                    batch_size=args.batch_size,
                )
                for model, result in zip(
                    stacking.cst_experts.models,
                    stacking.cst_experts.results,
                    strict=True,
                )
            ]
            batch_logits.append(
                predict_candidate_set_mlp_logits(
                    stacking.setwise_mlp[0],
                    batch_features,
                    mean=stacking.setwise_mlp[1].mean,
                    std=stacking.setwise_mlp[1].std,
                    batch_size=args.batch_size,
                )
            )
            stacked_logits = np.stack(batch_logits, axis=0)
            raw[:, row_indices, :] = stacked_logits
            stable = stable_expert_logit_features(stacked_logits)
            meta_logits = predict_candidate_set_mlp_logits(
                stacking.meta_mlp[0],
                stable,
                mean=stacking.meta_mlp[1].mean,
                std=stacking.meta_mlp[1].std,
                batch_size=args.batch_size,
            )
            meta_percentile = stable_expert_logit_features(meta_logits[None, ...])[..., 0]
            selected[row_indices] = (
                stacking.meta_weight * meta_percentile + (1.0 - stacking.meta_weight) * stable[..., -5]
            )
            if stop % 5_000 == 0 or stop == len(queries):
                raw.flush()
                selected.flush()
                print(
                    f"[oof-test-logits] rows={stop}/{len(queries)}",
                    flush=True,
                )
        raw.flush()
        selected.flush()
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(selected)):
            raise RuntimeError("generated OOF test logits are non-finite")
        del raw, selected, queries
        gc.collect()
    finally:
        builtins.__import__ = original_import

    report = {
        "status": "complete_rejected_candidate_diagnostics_only",
        "submission_generated": False,
        "rejection_report": str(evaluation_path.resolve()),
        "rejection_report_sha256": _sha256(evaluation_path),
        "stable_feature_version": STABLE_EXPERT_LOGIT_FEATURE_VERSION,
        "expert_names": list(EXPERT_NAMES),
        "meta_weight": stacking.meta_weight,
        "test_expert_logits": str(raw_path.resolve()),
        "test_expert_logits_sha256": _sha256(raw_path),
        "test_stacking_scores": str(selected_path.resolve()),
        "test_stacking_scores_sha256": _sha256(selected_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "blocked_imports": ["lightgbm", "sklearn", "legacy fusion"],
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(args.output_dir / "test-logits-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
