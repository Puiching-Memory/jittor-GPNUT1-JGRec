#!/usr/bin/env python3
"""Ablation script for baseline-inspired improvements.

Runs paired ablations on dataset1 and dataset2 to validate:
- SwiGLU scorer
- Gated cross-attention
- RoPE time encoding

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/ablation_baseline_inspired.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jittor as jt

from jgrec.core.io import read_interactions
from jgrec.core.types import DatasetPaths, FitContext
from jgrec.logging import log
from jgrec.rankers.temporal_graph.config import TemporalGraphTrainingConfig
from jgrec.rankers.temporal_graph.model import (
    EndToEndTemporalGraphModel,
    TemporalGraphModelConfig,
)
from jgrec.rankers.temporal_graph.ranker import TemporalGraphRanker
from jgrec.rankers.temporal_graph.trainer import (
    CANDIDATE_PRIOR_FEATURE_DIM,
)

# ---------------------------------------------------------------------------
# Ablation grid
# ---------------------------------------------------------------------------
ABLATIONS: list[tuple[str, dict[str, Any]]] = [
    ("baseline", {"use_swiglu": False, "use_gated_attn": False, "use_rope": False}),
    ("swiglu", {"use_swiglu": True, "use_gated_attn": False, "use_rope": False}),
    ("gated_attn", {"use_swiglu": False, "use_gated_attn": True, "use_rope": False}),
    ("rope", {"use_swiglu": False, "use_gated_attn": False, "use_rope": True}),
    ("all", {"use_swiglu": True, "use_gated_attn": True, "use_rope": True}),
]

# ---------------------------------------------------------------------------
# Fixed proxy protocol
# ---------------------------------------------------------------------------
SEED = 60
MAX_FIT_EVENTS = 10_000
MAX_TRAIN_EVENTS = 1_000
MAX_VAL_EVENTS = 200
NUM_NEGATIVES = 99
EPOCHS = 2
HIDDEN_SIZE = 32
LAYERS = 1
HEADS = 2
HISTORY_LEN = 16
CANDIDATE_HISTORY_LEN = 4
TRAIN_BATCH_SIZE = 128
LR = 0.001
WEIGHT_DECAY = 0.0
DROPOUT = 0.15
SELECTION_METRIC = "ap"
EARLY_STOP_PATIENCE = 10

DATA_DIR = Path(__file__).parent.parent / "data"
RESULT_DIR = Path(__file__).parent.parent / "result" / "ablation" / "baseline_inspired"

# ---------------------------------------------------------------------------
# Monkeypatch _build_model so we can inject ablation flags
# ---------------------------------------------------------------------------
_original_build_model = TemporalGraphRanker._build_model


def _patched_build_model(self, time_span: int) -> EndToEndTemporalGraphModel:
    if self.node_map is None or self.config is None:
        raise RuntimeError("ranker is not initialized")
    return EndToEndTemporalGraphModel(
        TemporalGraphModelConfig(
            num_nodes=self.node_map.num_nodes,
            history_len=self.config.history_len,
            candidate_history_len=self.config.candidate_history_len,
            hidden_size=self.config.hidden_size,
            layers=self.config.layers,
            heads=self.config.heads,
            dropout=self.config.dropout,
            time_span=time_span,
            candidate_feature_dim=CANDIDATE_PRIOR_FEATURE_DIM,
            use_swiglu=getattr(self, "_ablation_use_swiglu", False),
            use_gated_attn=getattr(self, "_ablation_use_gated_attn", False),
            use_rope=getattr(self, "_ablation_use_rope", False),
        )
    )


TemporalGraphRanker._build_model = _patched_build_model

# ---------------------------------------------------------------------------


def _make_dataset(name: str) -> DatasetPaths:
    return DatasetPaths(
        name=name,
        root=DATA_DIR / name,
        train_path=DATA_DIR / name / "train.csv",
        test_path=DATA_DIR / name / "test.csv",
    )


def _run_single(
    dataset: DatasetPaths,
    variant_name: str,
    flags: dict[str, Any],
) -> dict[str, Any]:
    log(f"[{dataset.name}] {variant_name} start", enabled=True)
    start = perf_counter()

    jt.flags.use_cuda = 1
    jt.set_global_seed(SEED)

    ranker = TemporalGraphRanker()
    ranker._ablation_use_swiglu = flags["use_swiglu"]  # type: ignore[attr-defined]
    ranker._ablation_use_gated_attn = flags["use_gated_attn"]  # type: ignore[attr-defined]
    ranker._ablation_use_rope = flags["use_rope"]  # type: ignore[attr-defined]

    config = TemporalGraphTrainingConfig(
        seed=SEED,
        verbose=False,
        history_len=HISTORY_LEN,
        candidate_history_len=CANDIDATE_HISTORY_LEN,
        hidden_size=HIDDEN_SIZE,
        layers=LAYERS,
        heads=HEADS,
        dropout=DROPOUT,
        epochs=EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        selection_metric=SELECTION_METRIC,
        early_stop_patience=EARLY_STOP_PATIENCE,
        max_fit_events=MAX_FIT_EVENTS,
        max_train_events=MAX_TRAIN_EVENTS,
        max_val_events=MAX_VAL_EVENTS,
        num_negatives=NUM_NEGATIVES,
        training_candidates="test_like",
        validation_candidates="test_like",
        candidate_recent_feature_group="recency_rank",
        refit_full=False,
    )

    interactions = read_interactions(dataset.train_path)
    context = FitContext(
        dataset=dataset,
        seed=SEED,
        verbose=False,
    )

    report = ranker.fit(interactions, training_config=config, context=context)

    elapsed = perf_counter() - start
    result = {
        "dataset": dataset.name,
        "variant": variant_name,
        "flags": flags,
        "ap": float(report.best_val_ap),
        "mrr": float(report.best_val_mrr),
        "best_epoch": float(report.metrics.get("best_epoch", 0.0)),
        "elapsed_sec": round(elapsed, 2),
    }
    log(
        f"[{dataset.name}] {variant_name} done  "
        f"AP={result['ap']:.5f} MRR={result['mrr']:.5f} "
        f"epoch={result['best_epoch']:.0f} {elapsed:.2f}s",
        enabled=True,
    )
    return result


def _run_dataset(name: str) -> list[dict[str, Any]]:
    dataset = _make_dataset(name)
    results: list[dict[str, Any]] = []
    for variant_name, flags in ABLATIONS:
        result = _run_single(dataset, variant_name, flags)
        results.append(result)
    return results


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    log(f"GPU={gpu_id}  ablation start", enabled=True)

    all_results: list[dict[str, Any]] = []
    for dataset_name in ("dataset1", "dataset2"):
        all_results.extend(_run_dataset(dataset_name))

    out_path = RESULT_DIR / f"baseline_inspired_seed{SEED}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log(f"Results written to {out_path}", enabled=True)
    for r in all_results:
        print(
            f"[{r['dataset']}] {r['variant']:12s}  "
            f"AP={r['ap']:.5f}  MRR={r['mrr']:.5f}  "
            f"epoch={r['best_epoch']:.0f}  {r['elapsed_sec']:.2f}s"
        )


if __name__ == "__main__":
    main()
