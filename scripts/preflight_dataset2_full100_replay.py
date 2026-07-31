from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.full100_training import (
    replay_feature_report,
    validate_candidate_matrix,
)
from jgrec.rankers.hybrid.ranker import (
    SupervisedFeatureBuilder,
    TemporalHybridRanker,
    _sample_events,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the first Dataset2 champion training-feature batch before a full-100 build."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-prefix", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replay-rows", type=int, default=4096)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite replay report: {args.output}")
    _configure_cuda()
    started = time.time()
    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    cached_train = np.load(
        args.cache_prefix.with_suffix(".train.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    if cached_train.ndim != 3 or cached_train.shape[-1] != len(feature_names):
        raise ValueError("source training cache does not match checkpoint feature names")
    expected_candidate_count = config.resolved_train_num_negatives() + 1
    if cached_train.shape[1] != expected_candidate_count:
        raise ValueError("source training cache candidate width does not match checkpoint config")

    interactions = read_interactions(args.train_csv).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    val_size = max(1, int(len(interactions) * config.val_ratio))
    train_end = max(2, len(interactions) - val_size)
    context_end = max(1, min(train_end - 1, int(train_end * config.context_ratio)))
    rng = np.random.default_rng(config.seed)
    sampled_train = _sample_events(
        interactions[context_end:train_end],
        config.max_train_events,
        rng,
    )
    sampled_val = _sample_events(
        interactions[train_end:],
        config.max_val_events,
        rng,
    )
    if len(sampled_train) != cached_train.shape[0]:
        raise ValueError("sampled training rows do not match source cache")

    supervised_config = replace(
        config,
        structure_future_only_transition_cooccur=True,
        supervised_feature_cache_dir=None,
        verbose=True,
    )
    ranker = TemporalHybridRanker(recent_window=int(state["recent_window"]))
    ranker.id_map = NodeIdMap.from_interactions(interactions)
    ranker.dataset_profile = state["dataset_profile"]
    encoder_cache = ranker._encoder_state_cache(
        interactions,
        supervised_config,
        verbose=True,
    )
    train_snapshot = (
        encoder_cache.snapshot_for_prefix(context_end)
        if encoder_cache is not None
        else None
    )
    encoder = ranker._timed_fit_encoder(
        "full100_replay_train_context_encoder",
        interactions[:context_end],
        supervised_config,
        rng,
        verbose=True,
        deterministic_snapshot=train_snapshot,
    )
    if tuple(encoder.feature_names) != feature_names:
        raise RuntimeError("replay encoder feature schema differs from checkpoint")
    if encoder_cache is not None:
        encoder_cache.release_except()
    del train_snapshot, encoder_cache
    gc.collect()

    replay_rows = min(max(args.replay_rows, 1), len(sampled_train))
    replay_rng_state = copy.deepcopy(rng.bit_generator.state)
    replay_rng = np.random.default_rng()
    replay_rng.bit_generator.state = copy.deepcopy(replay_rng_state)
    replay_config = replace(
        supervised_config,
        num_negatives=expected_candidate_count - 1,
    )
    builder = SupervisedFeatureBuilder(
        encoder=encoder,
        dst_pool=np.unique(interactions.dst).astype(np.int64, copy=False),
        config=replay_config,
        label="full100_replay",
    )
    replay_queries = builder.batch_for_events(sampled_train[:replay_rows], replay_rng)
    candidate_report = validate_candidate_matrix(
        sampled_train.dst[:replay_rows],
        replay_queries.candidates,
        expected_candidate_count=expected_candidate_count,
    )
    replay_features = encoder.features_for_query_array(replay_queries)
    feature_report = replay_feature_report(
        cached_train[:replay_rows],
        replay_features,
    )
    report = {
        "status": "passed" if feature_report["matched"] else "rejected",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "cache_prefix": str(args.cache_prefix.resolve()),
        "cache_manifest_sha256": _sha256(args.cache_prefix.with_suffix(".json")),
        "feature_names": list(feature_names),
        "source_train_shape": list(cached_train.shape),
        "replay_rows": replay_rows,
        "sampled_validation_rows": len(sampled_val),
        "split": {
            "context_end": context_end,
            "train_end": train_end,
            "interaction_rows": len(interactions),
        },
        "candidate_report": candidate_report,
        "feature_report": feature_report,
        "replay_rng_state": replay_rng_state,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if feature_report["matched"] else 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_cuda() -> None:
    import jittor as jt  # noqa: PLC0415

    if not jt.has_cuda:
        raise RuntimeError("CUDA is required for the champion replay")
    jt.flags.use_cuda = 1


if __name__ == "__main__":
    raise SystemExit(main())
