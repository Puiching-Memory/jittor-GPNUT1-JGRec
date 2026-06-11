#!/usr/bin/env python3
"""Behavioral analysis for the temporal graph ranker.

Analyzes model decisions on validation data:
  - Error pattern classification (repeat errors, cold-start errors, etc.)
  - Success vs failure case comparison
  - Candidate competition analysis

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/analyze_behavior.py \\
        --data-dir data --dataset dataset1 --max-events 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jittor as jt

from jgrec.core.io import read_interactions, read_test_queries
from jgrec.rankers.temporal_graph.config import TemporalGraphTrainingConfig
from jgrec.rankers.temporal_graph.index import (
    TemporalNodeMap,
    safe_neighbor_sampler,
    temporal_data_from_interactions,
    temporal_loader_api,
)
from jgrec.rankers.temporal_graph.model import (
    EndToEndTemporalGraphModel,
    TemporalGraphModelConfig,
)
from jgrec.rankers.temporal_graph.trainer import (
    CANDIDATE_PRIOR_FEATURE_DIM,
    CandidatePriorIndex,
    TestCandidateIndex,
    build_evaluation_batch,
    load_state,
    predict_logits,
    train_listwise,
    _batch_to_jittor,
    _event_batches,
    _sample_events,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavioral analysis for temporal graph ranker")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--dataset", type=str, default="dataset1")
    p.add_argument("--output-dir", type=Path, default=Path("result/diagnostics"))
    p.add_argument("--seed", type=int, default=60)
    p.add_argument("--max-train-events", type=int, default=5000)
    p.add_argument("--max-val-events", type=int, default=2000)
    p.add_argument("--num-negatives", type=int, default=49)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--history-len", type=int, default=64)
    p.add_argument("--candidate-history-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--load-state", type=Path, default=None)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--max-analysis-batches", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)

    dataset_dir = args.data_dir / args.dataset
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    print(f"[behavior] loading data from {dataset_dir}", flush=True)
    interactions = read_interactions(dataset_dir / "train.csv")
    interactions = interactions.sort_by_time()

    n_events = len(interactions)
    val_size = max(1, int(n_events * args.val_ratio))
    train_end = max(1, n_events - val_size)
    train_events = interactions[:train_end]
    val_events = interactions[train_end:]

    print(f"[behavior] train={len(train_events)} val={len(val_events)}", flush=True)

    # Build infrastructure
    test_path = dataset_dir / "test.csv"
    node_map = TemporalNodeMap.from_interactions_and_test(interactions, test_path)
    full_data = temporal_data_from_interactions(interactions, node_map)
    _, get_neighbor_sampler = temporal_loader_api()
    neighbor_sampler = safe_neighbor_sampler(get_neighbor_sampler(full_data, "recent", seed=args.seed))
    dst_pool = np.unique(node_map.dst_ids(interactions.dst)).astype(np.int32, copy=False)
    dst_pool = dst_pool[dst_pool > 0]

    time_span = max(int(interactions.time[-1]) - int(interactions.time[0]), 1)

    test_candidate_index = None
    if test_path.exists():
        test_candidate_index = TestCandidateIndex.from_queries(
            read_test_queries(test_path), node_map
        )

    candidate_prior_index = None
    if test_candidate_index is not None:
        candidate_prior_index = CandidatePriorIndex.from_test_candidates(
            test_candidate_index,
            node_map.dst_ids(train_events.dst),
            train_times=train_events.time,
            recent_feature_group="recency_rank",
        )

    # Build and train model
    config = TemporalGraphModelConfig(
        num_nodes=node_map.num_nodes,
        history_len=args.history_len,
        candidate_history_len=args.candidate_history_len,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        dropout=0.15,
        time_span=time_span,
        candidate_feature_dim=CANDIDATE_PRIOR_FEATURE_DIM,
    )
    model = EndToEndTemporalGraphModel(config)

    if args.load_state is not None and args.load_state.exists():
        print(f"[behavior] loading model state from {args.load_state}", flush=True)
        state = {k: v for k, v in np.load(str(args.load_state)).items()}
        load_state(model, state)
    else:
        print(f"[behavior] training model for {args.epochs} epochs", flush=True)
        rng = np.random.default_rng(args.seed)
        result = train_listwise(
            model=model,
            train_events=train_events,
            val_events=val_events,
            node_map=node_map,
            neighbor_sampler=neighbor_sampler,
            dst_pool=dst_pool,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_negatives=args.num_negatives,
            lr=args.lr,
            weight_decay=0.0,
            early_stop_patience=3,
            max_train_events=args.max_train_events,
            max_val_events=args.max_val_events,
            selection_metric="ap",
            train_candidate_index=test_candidate_index,
            validation_candidate_index=test_candidate_index,
            candidate_prior_index=candidate_prior_index,
            rng=rng,
            verbose=True,
        )
        print(f"[behavior] trained: AP={result.best_val_ap:.4f} MRR={result.best_val_mrr:.4f}", flush=True)

    # Build interaction history for behavioral analysis
    print("[behavior] building interaction history index", flush=True)
    history_index = build_history_index(interactions[:train_end], node_map)

    # --- Run behavioral analysis ---
    print("[behavior] running behavioral analysis on validation set", flush=True)
    analysis = run_behavioral_analysis(
        model=model,
        val_events=val_events,
        node_map=node_map,
        neighbor_sampler=neighbor_sampler,
        dst_pool=dst_pool,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
        candidate_index=test_candidate_index,
        candidate_prior_index=candidate_prior_index,
        history_index=history_index,
        max_batches=args.max_analysis_batches,
        rng=np.random.default_rng(args.seed + 2000),
    )

    # --- Generate report ---
    report = build_behavior_report(analysis, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"behavior_analysis_{args.dataset}_seed{args.seed}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[behavior] report written to {out_path}", flush=True)

    print_report_summary(report)
    return 0


# ---------------------------------------------------------------------------
# History index for behavioral analysis
# ---------------------------------------------------------------------------


class HistoryIndex:
    """Tracks historical interaction patterns for each node."""

    def __init__(self) -> None:
        self.src_dst_counts: dict[int, dict[int, int]] = {}
        self.dst_last_seen: dict[int, int] = {}
        self.src_last_seen: dict[int, int] = {}
        self.src_total_interactions: dict[int, int] = {}
        self.dst_total_interactions: dict[int, int] = {}

    def get_repeat_count(self, src_id: int, dst_id: int) -> int:
        return self.src_dst_counts.get(src_id, {}).get(dst_id, 0)

    def get_dst_recency(self, dst_id: int, current_time: int) -> int:
        last = self.dst_last_seen.get(dst_id, 0)
        return max(current_time - last, 0) if last > 0 else -1

    def get_src_recency(self, src_id: int, current_time: int) -> int:
        last = self.src_last_seen.get(src_id, 0)
        return max(current_time - last, 0) if last > 0 else -1

    def is_cold_start_src(self, src_id: int) -> bool:
        return self.src_total_interactions.get(src_id, 0) < 3

    def is_cold_start_dst(self, dst_id: int) -> bool:
        return self.dst_total_interactions.get(dst_id, 0) < 3


def build_history_index(interactions, node_map: TemporalNodeMap) -> HistoryIndex:
    index = HistoryIndex()
    src_ids = node_map.src_ids(interactions.src)
    dst_ids = node_map.dst_ids(interactions.dst)
    times = interactions.time

    for src, dst, t in zip(src_ids, dst_ids, times):
        src, dst, t = int(src), int(dst), int(t)
        if src == 0 or dst == 0:
            continue
        index.src_dst_counts.setdefault(src, {})
        index.src_dst_counts[src][dst] = index.src_dst_counts[src].get(dst, 0) + 1
        index.dst_last_seen[dst] = t
        index.src_last_seen[src] = t
        index.src_total_interactions[src] = index.src_total_interactions.get(src, 0) + 1
        index.dst_total_interactions[dst] = index.dst_total_interactions.get(dst, 0) + 1

    return index


# ---------------------------------------------------------------------------
# Behavioral analysis
# ---------------------------------------------------------------------------


class BehaviorCollector:
    """Collects per-sample behavioral data."""

    def __init__(self) -> None:
        self.samples: list[dict] = []

    def add_batch(
        self,
        batch,
        logits: np.ndarray,
        history_index: HistoryIndex,
        node_map: TemporalNodeMap,
    ) -> None:
        src_ids = batch.src_ids
        times = batch.times
        candidates = batch.candidates
        src_neighbor_ids = batch.src_neighbor_ids

        for i in range(len(src_ids)):
            src = int(src_ids[i])
            t = int(times[i])
            cands = candidates[i]
            positive = int(cands[0])
            scores = logits[i]
            pred_rank = int(np.sum(scores > scores[0]))

            # Compute features for each candidate
            candidate_features = []
            for j, cand in enumerate(cands):
                cand = int(cand)
                if cand == 0:
                    continue
                repeat_count = history_index.get_repeat_count(src, cand)
                dst_recency = history_index.get_dst_recency(cand, t)
                is_cold_dst = history_index.is_cold_start_dst(cand)
                in_src_history = int(cand in src_neighbor_ids[i])

                candidate_features.append({
                    "candidate_id": cand,
                    "is_positive": j == 0,
                    "score": float(scores[j]),
                    "repeat_count": repeat_count,
                    "dst_recency": dst_recency,
                    "is_cold_dst": is_cold_dst,
                    "in_src_history": in_src_history,
                })

            # Classify error type
            error_type = "correct"
            if pred_rank > 0:
                top_candidate = candidate_features[np.argmax(scores)]
                if top_candidate["is_cold_dst"]:
                    error_type = "cold_start"
                elif top_candidate["repeat_count"] > 0 and candidate_features[0]["repeat_count"] == 0:
                    error_type = "repeat_bias"
                elif top_candidate["dst_recency"] >= 0 and top_candidate["dst_recency"] < 1000:
                    error_type = "recency_bias"
                else:
                    error_type = "other"

            self.samples.append({
                "src_id": src,
                "time": t,
                "positive_id": positive,
                "pred_rank": pred_rank,
                "error_type": error_type,
                "src_is_cold": history_index.is_cold_start_src(src),
                "positive_repeat_count": history_index.get_repeat_count(src, positive),
                "positive_dst_recency": history_index.get_dst_recency(positive, t),
                "candidate_features": candidate_features,
            })


def run_behavioral_analysis(
    model: EndToEndTemporalGraphModel,
    val_events,
    node_map: TemporalNodeMap,
    neighbor_sampler,
    dst_pool: np.ndarray,
    num_negatives: int,
    batch_size: int,
    candidate_index: TestCandidateIndex | None,
    candidate_prior_index: CandidatePriorIndex | None,
    history_index: HistoryIndex,
    max_batches: int,
    rng: np.random.Generator,
) -> BehaviorCollector:
    model.eval()
    collector = BehaviorCollector()

    val_events = _sample_events(val_events, max_batches * batch_size, rng)
    batch_count = 0
    for batch_events in _event_batches(val_events, batch_size):
        if batch_count >= max_batches:
            break
        batch = build_evaluation_batch(
            events=batch_events,
            node_map=node_map,
            neighbor_sampler=neighbor_sampler,
            dst_pool=dst_pool,
            num_negatives=num_negatives,
            candidate_index=candidate_index,
            candidate_prior_index=candidate_prior_index,
            rng=rng,
            history_len=model.config.history_len,
            candidate_history_len=model.config.candidate_history_len,
        )
        with jt.no_grad():
            logits = model(*_batch_to_jittor(batch))
        jt.sync_all()
        logits_np = np.asarray(logits.numpy(), dtype=np.float32)

        collector.add_batch(batch, logits_np, history_index, node_map)
        batch_count += 1
        print(f"  [behavior] batch {batch_count}/{max_batches} done", flush=True)

    return collector


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_behavior_report(collector: BehaviorCollector, args: argparse.Namespace) -> dict:
    samples = collector.samples
    report: dict = {
        "meta": {
            "dataset": args.dataset,
            "seed": args.seed,
            "total_samples": len(samples),
        },
    }

    # --- 1. Error pattern classification ---
    report["error_patterns"] = analyze_error_patterns(samples)

    # --- 2. Success vs failure comparison ---
    report["success_vs_failure"] = analyze_success_vs_failure(samples)

    # --- 3. Candidate competition ---
    report["candidate_competition"] = analyze_candidate_competition(samples)

    # --- 4. Cold start analysis ---
    report["cold_start"] = analyze_cold_start(samples)

    return report


def analyze_error_patterns(samples: list[dict]) -> dict:
    """Classify and count error types."""
    error_counts: dict[str, int] = {}
    total = len(samples)
    for s in samples:
        et = s["error_type"]
        error_counts[et] = error_counts.get(et, 0) + 1

    result = {
        "total_samples": total,
        "correct": error_counts.get("correct", 0),
        "accuracy": error_counts.get("correct", 0) / max(total, 1),
    }
    for et in ("cold_start", "repeat_bias", "recency_bias", "other"):
        count = error_counts.get(et, 0)
        result[et] = {
            "count": count,
            "share": count / max(total - error_counts.get("correct", 0), 1),
        }

    return result


def analyze_success_vs_failure(samples: list[dict]) -> dict:
    """Compare features of successful vs failed predictions."""
    success = [s for s in samples if s["pred_rank"] == 0]
    failure = [s for s in samples if s["pred_rank"] > 0]

    def feature_dist(sample_list: list[dict], key: str) -> dict:
        values = [s[key] for s in sample_list if isinstance(s.get(key), (int, float))]
        if not values:
            return {"count": 0}
        arr = np.array(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "median": round(float(np.median(arr)), 4),
        }

    return {
        "success": {
            "count": len(success),
            "positive_repeat_count": feature_dist(success, "positive_repeat_count"),
            "positive_dst_recency": feature_dist(success, "positive_dst_recency"),
        },
        "failure": {
            "count": len(failure),
            "positive_repeat_count": feature_dist(failure, "positive_repeat_count"),
            "positive_dst_recency": feature_dist(failure, "positive_dst_recency"),
        },
    }


def analyze_candidate_competition(samples: list[dict]) -> dict:
    """Analyze why the model picks A over B."""
    # For each failed sample, compare the top-ranked candidate with the positive
    comparisons = []
    for s in samples:
        if s["pred_rank"] > 0:
            cands = s["candidate_features"]
            positive = cands[0]
            top_idx = int(np.argmax([c["score"] for c in cands]))
            top_cand = cands[top_idx]
            comparisons.append({
                "positive_repeat": positive["repeat_count"],
                "top_repeat": top_cand["repeat_count"],
                "positive_recency": positive["dst_recency"],
                "top_recency": top_cand["dst_recency"],
                "positive_in_history": positive["in_src_history"],
                "top_in_history": top_cand["in_src_history"],
                "score_gap": top_cand["score"] - positive["score"],
            })

    if not comparisons:
        return {"total_comparisons": 0}

    repeat_wins = sum(1 for c in comparisons if c["top_repeat"] > c["positive_repeat"])
    recency_wins = sum(1 for c in comparisons if c["top_recency"] >= 0 and (c["positive_recency"] < 0 or c["top_recency"] < c["positive_recency"]))
    history_wins = sum(1 for c in comparisons if c["top_in_history"] > c["positive_in_history"])

    return {
        "total_comparisons": len(comparisons),
        "repeat_wins": repeat_wins,
        "repeat_win_share": round(repeat_wins / len(comparisons), 4),
        "recency_wins": recency_wins,
        "recency_win_share": round(recency_wins / len(comparisons), 4),
        "history_wins": history_wins,
        "history_win_share": round(history_wins / len(comparisons), 4),
        "avg_score_gap": round(float(np.mean([c["score_gap"] for c in comparisons])), 4),
    }


def analyze_cold_start(samples: list[dict]) -> dict:
    """Analyze model performance on cold-start cases."""
    cold_src = [s for s in samples if s["src_is_cold"]]
    warm_src = [s for s in samples if not s["src_is_cold"]]

    cold_acc = sum(1 for s in cold_src if s["pred_rank"] == 0) / max(len(cold_src), 1)
    warm_acc = sum(1 for s in warm_src if s["pred_rank"] == 0) / max(len(warm_src), 1)

    cold_dst = [s for s in samples if any(c["is_cold_dst"] and c["is_positive"] for c in s["candidate_features"])]
    warm_dst = [s for s in samples if not any(c["is_cold_dst"] and c["is_positive"] for c in s["candidate_features"])]

    cold_dst_acc = sum(1 for s in cold_dst if s["pred_rank"] == 0) / max(len(cold_dst), 1)
    warm_dst_acc = sum(1 for s in warm_dst if s["pred_rank"] == 0) / max(len(warm_dst), 1)

    return {
        "cold_src": {"count": len(cold_src), "accuracy": round(cold_acc, 4)},
        "warm_src": {"count": len(warm_src), "accuracy": round(warm_acc, 4)},
        "cold_dst": {"count": len(cold_dst), "accuracy": round(cold_dst_acc, 4)},
        "warm_dst": {"count": len(warm_dst), "accuracy": round(warm_dst_acc, 4)},
        "src_gap": round(warm_acc - cold_acc, 4),
        "dst_gap": round(warm_dst_acc - cold_dst_acc, 4),
    }


def print_report_summary(report: dict) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("BEHAVIORAL ANALYSIS REPORT")
    print("=" * 70)

    # Error patterns
    ep = report["error_patterns"]
    print(f"\n[1] ERROR PATTERN CLASSIFICATION (n={ep['total_samples']})")
    print(f"  correct: {ep['correct']} ({ep['accuracy']:.1%})")
    for et in ("cold_start", "repeat_bias", "recency_bias", "other"):
        d = ep[et]
        print(f"  {et:15s}: {d['count']:5d} ({d['share']:.1%} of errors)")

    # Success vs failure
    sf = report["success_vs_failure"]
    print(f"\n[2] SUCCESS vs FAILURE COMPARISON")
    for group in ("success", "failure"):
        g = sf[group]
        print(f"  {group} (n={g['count']}):")
        prc = g["positive_repeat_count"]
        if "mean" in prc:
            print(f"    positive_repeat_count: mean={prc['mean']:.3f}  median={prc['median']:.3f}")
        pdr = g["positive_dst_recency"]
        if "mean" in pdr:
            print(f"    positive_dst_recency:  mean={pdr['mean']:.3f}  median={pdr['median']:.3f}")

    # Candidate competition
    cc = report["candidate_competition"]
    print(f"\n[3] CANDIDATE COMPETITION (failed cases, n={cc['total_comparisons']})")
    if cc["total_comparisons"] > 0:
        print(f"  top candidate beats positive by:")
        print(f"    repeat history: {cc['repeat_wins']} ({cc['repeat_win_share']:.1%})")
        print(f"    recency:        {cc['recency_wins']} ({cc['recency_win_share']:.1%})")
        print(f"    in src history: {cc['history_wins']} ({cc['history_win_share']:.1%})")
        print(f"    avg score gap:  {cc['avg_score_gap']:.4f}")

    # Cold start
    cs = report["cold_start"]
    print(f"\n[4] COLD START ANALYSIS")
    print(f"  src cold ({cs['cold_src']['count']}) acc={cs['cold_src']['accuracy']:.1%}  "
          f"warm ({cs['warm_src']['count']}) acc={cs['warm_src']['accuracy']:.1%}  "
          f"gap={cs['src_gap']:+.1%}")
    print(f"  dst cold ({cs['cold_dst']['count']}) acc={cs['cold_dst']['accuracy']:.1%}  "
          f"warm ({cs['warm_dst']['count']}) acc={cs['warm_dst']['accuracy']:.1%}  "
          f"gap={cs['dst_gap']:+.1%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
