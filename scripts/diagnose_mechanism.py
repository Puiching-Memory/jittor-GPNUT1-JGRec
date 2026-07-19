#!/usr/bin/env python3
"""Mechanism diagnosis for the temporal graph ranker.

Trains (or loads) a model, then runs diagnose_forward on validation data
to inspect internal components:
  - memory gate polarization
  - cross-attention weight distribution
  - time projection effectiveness
  - scorer input signal strengths

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/diagnose_mechanism.py \\
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
    _batch_to_jittor,
    _event_batches,
    _sample_events,
    build_evaluation_batch,
    load_state,
    train_listwise,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mechanism diagnosis for temporal graph ranker")
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
    p.add_argument("--load-state", type=Path, default=None, help="Path to a .npz saved model state")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--max-diagnosis-batches", type=int, default=20, help="Limit diagnosis batches for speed")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)

    dataset_dir = args.data_dir / args.dataset
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    print(f"[diagnose] loading data from {dataset_dir}", flush=True)
    interactions = read_interactions(dataset_dir / "train.csv")
    interactions = interactions.sort_by_time()

    n_events = len(interactions)
    val_size = max(1, int(n_events * args.val_ratio))
    train_end = max(1, n_events - val_size)
    train_events = interactions[:train_end]
    val_events = interactions[train_end:]

    print(f"[diagnose] train={len(train_events)} val={len(val_events)}", flush=True)

    # Build node map and neighbor sampler
    test_path = dataset_dir / "test.csv"
    node_map = TemporalNodeMap.from_interactions_and_test(interactions, test_path)
    full_data = temporal_data_from_interactions(interactions, node_map)
    _, get_neighbor_sampler = temporal_loader_api()
    neighbor_sampler = safe_neighbor_sampler(get_neighbor_sampler(full_data, "recent", seed=args.seed))
    dst_pool = np.unique(node_map.dst_ids(interactions.dst)).astype(np.int32, copy=False)
    dst_pool = dst_pool[dst_pool > 0]

    time_span = max(int(interactions.time[-1]) - int(interactions.time[0]), 1)

    # Build test candidate index
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

    # Build model
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
        print(f"[diagnose] loading model state from {args.load_state}", flush=True)
        state = dict(np.load(str(args.load_state)).items())
        load_state(model, state)
    else:
        print(f"[diagnose] training model for {args.epochs} epochs", flush=True)
        rng = np.random.default_rng(args.seed)
        training_config = TemporalGraphTrainingConfig(
            epochs=args.epochs,
            train_batch_size=args.batch_size,
            lr=args.lr,
            num_negatives=args.num_negatives,
            max_train_events=args.max_train_events,
            max_val_events=args.max_val_events,
            seed=args.seed,
            verbose=True,
            history_len=args.history_len,
            candidate_history_len=args.candidate_history_len,
            hidden_size=args.hidden_size,
            layers=args.layers,
            heads=args.heads,
            selection_metric="ap",
            early_stop_patience=3,
            training_candidates="test_like",
            validation_candidates="test_like",
            candidate_recent_feature_group="recency_rank",
            refit_full=False,
        )
        result = train_listwise(
            model=model,
            train_events=train_events,
            val_events=val_events,
            node_map=node_map,
            neighbor_sampler=neighbor_sampler,
            dst_pool=dst_pool,
            epochs=training_config.epochs,
            batch_size=training_config.train_batch_size,
            num_negatives=training_config.num_negatives,
            lr=training_config.lr,
            weight_decay=0.0,
            early_stop_patience=training_config.early_stop_patience,
            max_train_events=training_config.max_train_events,
            max_val_events=training_config.max_val_events,
            selection_metric=training_config.selection_metric,
            train_candidate_index=test_candidate_index,
            validation_candidate_index=test_candidate_index,
            candidate_prior_index=candidate_prior_index,
            rng=rng,
            verbose=True,
        )
        print(f"[diagnose] trained: AP={result.best_val_ap:.4f} MRR={result.best_val_mrr:.4f}", flush=True)

    # --- Run diagnosis ---
    print("[diagnose] running mechanism diagnosis on validation set", flush=True)
    diagnosis = run_diagnosis(
        model=model,
        val_events=val_events,
        node_map=node_map,
        neighbor_sampler=neighbor_sampler,
        dst_pool=dst_pool,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
        candidate_index=test_candidate_index,
        candidate_prior_index=candidate_prior_index,
        max_batches=args.max_diagnosis_batches,
        rng=np.random.default_rng(args.seed + 1000),
    )

    # --- Generate report ---
    report = build_report(diagnosis, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"mechanism_diagnosis_{args.dataset}_seed{args.seed}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[diagnose] report written to {out_path}", flush=True)

    print_report_summary(report)
    return 0


# ---------------------------------------------------------------------------
# Diagnosis collection
# ---------------------------------------------------------------------------


class DiagnosisCollector:
    """Accumulates diagnostic traces across batches."""

    def __init__(self) -> None:
        self.src_gates: list[np.ndarray] = []
        self.candidate_gates: list[np.ndarray] = []
        self.attention_weights: list[np.ndarray] = []
        self.attention_key_masks: list[np.ndarray] = []
        self.signal_norms: dict[str, list[np.ndarray]] = {}
        self.time_deltas_src: list[np.ndarray] = []
        self.time_encodings_src: list[np.ndarray] = []
        self.time_deltas_candidate: list[np.ndarray] = []
        self.time_encodings_candidate: list[np.ndarray] = []
        self.pair_stats_raw: list[np.ndarray] = []
        self.logits: list[np.ndarray] = []

    def add(self, trace, batch_size: int, candidate_count: int) -> None:
        # Gate values
        self.src_gates.append(np.asarray(trace.src_gate_values.numpy()))
        self.candidate_gates.append(np.asarray(trace.candidate_gate_values.numpy()))

        # Attention weights: [batch*cand, heads, 1, key_len]
        attn_w = np.asarray(trace.attention_weights.numpy())
        self.attention_weights.append(attn_w)
        self.attention_key_masks.append(np.asarray(trace.attention_key_mask.numpy()))

        # Signal norms
        for key, var in trace.signal_norms.items():
            self.signal_norms.setdefault(key, []).append(np.asarray(var.numpy()))

        # Time encodings
        self.time_deltas_src.append(np.asarray(trace.time_deltas_src.numpy()))
        self.time_encodings_src.append(np.asarray(trace.time_encodings_src.numpy()))
        self.time_deltas_candidate.append(np.asarray(trace.time_deltas_candidate.numpy()))
        self.time_encodings_candidate.append(np.asarray(trace.time_encodings_candidate.numpy()))

        # Stats and logits
        self.pair_stats_raw.append(np.asarray(trace.pair_stats_raw.numpy()))
        self.logits.append(np.asarray(trace.logits.numpy()))


def run_diagnosis(
    model: EndToEndTemporalGraphModel,
    val_events,
    node_map: TemporalNodeMap,
    neighbor_sampler,
    dst_pool: np.ndarray,
    num_negatives: int,
    batch_size: int,
    candidate_index: TestCandidateIndex | None,
    candidate_prior_index: CandidatePriorIndex | None,
    max_batches: int,
    rng: np.random.Generator,
) -> DiagnosisCollector:
    model.eval()
    collector = DiagnosisCollector()

    val_events = _sample_events(val_events, max_batches * batch_size, rng)
    for batch_count, batch_events in enumerate(_event_batches(val_events, batch_size), start=1):
        if batch_count > max_batches:
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
        jittor_inputs = _batch_to_jittor(batch)
        with jt.no_grad():
            trace = model.diagnose_forward(*jittor_inputs)
        jt.sync_all()

        bsz = batch.src_ids.shape[0]
        cand_count = batch.candidates.shape[1]
        collector.add(trace, bsz, cand_count)
        print(f"  [diagnose] batch {batch_count}/{max_batches} done", flush=True)

    return collector


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_report(collector: DiagnosisCollector, args: argparse.Namespace) -> dict:
    report: dict = {
        "meta": {
            "dataset": args.dataset,
            "seed": args.seed,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "heads": args.heads,
            "history_len": args.history_len,
            "candidate_history_len": args.candidate_history_len,
        },
    }

    # --- 1. Gate diagnostics ---
    report["gate"] = analyze_gates(collector)

    # --- 2. Attention diagnostics ---
    report["attention"] = analyze_attention(collector)

    # --- 3. Time projection diagnostics ---
    report["time_projection"] = analyze_time_projection(collector)

    # --- 4. Scorer signal diagnostics ---
    report["scorer_signals"] = analyze_scorer_signals(collector)

    # --- 5. Stats distribution ---
    report["pair_stats"] = analyze_pair_stats(collector)

    return report


def analyze_gates(collector: DiagnosisCollector) -> dict:
    """Analyze memory gate polarization."""
    src_gates = np.concatenate([g.reshape(-1) for g in collector.src_gates])
    cand_gates = np.concatenate([g.reshape(-1) for g in collector.candidate_gates])

    def gate_stats(values: np.ndarray, label: str) -> dict:
        hist_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(values, bins=hist_bins)
        _ = max(len(values), 1)  # sample count; kept for future diagnostics
        polarization_low = float(np.mean(values < 0.2))
        polarization_high = float(np.mean(values > 0.8))
        polarization_ratio = polarization_low + polarization_high
        near_half = float(np.mean(np.abs(values - 0.5) < 0.1))
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "polarization_ratio": round(polarization_ratio, 4),
            "near_zero_pct": round(polarization_low, 4),
            "near_one_pct": round(polarization_high, 4),
            "near_half_pct": round(near_half, 4),
            "histogram": {
                f"{hist_bins[i]:.1f}-{hist_bins[i+1]:.1f}": int(hist[i])
                for i in range(len(hist))
            },
        }

    return {
        "src_gate": gate_stats(src_gates, "src"),
        "candidate_gate": gate_stats(cand_gates, "candidate"),
    }


def analyze_attention(collector: DiagnosisCollector) -> dict:
    """Analyze attention weight distribution across key positions.

    Key layout: [src_self(1), dst_self(1), src_hist(history_len), dst_hist(candidate_history_len)]
    """
    all_attn = np.concatenate(collector.attention_weights, axis=0)  # [N, heads, 1, key_len]
    _ = np.concatenate(collector.attention_key_masks, axis=0)  # [N, key_len]; kept for future mask-aware analysis

    _, n_heads, _, key_len = all_attn.shape

    # Average attention weight per position across all samples and heads
    # Shape: [N, heads, key_len] (squeeze the query dim)
    attn_squeezed = all_attn[:, :, 0, :]  # [N, heads, key_len]

    # Segment analysis: src_self, dst_self, src_hist, dst_hist
    # We need to know the key layout dimensions
    # key_len = 2 + history_len + candidate_history_len
    # But we don't know these from the collector alone; infer from key_len
    # The first 2 positions are always src_self and dst_self

    # Average attention across all samples, per head
    avg_per_head = attn_squeezed.mean(axis=0)  # [heads, key_len]

    # For each head, compute the fraction of attention going to each segment
    # We'll report the first/last positions and the middle segments
    head_analysis = {}
    for h in range(n_heads):
        head_weights = avg_per_head[h]  # [key_len]
        src_self_attn = float(head_weights[0])
        dst_self_attn = float(head_weights[1])
        # The rest is split between src_hist and dst_hist
        # We don't know the exact split, so report the total history attention
        hist_attn = float(head_weights[2:].sum())
        head_analysis[f"head_{h}"] = {
            "src_self": round(src_self_attn, 6),
            "dst_self": round(dst_self_attn, 6),
            "history_total": round(hist_attn, 6),
            "max_position": int(np.argmax(head_weights)),
            "max_weight": round(float(np.max(head_weights)), 6),
            "entropy": round(float(-np.sum(head_weights * np.log(head_weights + 1e-10))), 4),
        }

    # Overall attention distribution
    avg_across_heads = attn_squeezed.mean(axis=1)  # [N, key_len]
    overall_src_self = float(avg_across_heads[:, 0].mean())
    overall_dst_self = float(avg_across_heads[:, 1].mean())
    overall_hist = float(avg_across_heads[:, 2:].mean(axis=1).mean())

    return {
        "n_heads": n_heads,
        "key_len": key_len,
        "overall": {
            "src_self_avg": round(overall_src_self, 6),
            "dst_self_avg": round(overall_dst_self, 6),
            "history_avg": round(overall_hist, 6),
        },
        "per_head": head_analysis,
        "attention_concentration": {
            "max_weight_mean": round(float(attn_squeezed.max(axis=-1).mean()), 6),
            "top3_share_mean": round(float(
                np.sort(attn_squeezed, axis=-1)[:, :, -3:].sum(axis=-1).mean()
            ), 6),
        },
    }


def analyze_time_projection(collector: DiagnosisCollector) -> dict:
    """Analyze time encoding effectiveness.

    Check if time encodings vary monotonically with time delta.
    """
    # src time deltas: [batch, history_len]
    src_deltas = np.concatenate(collector.time_deltas_src, axis=0)
    src_encodings = np.concatenate(collector.time_encodings_src, axis=0)  # [batch, history_len, hidden]

    # For each position in history, compute cosine similarity between
    # the time encoding and a reference (position 0)
    # This checks if different time deltas produce different encodings

    # Flatten to [N, hidden] where N = batch * history_len
    flat_deltas = src_deltas.reshape(-1)
    flat_encodings = src_encodings.reshape(-1, src_encodings.shape[-1])

    # Filter out zero deltas (padding)
    valid_mask = flat_deltas > 0
    valid_deltas = flat_deltas[valid_mask]
    valid_encodings = flat_encodings[valid_mask]

    if len(valid_deltas) < 10:
        return {"error": "insufficient valid time deltas for analysis"}

    # Sort by delta and compute cosine similarity between consecutive encodings
    sort_idx = np.argsort(valid_deltas)
    sorted_deltas = valid_deltas[sort_idx]
    sorted_encodings = valid_encodings[sort_idx]

    # Compute cosine similarity between consecutive pairs
    n_pairs = min(1000, len(sorted_deltas) - 1)
    step = max(1, len(sorted_deltas) // n_pairs)
    cos_sims = []
    delta_diffs = []
    for i in range(0, len(sorted_deltas) - step, step):
        a = sorted_encodings[i]
        b = sorted_encodings[i + step]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 1e-8 and norm_b > 1e-8:
            cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
            cos_sims.append(cos_sim)
            delta_diffs.append(float(sorted_deltas[i + step] - sorted_deltas[i]))

    if not cos_sims:
        return {"error": "could not compute cosine similarities"}

    cos_sims = np.array(cos_sims)
    delta_diffs = np.array(delta_diffs)

    # Correlation between delta difference and (1 - cosine similarity)
    # Higher correlation means time encoding is more monotonic
    dissimilarities = 1.0 - cos_sims
    correlation = float(np.corrcoef(delta_diffs, dissimilarities)[0, 1]) if len(cos_sims) > 2 else 0.0

    return {
        "valid_samples": len(valid_deltas),
        "delta_range": [float(sorted_deltas.min()), float(sorted_deltas.max())],
        "encoding_norm_mean": round(float(np.linalg.norm(valid_encodings, axis=1).mean()), 4),
        "encoding_norm_std": round(float(np.linalg.norm(valid_encodings, axis=1).std()), 4),
        "cos_sim_pairs_analyzed": len(cos_sims),
        "cos_sim_mean": round(float(cos_sims.mean()), 4),
        "cos_sim_std": round(float(cos_sims.std()), 4),
        "delta_dissimilarity_correlation": round(correlation, 4),
        "interpretation": (
            "GOOD: time encoding varies monotonically with delta"
            if correlation > 0.3
            else "WEAK: time encoding may not distinguish time deltas well"
        ),
    }


def analyze_scorer_signals(collector: DiagnosisCollector) -> dict:
    """Analyze which input signals dominate the scorer."""
    signal_stats = {}
    for signal_name, norm_list in collector.signal_norms.items():
        norms = np.concatenate([n.reshape(-1) for n in norm_list])
        signal_stats[signal_name] = {
            "mean": round(float(norms.mean()), 4),
            "std": round(float(norms.std()), 4),
            "min": round(float(norms.min()), 4),
            "max": round(float(norms.max()), 4),
            "median": round(float(np.median(norms)), 4),
        }

    # Compute relative importance: mean norm / total mean norm
    total_mean = sum(s["mean"] for s in signal_stats.values())
    if total_mean > 0:
        for name in signal_stats:
            signal_stats[name]["relative_share"] = round(
                signal_stats[name]["mean"] / total_mean, 4
            )

    return signal_stats


def analyze_pair_stats(collector: DiagnosisCollector) -> dict:
    """Analyze the distribution of raw pair statistics."""
    all_stats = np.concatenate(
        [s.reshape(-1, s.shape[-1]) for s in collector.pair_stats_raw], axis=0
    )
    stat_names = [
        "repeat_count",
        "repeat_recent_position",
        "src_len",
        "candidate_len",
        "src_recency",
        "candidate_recency",
        "pair_recency",
    ]
    # Add candidate feature names if present
    n_base = len(stat_names)
    n_extra = all_stats.shape[1] - n_base
    if n_extra > 0:
        stat_names.extend([f"candidate_feat_{i}" for i in range(n_extra)])

    result = {}
    for i, name in enumerate(stat_names):
        col = all_stats[:, i]
        result[name] = {
            "mean": round(float(col.mean()), 6),
            "std": round(float(col.std()), 6),
            "min": round(float(col.min()), 6),
            "max": round(float(col.max()), 6),
            "zero_pct": round(float(np.mean(col == 0)), 4),
        }

    return result


def print_report_summary(report: dict) -> None:
    """Print a human-readable summary of the diagnosis report."""
    print("\n" + "=" * 70)
    print("MECHANISM DIAGNOSIS REPORT")
    print("=" * 70)

    # Gate
    gate = report["gate"]
    print("\n[1] MEMORY GATE DIAGNOSIS")
    for role in ("src_gate", "candidate_gate"):
        g = gate[role]
        print(f"  {role}:")
        print(f"    mean={g['mean']:.3f}  std={g['std']:.3f}  median={g['median']:.3f}")
        print(f"    polarization: near_0={g['near_zero_pct']:.1%}  near_1={g['near_one_pct']:.1%}  near_half={g['near_half_pct']:.1%}")
        if g["polarization_ratio"] > 0.5:
            print(f"    -> GOOD: gate is polarized (ratio={g['polarization_ratio']:.2f})")
        elif g["near_half_pct"] > 0.5:
            print(f"    -> WEAK: gate is stuck near 0.5 (ratio={g['polarization_ratio']:.2f})")
        else:
            print(f"    -> MODERATE: gate is diffuse (ratio={g['polarization_ratio']:.2f})")

    # Attention
    attn = report["attention"]
    print(f"\n[2] CROSS-ATTENTION DIAGNOSIS (key_len={attn['key_len']}, heads={attn['n_heads']})")
    ov = attn["overall"]
    print(f"  overall: src_self={ov['src_self_avg']:.4f}  dst_self={ov['dst_self_avg']:.4f}  history={ov['history_avg']:.4f}")
    for head_name, head_data in attn["per_head"].items():
        print(f"  {head_name}: src_self={head_data['src_self']:.4f}  dst_self={head_data['dst_self']:.4f}  "
              f"hist={head_data['history_total']:.4f}  max_pos={head_data['max_position']}  "
              f"entropy={head_data['entropy']:.3f}")
    conc = attn["attention_concentration"]
    print(f"  concentration: max_weight_mean={conc['max_weight_mean']:.4f}  top3_share={conc['top3_share_mean']:.4f}")

    # Time projection
    tp = report["time_projection"]
    print("\n[3] TIME PROJECTION DIAGNOSIS")
    if "error" in tp:
        print(f"  ERROR: {tp['error']}")
    else:
        print(f"  valid_samples={tp['valid_samples']}  delta_range={tp['delta_range']}")
        print(f"  encoding_norm: mean={tp['encoding_norm_mean']:.4f}  std={tp['encoding_norm_std']:.4f}")
        print(f"  cos_sim: mean={tp['cos_sim_mean']:.4f}  std={tp['cos_sim_std']:.4f}")
        print(f"  delta-dissimilarity correlation={tp['delta_dissimilarity_correlation']:.4f}")
        print(f"  -> {tp['interpretation']}")

    # Scorer signals
    ss = report["scorer_signals"]
    print("\n[4] SCORER INPUT SIGNALS")
    for name, stats in sorted(ss.items(), key=lambda x: -x[1].get("relative_share", 0)):
        share = stats.get("relative_share", 0)
        print(f"  {name:20s}: mean_norm={stats['mean']:.4f}  std={stats['std']:.4f}  share={share:.1%}")

    # Pair stats
    ps = report["pair_stats"]
    print("\n[5] PAIR STATISTICS DISTRIBUTION")
    for name, stats in ps.items():
        print(f"  {name:25s}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"zero_pct={stats['zero_pct']:.1%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
