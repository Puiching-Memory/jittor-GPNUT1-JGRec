from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    "pair_strength",
    "repeat_rate",
    "pair_recency",
    "dst_popularity",
    "dst_recency",
    "recent_hit",
    "src_activity",
    "src_recency",
)
RECENT_KS = (1, 5, 10, 32)


@dataclass(frozen=True)
class Event:
    src: int
    dst: int
    time: int


@dataclass(frozen=True)
class Query:
    src: int
    time: int
    candidates: tuple[int, ...]


@dataclass
class SourceState:
    total: int = 0
    last_time: int = 0
    dst_counts: Counter[int] = field(default_factory=Counter)
    dst_last_time: dict[int, int] = field(default_factory=dict)
    recent_dsts: deque[int] = field(default_factory=lambda: deque(maxlen=max(RECENT_KS)))
    recent_ranks: dict[int, int] = field(default_factory=dict)


@dataclass
class DataState:
    sources: dict[int, SourceState]
    dst_counts: Counter[int]
    dst_last_time: dict[int, int]
    pair_counts: Counter[tuple[int, int]]
    in_neighbors: dict[int, set[int]]
    out_neighbors: dict[int, set[int]]
    src_set: set[int]
    dst_set: set[int]
    min_time: int
    max_time: int
    graph_span: int
    log_total_edges: float
    log_total_src: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze local train/test data distributions and feature signals.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("result/benchmarks/current_data_deep_profile.json"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--feature-sample", type=int, default=3000)
    parser.add_argument("--fusion-train-sample", type=int, default=2000)
    parser.add_argument("--fusion-val-sample", type=int, default=3000)
    parser.add_argument("--graph-sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    results = []
    for dataset_dir in sorted(args.data_dir.iterdir()):
        if not (dataset_dir / "train.csv").exists() or not (dataset_dir / "test.csv").exists():
            continue
        started = time.perf_counter()
        print(f"ANALYZE {dataset_dir.name}", flush=True)
        result = analyze_dataset(dataset_dir, args)
        result["seconds"] = round(time.perf_counter() - started, 3)
        results.append(result)
        print("SUMMARY " + json.dumps(_brief_summary(result), ensure_ascii=False, sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {args.output}", flush=True)
    return 0


def analyze_dataset(dataset_dir: Path, args: argparse.Namespace) -> dict:
    events = read_train(dataset_dir / "train.csv")
    test_queries = read_test(dataset_dir / "test.csv")
    split = max(1, int(len(events) * (1.0 - args.val_ratio)))
    prefix_events = events[:split]
    val_events = events[split:]
    full_state = build_state(events)
    prefix_state = build_state(prefix_events)
    rng = np.random.default_rng(args.seed + sum(ord(ch) for ch in dataset_dir.name))
    test_index = build_test_index(test_queries)

    return {
        "dataset": dataset_dir.name,
        "protocol": {
            "val_ratio": args.val_ratio,
            "feature_sample": args.feature_sample,
            "fusion_train_sample": args.fusion_train_sample,
            "fusion_val_sample": args.fusion_val_sample,
            "graph_sample": args.graph_sample,
            "seed": args.seed,
        },
        "basic": basic_stats(events, test_queries, full_state),
        "time_drift": time_drift(events),
        "test_candidate_distribution": test_candidate_distribution(test_queries, full_state),
        "unseen_dst_analysis": unseen_dst_analysis(test_queries, full_state),
        "graph_structure": graph_structure(full_state),
        "sequence_behavior": sequence_behavior(events, prefix_state, val_events),
        "segmented_feature_effectiveness": segmented_feature_effectiveness(
            prefix_state,
            val_events,
            test_index,
            rng,
            sample_size=args.feature_sample,
        ),
        "graph_signal_proxy": graph_signal_proxy(
            prefix_state,
            val_events,
            test_index,
            rng,
            sample_size=args.graph_sample,
        ),
        "proxy_validation_calibration": proxy_validation_calibration(
            prefix_state,
            val_events,
            test_index,
            rng,
            sample_size=args.feature_sample,
        ),
        "fusion_permutation_importance": fusion_permutation_importance(
            prefix_events,
            val_events,
            test_index,
            rng,
            train_sample=args.fusion_train_sample,
            val_sample=args.fusion_val_sample,
        ),
    }


def read_train(path: Path) -> list[Event]:
    events: list[Event] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            events.append(Event(src=int(row["src"]), dst=int(row["dst"]), time=int(row["time"])))
    events.sort(key=lambda item: item.time)
    return events


def read_test(path: Path) -> list[Query]:
    queries: list[Query] = []
    with path.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        expected = len(header) - 2
        for row in reader:
            candidates = tuple(int(value) for value in row[2:])
            if len(candidates) != expected:
                raise ValueError(f"{path} has inconsistent candidate count")
            queries.append(Query(src=int(row[0]), time=int(row[1]), candidates=candidates))
    return queries


def build_state(events: list[Event]) -> DataState:
    if not events:
        raise ValueError("events cannot be empty")
    sources: dict[int, SourceState] = defaultdict(SourceState)
    dst_counts: Counter[int] = Counter()
    dst_last_time: dict[int, int] = {}
    pair_counts: Counter[tuple[int, int]] = Counter()
    in_neighbors: dict[int, set[int]] = defaultdict(set)
    out_neighbors: dict[int, set[int]] = defaultdict(set)
    for event in events:
        source = sources[event.src]
        source.total += 1
        source.last_time = event.time
        source.dst_counts[event.dst] += 1
        source.dst_last_time[event.dst] = event.time
        source.recent_dsts.append(event.dst)
        dst_counts[event.dst] += 1
        dst_last_time[event.dst] = event.time
        pair_counts[(event.src, event.dst)] += 1
        in_neighbors[event.dst].add(event.src)
        out_neighbors[event.src].add(event.dst)

    for source in sources.values():
        ranks: dict[int, int] = {}
        for rank, dst in enumerate(reversed(source.recent_dsts), start=1):
            ranks.setdefault(dst, rank)
        source.recent_ranks = ranks

    min_time = events[0].time
    max_time = events[-1].time
    return DataState(
        sources=dict(sources),
        dst_counts=dst_counts,
        dst_last_time=dst_last_time,
        pair_counts=pair_counts,
        in_neighbors=dict(in_neighbors),
        out_neighbors=dict(out_neighbors),
        src_set=set(sources),
        dst_set=set(dst_counts),
        min_time=min_time,
        max_time=max_time,
        graph_span=max(max_time - min_time, 1),
        log_total_edges=math.log1p(len(events)),
        log_total_src=math.log1p(max(len(sources), 1)),
    )


def build_test_index(queries: list[Query]) -> dict:
    by_src: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    global_candidates: list[int] = []
    for query in queries:
        by_src[query.src].append(query.candidates)
        global_candidates.extend(query.candidates)
    return {
        "by_src": dict(by_src),
        "global_candidates": np.asarray(global_candidates, dtype=np.int64),
    }


def basic_stats(events: list[Event], test_queries: list[Query], state: DataState) -> dict:
    repeated_pairs = sum(1 for count in state.pair_counts.values() if count > 1)
    repeated_pair_events = sum(count for count in state.pair_counts.values() if count > 1)
    return {
        "train_edges": len(events),
        "test_queries": len(test_queries),
        "unique_src": len(state.src_set),
        "unique_dst": len(state.dst_set),
        "unique_pairs": len(state.pair_counts),
        "duplicate_event_rate": 1.0 - len(state.pair_counts) / len(events),
        "repeated_pair_rate_among_pairs": repeated_pairs / len(state.pair_counts),
        "event_share_on_repeated_pairs": repeated_pair_events / len(events),
        "src_dst_id_overlap": len(state.src_set & state.dst_set),
        "src_dst_id_overlap_src_rate": len(state.src_set & state.dst_set) / len(state.src_set),
        "src_dst_id_overlap_dst_rate": len(state.src_set & state.dst_set) / len(state.dst_set),
        "min_time": events[0].time,
        "max_time": events[-1].time,
        "time_span": events[-1].time - events[0].time,
        "src_event_degree": qstats(source.total for source in state.sources.values()),
        "dst_event_degree": qstats(state.dst_counts.values()),
        "src_event_gini": gini(source.total for source in state.sources.values()),
        "dst_event_gini": gini(state.dst_counts.values()),
    }


def time_drift(events: list[Event]) -> dict:
    seen_src: set[int] = set()
    seen_dst: set[int] = set()
    seen_pair: set[tuple[int, int]] = set()
    recent: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=max(RECENT_KS)))
    buckets = np.array_split(np.arange(len(events)), 10)
    rows = []
    previous_top100: set[int] | None = None
    for bucket_id, indices in enumerate(buckets):
        counts = Counter()
        dst_counter: Counter[int] = Counter()
        src_counter: Counter[int] = Counter()
        pair_counter: Counter[tuple[int, int]] = Counter()
        start_time = events[int(indices[0])].time
        end_time = events[int(indices[-1])].time
        for raw_idx in indices:
            event = events[int(raw_idx)]
            counts["events"] += 1
            if event.src in seen_src:
                counts["src_seen_before"] += 1
            if event.dst in seen_dst:
                counts["dst_seen_before"] += 1
            if (event.src, event.dst) in seen_pair:
                counts["pair_seen_before"] += 1
            history = list(recent[event.src])
            for k in RECENT_KS:
                if event.dst in set(history[-k:]):
                    counts[f"recent_{k}_hit"] += 1
            seen_src.add(event.src)
            seen_dst.add(event.dst)
            seen_pair.add((event.src, event.dst))
            recent[event.src].append(event.dst)
            src_counter[event.src] += 1
            dst_counter[event.dst] += 1
            pair_counter[(event.src, event.dst)] += 1

        top100 = {dst for dst, _ in dst_counter.most_common(100)}
        rows.append(
            {
                "bucket": bucket_id,
                "events": counts["events"],
                "start_time": start_time,
                "end_time": end_time,
                "unique_src": len(src_counter),
                "unique_dst": len(dst_counter),
                "unique_pairs": len(pair_counter),
                "src_seen_before": rate(counts["src_seen_before"], counts["events"]),
                "dst_seen_before": rate(counts["dst_seen_before"], counts["events"]),
                "pair_seen_before": rate(counts["pair_seen_before"], counts["events"]),
                "new_pair_rate": 1.0 - rate(counts["pair_seen_before"], counts["events"]),
                "recent_1_hit": rate(counts["recent_1_hit"], counts["events"]),
                "recent_5_hit": rate(counts["recent_5_hit"], counts["events"]),
                "recent_10_hit": rate(counts["recent_10_hit"], counts["events"]),
                "recent_32_hit": rate(counts["recent_32_hit"], counts["events"]),
                "top10_dst_event_share": sum(count for _, count in dst_counter.most_common(10)) / counts["events"],
                "top100_dst_jaccard_prev": (
                    len(top100 & previous_top100) / len(top100 | previous_top100)
                    if previous_top100 is not None and (top100 | previous_top100)
                    else None
                ),
            }
        )
        previous_top100 = top100
    return {"deciles": rows}


def test_candidate_distribution(queries: list[Query], state: DataState) -> dict:
    per_query = {
        "known_dst": [],
        "unseen_dst": [],
        "pair_hit": [],
        "recent32_hit": [],
        "top100_dst": [],
        "top1000_dst": [],
        "same_id_as_train_src": [],
    }
    dst_rank = {dst: rank for rank, (dst, _) in enumerate(state.dst_counts.most_common(), start=1)}
    counts = Counter()
    seen_candidate_ranks: list[int] = []
    seen_candidate_counts: list[int] = []
    for query in queries:
        source = state.sources.get(query.src)
        recent32 = source.recent_ranks if source is not None else {}
        query_counts = Counter()
        for dst in query.candidates:
            counts["candidates"] += 1
            rank = dst_rank.get(dst)
            if rank is None:
                query_counts["unseen_dst"] += 1
                counts["unseen_dst"] += 1
            else:
                query_counts["known_dst"] += 1
                counts["known_dst"] += 1
                seen_candidate_ranks.append(rank)
                seen_candidate_counts.append(state.dst_counts[dst])
                if rank <= 100:
                    query_counts["top100_dst"] += 1
                    counts["top100_dst"] += 1
                if rank <= 1000:
                    query_counts["top1000_dst"] += 1
                    counts["top1000_dst"] += 1
            if (query.src, dst) in state.pair_counts:
                query_counts["pair_hit"] += 1
                counts["pair_hit"] += 1
            if dst in recent32:
                query_counts["recent32_hit"] += 1
                counts["recent32_hit"] += 1
            if dst in state.src_set:
                query_counts["same_id_as_train_src"] += 1
                counts["same_id_as_train_src"] += 1
        for key in per_query:
            per_query[key].append(query_counts[key])
        if query_counts["pair_hit"]:
            counts["queries_with_pair_hit"] += 1
        if query_counts["recent32_hit"]:
            counts["queries_with_recent32_hit"] += 1
    total_candidates = counts["candidates"]
    total_queries = len(queries)
    return {
        "rates": {
            "known_dst_candidate_rate": rate(counts["known_dst"], total_candidates),
            "unseen_dst_candidate_rate": rate(counts["unseen_dst"], total_candidates),
            "pair_hit_candidate_rate": rate(counts["pair_hit"], total_candidates),
            "recent32_candidate_rate": rate(counts["recent32_hit"], total_candidates),
            "top100_dst_candidate_rate": rate(counts["top100_dst"], total_candidates),
            "top1000_dst_candidate_rate": rate(counts["top1000_dst"], total_candidates),
            "same_id_as_train_src_candidate_rate": rate(counts["same_id_as_train_src"], total_candidates),
            "query_with_pair_hit_rate": rate(counts["queries_with_pair_hit"], total_queries),
            "query_with_recent32_hit_rate": rate(counts["queries_with_recent32_hit"], total_queries),
        },
        "per_query": {key: qstats(value) for key, value in per_query.items()},
        "seen_candidate_dst_train_count": qstats(seen_candidate_counts),
        "seen_candidate_dst_rank": qstats(seen_candidate_ranks),
    }


def unseen_dst_analysis(queries: list[Query], state: DataState) -> dict:
    unseen_values: list[int] = []
    all_candidate_values: list[int] = []
    for query in queries:
        for dst in query.candidates:
            all_candidate_values.append(dst)
            if dst not in state.dst_set:
                unseen_values.append(dst)
    unseen_set = set(unseen_values)
    all_set = set(all_candidate_values)
    if unseen_values:
        min_train_dst = min(state.dst_set)
        max_train_dst = max(state.dst_set)
        inside_range = sum(1 for value in unseen_values if min_train_dst <= value <= max_train_dst)
    else:
        min_train_dst = max_train_dst = 0
        inside_range = 0
    return {
        "unique_test_candidate_dst": len(all_set),
        "unique_unseen_dst": len(unseen_set),
        "unseen_candidate_events": len(unseen_values),
        "unseen_candidate_event_rate": rate(len(unseen_values), len(all_candidate_values)),
        "unseen_unique_overlap_train_src": len(unseen_set & state.src_set),
        "unseen_event_overlap_train_src_rate": rate(sum(1 for value in unseen_values if value in state.src_set), len(unseen_values)),
        "unseen_unique_inside_train_dst_id_range": sum(1 for value in unseen_set if min_train_dst <= value <= max_train_dst),
        "unseen_event_inside_train_dst_id_range_rate": rate(inside_range, len(unseen_values)),
        "unseen_min": min(unseen_values) if unseen_values else None,
        "unseen_max": max(unseen_values) if unseen_values else None,
        "train_dst_min": min_train_dst,
        "train_dst_max": max_train_dst,
    }


def graph_structure(state: DataState) -> dict:
    src_ids = {src: idx for idx, src in enumerate(state.src_set)}
    offset = len(src_ids)
    dst_ids = {dst: offset + idx for idx, dst in enumerate(state.dst_set)}
    parent = list(range(len(src_ids) + len(dst_ids)))
    size = [1] * len(parent)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for src, dst in state.pair_counts:
        union(src_ids[src], dst_ids[dst])

    component_src = Counter()
    component_dst = Counter()
    component_edges = Counter()
    for src in state.src_set:
        component_src[find(src_ids[src])] += 1
    for dst in state.dst_set:
        component_dst[find(dst_ids[dst])] += 1
    for src, dst in state.pair_counts:
        component_edges[find(src_ids[src])] += state.pair_counts[(src, dst)]

    components = sorted(component_edges, key=lambda root: component_edges[root], reverse=True)
    largest = components[0] if components else None
    return {
        "components": len(components),
        "largest_component": {
            "src": component_src[largest] if largest is not None else 0,
            "dst": component_dst[largest] if largest is not None else 0,
            "edges": component_edges[largest] if largest is not None else 0,
            "src_share": rate(component_src[largest], len(state.src_set)) if largest is not None else 0.0,
            "dst_share": rate(component_dst[largest], len(state.dst_set)) if largest is not None else 0.0,
            "edge_share": rate(component_edges[largest], sum(component_edges.values())) if largest is not None else 0.0,
        },
        "component_edge_qstats": qstats(component_edges.values()),
        "src_dst_id_overlap": len(state.src_set & state.dst_set),
        "src_dst_id_overlap_src_rate": rate(len(state.src_set & state.dst_set), len(state.src_set)),
        "src_dst_id_overlap_dst_rate": rate(len(state.src_set & state.dst_set), len(state.dst_set)),
    }


def sequence_behavior(events: list[Event], prefix_state: DataState, val_events: list[Event]) -> dict:
    by_src: dict[int, list[Event]] = defaultdict(list)
    last_pair_time: dict[tuple[int, int], int] = {}
    pair_repeat_gaps: list[int] = []
    for event in events:
        by_src[event.src].append(event)
        key = (event.src, event.dst)
        if key in last_pair_time:
            pair_repeat_gaps.append(event.time - last_pair_time[key])
        last_pair_time[key] = event.time

    inter_event_gaps = []
    for history in by_src.values():
        for prev, cur in zip(history, history[1:]):
            inter_event_gaps.append(cur.time - prev.time)

    holdout_history_lengths = []
    holdout_repeat_hits = Counter()
    for event in val_events:
        source = prefix_state.sources.get(event.src)
        if source is None:
            holdout_history_lengths.append(0)
            continue
        holdout_history_lengths.append(source.total)
        for k in RECENT_KS:
            if event.dst in set(list(source.recent_dsts)[-k:]):
                holdout_repeat_hits[f"recent_{k}_hit"] += 1
    return {
        "src_history_length": qstats(len(history) for history in by_src.values()),
        "src_inter_event_gap": qstats(inter_event_gaps),
        "pair_repeat_gap": qstats(pair_repeat_gaps),
        "holdout_history_length": qstats(holdout_history_lengths),
        "holdout_recent_hit_rates": {
            key: rate(value, len(val_events)) for key, value in holdout_repeat_hits.items()
        },
    }


def segmented_feature_effectiveness(
    state: DataState,
    val_events: list[Event],
    test_index: dict,
    rng: np.random.Generator,
    sample_size: int,
) -> dict:
    sampled = sample_events(val_events, sample_size, rng)
    src_degrees = np.asarray([source.total for source in state.sources.values()], dtype=np.float64)
    dst_degrees = np.asarray(list(state.dst_counts.values()), dtype=np.float64)
    src_p25, src_p75 = np.percentile(src_degrees, [25, 75])
    dst_p50, dst_p90 = np.percentile(dst_degrees, [50, 90])
    out = {}
    for proxy in ("random", "test_like"):
        queries = build_proxy_queries(sampled, state, test_index, rng, proxy=proxy)
        features = feature_tensor(state, queries)
        segment_masks = build_segment_masks(sampled, state, src_p25, src_p75, dst_p50, dst_p90)
        out[proxy] = {
            name: feature_metrics_for_mask(features, mask)
            for name, mask in segment_masks.items()
            if int(mask.sum()) >= 30
        }
    return out


def build_segment_masks(
    events: list[Event],
    state: DataState,
    src_p25: float,
    src_p75: float,
    dst_p50: float,
    dst_p90: float,
) -> dict[str, np.ndarray]:
    masks = defaultdict(list)
    for event in events:
        source = state.sources.get(event.src)
        dst_count = state.dst_counts.get(event.dst, 0)
        pair_seen = (event.src, event.dst) in state.pair_counts
        src_seen = source is not None
        dst_seen = event.dst in state.dst_counts
        values = {
            "all": True,
            "positive_pair_seen": pair_seen,
            "positive_known_nodes_new_pair": src_seen and dst_seen and not pair_seen,
            "positive_unseen_dst": not dst_seen,
            "high_src_degree": src_seen and source.total >= src_p75,
            "low_src_degree": (not src_seen) or source.total <= src_p25,
            "hot_dst": dst_seen and dst_count >= dst_p90,
            "tail_dst": (not dst_seen) or dst_count <= dst_p50,
        }
        for key, value in values.items():
            masks[key].append(bool(value))
    return {key: np.asarray(value, dtype=bool) for key, value in masks.items()}


def feature_metrics_for_mask(features: np.ndarray, mask: np.ndarray) -> dict:
    subset = features[mask]
    labels = np.zeros(subset.shape[:2], dtype=np.int8)
    labels[:, 0] = 1
    rows = []
    for idx, name in enumerate(FEATURE_NAMES):
        scores = subset[:, :, idx]
        rows.append(single_score_metrics(name, scores, labels))
    return {
        "queries": int(mask.sum()),
        "features": sorted(rows, key=lambda row: row["auc"], reverse=True),
    }


def graph_signal_proxy(
    state: DataState,
    val_events: list[Event],
    test_index: dict,
    rng: np.random.Generator,
    sample_size: int,
) -> dict:
    sampled = sample_events(val_events, sample_size, rng)
    out = {}
    for proxy in ("random", "test_like"):
        queries = build_proxy_queries(sampled, state, test_index, rng, proxy=proxy)
        scores = np.zeros((len(queries), len(queries[0].candidates)), dtype=np.float32)
        for row, query in enumerate(queries):
            for col, dst in enumerate(query.candidates):
                scores[row, col] = cooccur_score_from_recent(state, query.src, dst)
        labels = np.zeros(scores.shape, dtype=np.int8)
        labels[:, 0] = 1
        out[proxy] = single_score_metrics("recent16_cooccur", scores, labels)
    return out


def cooccur_score_from_recent(state: DataState, src: int, dst: int) -> float:
    source = state.sources.get(src)
    dst_sources = state.in_neighbors.get(dst)
    if source is None or not source.recent_dsts or not dst_sources:
        return 0.0
    total = 0.0
    seen_recent: set[int] = set()
    for rank, recent_dst in enumerate(reversed(source.recent_dsts), start=1):
        if rank > 16:
            break
        if recent_dst in seen_recent:
            continue
        seen_recent.add(recent_dst)
        recent_sources = state.in_neighbors.get(recent_dst)
        if not recent_sources:
            continue
        if len(recent_sources) < len(dst_sources):
            common = sum(1 for value in recent_sources if value in dst_sources)
        else:
            common = sum(1 for value in dst_sources if value in recent_sources)
        total += common / rank
    return math.log1p(total) / state.log_total_edges


def proxy_validation_calibration(
    state: DataState,
    val_events: list[Event],
    test_index: dict,
    rng: np.random.Generator,
    sample_size: int,
) -> dict:
    sampled = sample_events(val_events, sample_size, rng)
    out = {}
    for proxy in ("random", "test_like"):
        queries = build_proxy_queries(sampled, state, test_index, rng, proxy=proxy)
        features = feature_tensor(state, queries)
        labels = np.zeros(features.shape[:2], dtype=np.int8)
        labels[:, 0] = 1
        out[proxy] = {
            "queries": len(queries),
            "feature_metrics": [
                single_score_metrics(name, features[:, :, idx], labels)
                for idx, name in enumerate(FEATURE_NAMES)
            ],
        }
    return out


def fusion_permutation_importance(
    prefix_events: list[Event],
    val_events: list[Event],
    test_index: dict,
    rng: np.random.Generator,
    train_sample: int,
    val_sample: int,
) -> dict:
    if len(prefix_events) < 100:
        return {"status": "skipped"}
    context_end = max(1, int(len(prefix_events) * 0.75))
    context_events = prefix_events[:context_end]
    supervised_events = prefix_events[context_end:]
    context_state = build_state(context_events)
    full_state = build_state(prefix_events)
    train_events = sample_events(supervised_events, train_sample, rng)
    sampled_val = sample_events(val_events, val_sample, rng)
    train_queries = build_proxy_queries(train_events, context_state, test_index, rng, proxy="test_like")
    val_queries = build_proxy_queries(sampled_val, full_state, test_index, rng, proxy="test_like")
    train_features = feature_tensor(context_state, train_queries)
    val_features = feature_tensor(full_state, val_queries)
    train_x = train_features.reshape((-1, len(FEATURE_NAMES)))
    train_y = np.zeros(train_features.shape[:2], dtype=np.int8)
    train_y[:, 0] = 1
    train_y = train_y.ravel()
    val_y = np.zeros(val_features.shape[:2], dtype=np.int8)
    val_y[:, 0] = 1

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    model = LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs")
    model.fit(train_x, train_y)

    val_shape = val_features.shape
    val_x = scaler.transform(val_features.reshape((-1, len(FEATURE_NAMES))))
    base_scores = model.decision_function(val_x).reshape(val_shape[:2])
    base = ranking_metrics(base_scores, val_y)
    permutation = []
    for idx, name in enumerate(FEATURE_NAMES):
        shuffled = val_features.copy()
        flat = shuffled[:, :, idx].reshape(-1)
        flat[:] = flat[rng.permutation(flat.shape[0])]
        shuffled_x = scaler.transform(shuffled.reshape((-1, len(FEATURE_NAMES))))
        scores = model.decision_function(shuffled_x).reshape(val_shape[:2])
        metrics = ranking_metrics(scores, val_y)
        permutation.append(
            {
                "feature": name,
                "ap_drop": base["ap"] - metrics["ap"],
                "tie_mrr_drop": base["tie_mrr"] - metrics["tie_mrr"],
                "strict_hit1_drop": base["strict_hit1"] - metrics["strict_hit1"],
            }
        )
    coefficients = [
        {"feature": name, "coefficient": float(value)}
        for name, value in zip(FEATURE_NAMES, model.coef_[0])
    ]
    return {
        "status": "ok",
        "train_queries": len(train_queries),
        "val_queries": len(val_queries),
        "baseline": base,
        "coefficients": sorted(coefficients, key=lambda row: abs(row["coefficient"]), reverse=True),
        "permutation": sorted(permutation, key=lambda row: row["tie_mrr_drop"], reverse=True),
    }


def build_proxy_queries(
    events: list[Event],
    state: DataState,
    test_index: dict,
    rng: np.random.Generator,
    *,
    proxy: str,
    candidate_count: int = 100,
) -> list[Query]:
    dst_pool = np.asarray(sorted(state.dst_set), dtype=np.int64)
    global_candidates = test_index["global_candidates"]
    queries: list[Query] = []
    for event in events:
        negatives: list[int] = []
        used = {event.dst}
        if proxy == "test_like":
            source_rows = test_index["by_src"].get(event.src)
            if source_rows:
                row = source_rows[int(rng.integers(0, len(source_rows)))]
                for dst in row:
                    if dst in used:
                        continue
                    used.add(dst)
                    negatives.append(dst)
                    if len(negatives) >= candidate_count - 1:
                        break
            pool = global_candidates
        elif proxy == "random":
            pool = dst_pool
        else:
            raise ValueError(f"unknown proxy: {proxy}")

        attempts = 0
        while len(negatives) < candidate_count - 1 and attempts < 100:
            attempts += 1
            need = candidate_count - 1 - len(negatives)
            draw_size = max(need * 3, 128)
            sampled = rng.choice(pool, size=draw_size, replace=pool.size < draw_size)
            for raw_dst in sampled:
                dst = int(raw_dst)
                if dst in used:
                    continue
                used.add(dst)
                negatives.append(dst)
                if len(negatives) >= candidate_count - 1:
                    break
        if len(negatives) < candidate_count - 1:
            for raw_dst in dst_pool:
                dst = int(raw_dst)
                if dst in used:
                    continue
                used.add(dst)
                negatives.append(dst)
                if len(negatives) >= candidate_count - 1:
                    break
        if len(negatives) == candidate_count - 1:
            queries.append(Query(src=event.src, time=event.time, candidates=(event.dst, *negatives)))
    return queries


def feature_tensor(state: DataState, queries: list[Query]) -> np.ndarray:
    output = np.zeros((len(queries), len(queries[0].candidates), len(FEATURE_NAMES)), dtype=np.float32)
    for row, query in enumerate(queries):
        source = state.sources.get(query.src)
        if source is None:
            continue
        src_activity = math.log1p(source.total) / state.log_total_edges
        src_recency = decay(query.time - source.last_time, state.graph_span)
        for col, dst in enumerate(query.candidates):
            pair_count = source.dst_counts.get(dst, 0)
            if pair_count:
                output[row, col, 0] = math.log1p(pair_count)
                output[row, col, 1] = pair_count / source.total
                output[row, col, 2] = decay(query.time - source.dst_last_time[dst], state.graph_span)
            dst_count = state.dst_counts.get(dst, 0)
            if dst_count:
                output[row, col, 3] = math.log1p(dst_count) / state.log_total_edges
                output[row, col, 4] = decay(query.time - state.dst_last_time[dst], state.graph_span)
            rank = source.recent_ranks.get(dst)
            if rank is not None:
                output[row, col, 5] = 1.0 / rank
            output[row, col, 6] = src_activity
            output[row, col, 7] = src_recency
    return output


def single_score_metrics(name: str, scores: np.ndarray, labels: np.ndarray) -> dict:
    flat_y = labels.ravel()
    flat_scores = scores.ravel()
    try:
        auc = float(roc_auc_score(flat_y, flat_scores))
    except ValueError:
        auc = float("nan")
    metrics = ranking_metrics(scores, labels)
    pos = scores[:, 0]
    neg = scores[:, 1:].reshape(-1)
    return {
        "feature": name,
        "auc": auc,
        "ap": metrics["ap"],
        "tie_mrr": metrics["tie_mrr"],
        "strict_hit1": metrics["strict_hit1"],
        "pos_mean": float(pos.mean()),
        "neg_mean": float(neg.mean()),
        "pos_nonzero": float(np.mean(pos != 0)),
        "neg_nonzero": float(np.mean(neg != 0)),
    }


def ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    pos = scores[:, 0:1]
    greater = (scores[:, 1:] > pos).sum(axis=1)
    equal = (scores[:, 1:] == pos).sum(axis=1)
    tie_rank = 1.0 + greater + 0.5 * equal
    strict_hit1 = (greater == 0) & (equal == 0)
    return {
        "ap": float(average_precision_score(labels.ravel(), scores.ravel())),
        "tie_mrr": float(np.mean(1.0 / tie_rank)),
        "strict_hit1": float(np.mean(strict_hit1)),
        "median_tie_rank": float(np.median(tie_rank)),
    }


def sample_events(events: list[Event], sample_size: int, rng: np.random.Generator) -> list[Event]:
    if len(events) <= sample_size:
        return list(events)
    indices = np.sort(rng.choice(len(events), size=sample_size, replace=False))
    return [events[int(index)] for index in indices]


def decay(gap: int, span: int) -> float:
    return math.exp(-max(gap, 0) / max(span, 1))


def rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def qstats(values) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    p50, p75, p90, p95, p99 = np.percentile(arr, [50, 75, 90, 95, 99])
    return {
        "mean": float(arr.mean()),
        "median": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(arr.max()),
    }


def gini(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    arr.sort()
    n = arr.size
    return float((2.0 * np.arange(1, n + 1) @ arr) / (n * arr.sum()) - (n + 1) / n)


def _brief_summary(result: dict) -> dict:
    random_top = sorted(
        result["proxy_validation_calibration"]["random"]["feature_metrics"],
        key=lambda row: row["auc"],
        reverse=True,
    )[:3]
    test_top = sorted(
        result["proxy_validation_calibration"]["test_like"]["feature_metrics"],
        key=lambda row: row["auc"],
        reverse=True,
    )[:3]
    return {
        "dataset": result["dataset"],
        "duplicate_event_rate": result["basic"]["duplicate_event_rate"],
        "test_unseen_dst_rate": result["test_candidate_distribution"]["rates"]["unseen_dst_candidate_rate"],
        "components": result["graph_structure"]["components"],
        "largest_component_edge_share": result["graph_structure"]["largest_component"]["edge_share"],
        "random_top_features": random_top,
        "test_like_top_features": test_top,
        "fusion_baseline": result["fusion_permutation_importance"].get("baseline"),
        "seconds": result["seconds"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
