"""One-shot diagnostic: measure where the hybrid structure index spends memory.

Builds the StructureFeatureTower temporal index on a dataset's full train.csv
(future-only transition/cooccur mode, matching the champion dataset2 path) and
reports per-field deep memory usage plus process RSS, so we can confirm whether
the predict-stage peak is dominated by `future_cooccur_count_maps` / `pair_times`
rather than the LRU caches cleared by commit 6129342.

Usage:
    uv run python scripts/diagnose_index_memory.py --dataset dataset2 --cooccur-history-limit 32
"""

from __future__ import annotations

import argparse
import gc
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter

from jgrec.core.io import read_interactions
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex


def _rss_bytes() -> int:
    """Resident set size in bytes (Linux /proc, fallback to resource)."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    import resource  # noqa: PLC0415

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _fmt(num_bytes: float) -> str:
    gb = num_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{num_bytes / (1024**2):.1f} MB"


def _size_pair_times(d: dict) -> tuple[int, int]:
    total = sys.getsizeof(d)
    for key, arr in d.items():
        total += sys.getsizeof(key) + sys.getsizeof(key[0]) + sys.getsizeof(key[1])
        total += sys.getsizeof(arr) + int(arr.nbytes)
    return total, len(d)


def _size_int_array_map(d: dict) -> tuple[int, int]:
    total = sys.getsizeof(d)
    for key, arr in d.items():
        total += sys.getsizeof(key) + sys.getsizeof(arr) + int(getattr(arr, "nbytes", 0))
    return total, len(d)


def _size_count_maps(d: dict) -> tuple[int, int]:
    """dict[int, dict[int, int]] deep size (upper bound; counts shared ints each time)."""
    total = sys.getsizeof(d)
    inner_entries = 0
    for key, inner in d.items():
        total += sys.getsizeof(key) + sys.getsizeof(inner)
        for inner_key, inner_val in inner.items():
            total += sys.getsizeof(inner_key) + sys.getsizeof(inner_val)
            inner_entries += 1
    return total, inner_entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset2")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-fit-events", type=int, default=0, help="tail N events (0 = full history)")
    parser.add_argument("--cooccur-history-limit", type=int, default=32)
    parser.add_argument("--future-only", action="store_true", default=True)
    parser.add_argument("--no-future-only", dest="future_only", action="store_false")
    args = parser.parse_args()

    train_path = Path(args.data_dir) / args.dataset / "train.csv"
    print(f"[diag] dataset={args.dataset} train={train_path}")
    print(f"[diag] cooccur_history_limit={args.cooccur_history_limit} future_only={args.future_only}")

    rss0 = _rss_bytes()
    t0 = perf_counter()
    interactions = read_interactions(train_path)
    if args.max_fit_events > 0:
        interactions = interactions.tail(args.max_fit_events)
    rss_loaded = _rss_bytes()
    print(f"[diag] loaded edges={len(interactions)} in {perf_counter() - t0:.1f}s rss={_fmt(rss_loaded)}")

    gc.collect()
    tracemalloc.start()
    t1 = perf_counter()
    index = TemporalInteractionIndex()
    index.fit(
        interactions,
        build_transitions=True,
        build_cooccurs=True,
        cooccur_history_limit=args.cooccur_history_limit,
        future_only_transition_cooccur=args.future_only,
    )
    fit_secs = perf_counter() - t1
    _current, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del interactions
    gc.collect()
    rss_fit = _rss_bytes()

    print(f"\n[diag] index fit in {fit_secs:.1f}s")
    print(f"[diag] tracemalloc peak during fit = {_fmt(tm_peak)}")
    print(f"[diag] RSS: start={_fmt(rss0)} loaded={_fmt(rss_loaded)} after_fit={_fmt(rss_fit)}")
    print(f"[diag] RSS delta (fit) = {_fmt(rss_fit - rss_loaded)}\n")

    rows: list[tuple[str, int, int]] = []
    pt_bytes, pt_n = _size_pair_times(index.pair_times)
    rows.append(("pair_times", pt_bytes, pt_n))
    for name in ("src_dsts", "dst_srcs", "src_times", "dst_times"):
        size, n = _size_int_array_map(getattr(index, name))
        rows.append((name, size, n))
    fc_bytes, fc_n = _size_count_maps(index.future_cooccur_count_maps)
    rows.append(("future_cooccur_count_maps", fc_bytes, fc_n))
    ft_bytes, ft_n = _size_count_maps(index.future_transition_count_maps)
    rows.append(("future_transition_count_maps", ft_bytes, ft_n))
    ct_bytes, ct_n = _size_pair_times(index.cooccur_times) if index.cooccur_times else (0, 0)
    rows.append(("cooccur_times", ct_bytes, ct_n))
    tt_bytes, tt_n = _size_pair_times(index.transition_times) if index.transition_times else (0, 0)
    rows.append(("transition_times", tt_bytes, tt_n))

    rows.sort(key=lambda r: -r[1])
    print(f"{'field':<32}{'deep size':>14}{'entries':>16}")
    print("-" * 62)
    total = 0
    for name, size, n in rows:
        total += size
        print(f"{name:<32}{_fmt(size):>14}{n:>16,}")
    print("-" * 62)
    print(f"{'TOTAL (measured fields)':<32}{_fmt(total):>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
