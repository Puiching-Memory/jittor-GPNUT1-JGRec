from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import optuna

from jgrec.core.io import read_interactions
from jgrec.core.types import DatasetPaths, FitContext


def main() -> int:
    args = _parse_args()
    device_label = _configure_device(args)
    worker_id = _worker_id(args)
    sampler_seed = args.sampler_seed if args.sampler_seed is not None else args.seed + worker_id * 9973
    output_dir = args.output_dir / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{output_dir / 'study.db'}"
    study = _create_or_load_study(args, output_dir, storage_url, sampler_seed)
    datasets = _discover_selected_datasets(args.data_dir, args.datasets)
    if args.n_jobs > 1 and not args.cpu:
        print(
            "warning: --n-jobs > 1 shares one Python/Jittor process; prefer one process per GPU "
            "with --n-jobs 1 and the same --study-name."
        )

    def objective(trial: optuna.Trial) -> float:
        from jgrec.rankers.temporal_graph.ranker import TemporalGraphRanker  # noqa: PLC0415

        config = _suggest_config(trial, args)
        started = time.perf_counter()
        total = 0.0
        metrics: dict[str, float] = {}
        for dataset_idx, dataset in enumerate(datasets, start=1):
            ranker = TemporalGraphRanker()
            interactions = read_interactions(dataset.train_path)
            report = ranker.fit(
                interactions,
                training_config=config,
                context=FitContext(dataset=dataset, seed=args.seed + trial.number, verbose=not args.quiet),
            )
            prefix = dataset.name
            metrics[f"{prefix}_ap"] = report.best_val_ap
            metrics[f"{prefix}_mrr"] = report.best_val_mrr
            metrics[f"{prefix}_best_epoch"] = report.metrics.get("best_epoch", 0.0)
            total += _score(report.best_val_ap, report.best_val_mrr, args.objective_metric)
            trial.report(total / dataset_idx, step=dataset_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        elapsed = time.perf_counter() - started
        objective_value = total / len(datasets)
        payload = {
            "trial": trial.number,
            "value": objective_value,
            "elapsed_sec": elapsed,
            "device": device_label,
            "worker_id": worker_id,
            "sampler_seed": sampler_seed,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "config": asdict(config),
            "metrics": metrics,
        }
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("device", device_label)
        trial.set_user_attr("worker_id", worker_id)
        trial.set_user_attr("sampler_seed", sampler_seed)
        trial.set_user_attr("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        trial.set_user_attr("config", asdict(config))
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        _append_jsonl(output_dir / "trials.jsonl", payload)
        _write_best(study, output_dir / "best.json")
        return objective_value

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, gc_after_trial=True)
    _write_best(study, output_dir / "best.json")
    try:
        best_trial = study.best_trial
    except ValueError:
        print("best_value unavailable: no completed trials")
        print(f"study={storage_url}")
        return 1
    print(f"best_value={best_trial.value:.6f}")
    print(json.dumps(best_trial.params, ensure_ascii=False, sort_keys=True))
    if "config" in best_trial.user_attrs:
        print("best_config=" + json.dumps(best_trial.user_attrs["config"], ensure_ascii=False, sort_keys=True))
    print(f"study={storage_url}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune temporal-graph hyperparameters with Optuna.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=["dataset1", "dataset2"])
    parser.add_argument("--study-name", default="temporal_graph_mrr")
    parser.add_argument("--output-dir", type=Path, default=Path("result/optuna"))
    parser.add_argument("--n-trials", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--sampler-seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Rejected for temporal-graph; CUDA is required.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective-metric", choices=["mrr", "ap", "sum"], default="mrr")
    parser.add_argument("--max-fit-events", type=int, default=0)
    parser.add_argument("--max-train-events", type=int, default=20_000)
    parser.add_argument("--max-val-events", type=int, default=5_000)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument("--validation-candidates", choices=["random", "test_like"], default="test_like")
    parser.add_argument(
        "--candidate-recent-feature-group",
        choices=["none", "recency_rank"],
        default="recency_rank",
    )
    parser.add_argument("--epochs-max", type=int, default=6)
    parser.add_argument("--sqlite-timeout", type=float, default=120.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _configure_device(args: argparse.Namespace) -> str:
    if args.cpu:
        raise ValueError("temporal-graph tuning requires CUDA; do not pass --cpu")
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        return f"cuda:{args.gpu_id}"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return f"cuda:{visible}"
    return "cuda"


def _worker_id(args: argparse.Namespace) -> int:
    if args.worker_id is not None:
        return args.worker_id
    if args.gpu_id is not None:
        return args.gpu_id
    return 0


def _create_or_load_study(
    args: argparse.Namespace,
    output_dir: Path,
    storage_url: str,
    sampler_seed: int,
) -> optuna.Study:
    lock_path = output_dir / ".study-init.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            storage = optuna.storages.RDBStorage(
                url=storage_url,
                engine_kwargs={"connect_args": {"timeout": args.sqlite_timeout}},
            )
            return optuna.create_study(
                study_name=args.study_name,
                storage=storage,
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=sampler_seed, multivariate=True),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=max(4, args.n_jobs)),
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _discover_selected_datasets(data_dir: Path, names: list[str]) -> list[DatasetPaths]:
    datasets: list[DatasetPaths] = []
    for name in names:
        root = data_dir / name
        train_path = root / "train.csv"
        test_path = root / "test.csv"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"missing train/test files for {name} under {data_dir}")
        datasets.append(DatasetPaths(name=name, root=root, train_path=train_path, test_path=test_path))
    return datasets


def _suggest_config(trial: optuna.Trial, args: argparse.Namespace):
    from jgrec.rankers.temporal_graph.config import TemporalGraphTrainingConfig  # noqa: PLC0415

    hidden_size = trial.suggest_categorical("hidden_size", [64, 96, 128, 192])
    possible_heads = [head for head in [2, 3, 4, 6, 8] if hidden_size % head == 0]
    heads = trial.suggest_categorical(f"heads_h{hidden_size}", possible_heads)
    return TemporalGraphTrainingConfig(
        val_ratio=0.15,
        max_train_events=args.max_train_events,
        max_val_events=args.max_val_events,
        num_negatives=args.num_negatives,
        max_fit_events=args.max_fit_events,
        epochs=trial.suggest_int("epochs", 2, args.epochs_max),
        train_batch_size=trial.suggest_categorical("train_batch_size", [128, 256, 384, 512]),
        lr=trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-7, 3e-3, log=True),
        selection_metric=trial.suggest_categorical("selection_metric", ["mrr", "ap"]),
        early_stop_patience=trial.suggest_int("early_stop_patience", 2, 5),
        seed=args.seed + trial.number,
        verbose=not args.quiet,
        history_len=trial.suggest_categorical("history_len", [16, 32, 64, 96]),
        candidate_history_len=trial.suggest_categorical("candidate_history_len", [8, 16, 32, 48]),
        hidden_size=hidden_size,
        layers=trial.suggest_int("layers", 1, 4),
        heads=heads,
        dropout=trial.suggest_float("dropout", 0.05, 0.45),
        validation_candidates=args.validation_candidates,
        candidate_recent_feature_group=args.candidate_recent_feature_group,
        refit_full=False,
    )


def _score(ap: float, mrr: float, metric: str) -> float:
    if metric == "ap":
        return ap
    if metric == "mrr":
        return mrr
    return ap + mrr


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _write_best(study: optuna.Study, path: Path) -> None:
    if not study.trials:
        return
    try:
        trial = study.best_trial
    except ValueError:
        return
    payload = {
        "trial": trial.number,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
    }
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
