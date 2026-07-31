from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.rankers.hybrid.confidence_routed_topk_id import (
    correction_improvement_labels,
)
from jgrec.rankers.hybrid.oof_stacking import tie_neutral_mrr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose the frozen strict-temporal correction route.",
    )
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--sequence-cache-dir", required=True, type=Path)
    parser.add_argument("--base-result-dir", required=True, type=Path)
    parser.add_argument(
        "--validation-expert-logits",
        required=True,
        type=Path,
    )
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = _read_json(args.sequence_cache_dir / "fold-manifest.json")
    train_prefix = str(args.train_cache_prefix)
    train_times = np.load(
        f"{train_prefix}.train-time.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    rows = []
    for fold in manifest["folds"]:
        index = int(fold["index"])
        start, stop = (int(value) for value in fold["score_rows"])
        base = np.load(
            args.base_result_dir
            / "folds"
            / "variant-A"
            / f"fold-{index}"
            / "score-logits.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        model_dir = (
            args.experiment_dir
            / "models"
            / "strict-temporal-support-top10-route05"
            / f"fold-{index}"
        )
        rows.append(
            _diagnose(
                f"fold-{index}",
                base,
                np.load(
                    model_dir / "score-proposal.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                np.load(
                    model_dir / "score-route-probabilities.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                np.load(
                    args.experiment_dir
                    / "folds"
                    / "strict-temporal-support-top10-route05"
                    / f"fold-{index}"
                    / "route-mask.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                np.asarray(train_times[start:stop]),
            )
        )

    validation_prefix = str(args.validation_cache_prefix)
    validation_times = np.load(
        f"{validation_prefix}.val-time.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_experts = np.load(
        args.validation_expert_logits,
        mmap_mode="r",
        allow_pickle=False,
    )
    full_model_dir = (
        args.experiment_dir
        / "models"
        / "strict-temporal-support-top10-route05"
        / "full"
    )
    rows.append(
        _diagnose(
            "external",
            validation_experts[0],
            np.load(
                full_model_dir / "score-proposal.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            np.load(
                full_model_dir / "score-route-probabilities.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            np.load(
                args.experiment_dir / "full" / "route-mask.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            np.asarray(validation_times),
        )
    )
    report = {
        "status": "complete",
        "diagnostic_only": True,
        "frozen_outputs_unchanged": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _diagnose(
    name: str,
    base_scores: Any,
    proposal_scores: Any,
    probabilities: Any,
    route_mask: Any,
    times: np.ndarray,
) -> dict[str, Any]:
    base = np.asarray(base_scores)
    proposal = np.asarray(proposal_scores)
    probability = np.asarray(probabilities)
    route = np.asarray(route_mask, dtype=bool)
    positives = np.zeros(base.shape[0], dtype=np.int32)
    labels, rewards = correction_improvement_labels(
        base,
        proposal,
        positives,
    )
    order = np.argsort(-probability, kind="stable")
    fractions = {}
    for fraction in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05):
        count = max(1, int(np.floor(base.shape[0] * fraction)))
        selected = order[:count]
        fractions[f"{fraction:.4f}"] = {
            "rows": count,
            "improved": int(np.sum(labels[selected] > 0.0)),
            "harmed": int(np.sum(rewards[selected] < 0.0)),
            "neutral": int(np.sum(rewards[selected] == 0.0)),
            "full_mrr_delta_if_routed": float(
                np.sum(rewards[selected]) / base.shape[0]
            ),
            "selected_mean_reward": float(np.mean(rewards[selected])),
            "minimum_probability": float(
                np.min(probability[selected])
            ),
        }
    boundaries = np.linspace(0, base.shape[0], 4, dtype=np.int64)
    time_slices = {}
    for index in range(3):
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        selected = route[start:stop]
        time_slices[str(index)] = {
            "rows": stop - start,
            "routed_rows": int(np.sum(selected)),
            "routed_reward_sum": float(
                np.sum(rewards[start:stop][selected])
            ),
            "full_mrr_delta": float(
                np.sum(rewards[start:stop][selected]) / (stop - start)
            ),
        }
    routed_rewards = rewards[route]
    return {
        "name": name,
        "rows": int(base.shape[0]),
        "time_min": int(np.min(times)),
        "time_max": int(np.max(times)),
        "base_mrr": tie_neutral_mrr(base, positives),
        "ungated_proposal_mrr": tie_neutral_mrr(proposal, positives),
        "ungated_proposal_delta": float(np.mean(rewards)),
        "proposal_positive_fraction": float(np.mean(labels)),
        "proposal_negative_fraction": float(np.mean(rewards < 0.0)),
        "routed": {
            "rows": int(np.sum(route)),
            "improved": int(np.sum(routed_rewards > 0.0)),
            "harmed": int(np.sum(routed_rewards < 0.0)),
            "neutral": int(np.sum(routed_rewards == 0.0)),
            "mean_reward": (
                float(np.mean(routed_rewards))
                if routed_rewards.size
                else 0.0
            ),
            "full_mrr_delta": float(
                np.sum(routed_rewards) / base.shape[0]
            ),
        },
        "top_probability_fractions": fractions,
        "time_slices": time_slices,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
