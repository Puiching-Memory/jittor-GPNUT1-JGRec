# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

第六届计图人工智能挑战赛赛道一动态推荐项目。给定历史时序交互 `(src, dst, time)`
和测试候选集合 `(src, time, c1...c100)`，为每行 100 个候选输出交互概率分布。线上评分指标是
MRR。主线模型是 `hybrid`。

## Environment & Commands

This project uses **uv** for all Python/env/tool execution (targets Python 3.12). Never call
`pip`/`python` directly — always go through `uv run`. Jittor and jittor-geometric are editable
deps under `third_party/` (git submodules).

```bash
uv sync                       # create .venv and install all deps (incl. third_party editables)
uv run jgrec-build            # generate full submission (all datasets) -> result/<run_id>/result.zip
```

Smoke test (fast, exercises full pipeline with tiny budgets):

```bash
uv run jgrec-build --limit-rows 2 --max-fit-events 512 --max-train-events 32 --max-val-events 16 \
  --num-negatives 3 --epochs 1 --history-len 8 --candidate-history-len 4 --hidden-size 32 \
  --layers 1 --heads 2 --no-refit-full --quiet-ranker
```

Checks (all dev tools are in the `dev` dependency group):

```bash
uv run python -m compileall -q src scripts tests
uv run --group dev pytest                      # config in pyproject; coverage on jgrec is enforced
uv run --group dev pytest tests/test_cli.py -k some_test   # single test
uv run --group dev ruff check .                # lint (line-length 120, E501 ignored)
uv run --group dev ruff format .               # format (double quotes, lf)
uv lock --check                                # verify lockfile
uv run zensical build                          # build docs (only if docs changed)
```

When running backend tests, enforce a hard 60s timeout to avoid stuck tasks.

### Local Jittor / WSL

Jittor needs CUDA. `sitecustomize.py` auto-builds a CUDA 12.6 overlay (symlinking the pip
`nvidia-cudnn-cu12` into a discovered system CUDA toolkit) and sets `nvcc_path`/`CUDA_HOME` at
import time — do not hardcode CUDA paths elsewhere. To run from source without uv:

```bash
export PYTHONPATH=src
/mnt/d/work/jittor-local/env/bin/python -m jgrec.cli --help
```

Use `--cpu` to force CPU.

## Architecture

Every model implements the same `Ranker` protocol (`src/jgrec/rankers/base.py`):

```python
ranker.fit(interactions, FitContext) -> TrainingReport
ranker.predict_batch(queries) -> np.ndarray   # shape == (batch, 100)
```

`core/runner.build_dataset_submission()` is the only orchestration path and depends solely on this
protocol — it reads train CSV, calls `fit`, streams `predict_batch` to `<dataset>.csv`, and never
imports a concrete model. A **fresh ranker instance is created per dataset** (see `cli.py` loop) to
avoid cross-dataset state leakage. `submission.py` handles CSV/ZIP writing + format validation only.

Layers:
- `core/io.py` — dataset discovery (`data/<dataset>/{train,test}.csv`), CSV reading, row counting. **The core pipeline does not use pandas** — keep it that way.
- `core/types.py` — `Interaction` / `TestQuery` / `FitContext` / `TrainingReport` / `DatasetPaths`.
- `core/memory.py` — phase-level memory + progress logging (writes `result/<run_id>/memory.log`).
- `rankers/registry.py` — lazy factories for `hybrid`, `craft`, `temporal-graph`.
- `cli.py` — single `tyro`-based `CLIConfig` dataclass (one flat flag namespace); `_ranker_config()` maps flags to per-model config dataclasses.

### Models

| `--model`        | Notes |
| ---------------- | ----- |
| `hybrid` (default) | The competition model; see below. |
| `craft`          | Adapter around the official CRAFT baseline (`rankers/craft/`). |
| `temporal-graph` | Temporal graph backend (`rankers/temporal_graph/`); has its own tuning entry `jgrec-tune-temporal-graph`. |

### Hybrid model (`rankers/hybrid/`)

Time-causally split training fits an MLP **fusion** head over candidate-level features. Feature
order is fixed:

```
stats + candidate_prior + structure + two_tower + graph + sequence
```

- `stats.py` — causal statistics: history repeat, recent hits, target popularity, source activity.
- `candidate_prior.py` — uses only `test.csv` candidate frequency + within-row rank (**no labels**); supplies signal for unseen target nodes.
- `structure.py` — common-neighbor, Jaccard, co-occurrence, transition features.
- `two_tower.py` — Jittor two-tower candidate representation (dot/cosine features).
- `gnn.py` — XSimGCL / LightGCN graph CF tower.
- `sequence.py` — SASRec / GRU sequence-preference tower.
- `fusion.py` — Jittor MLP; selects the final feature group from several feature masks by validation score.
- `auto_strategy.py` — profiles the data as `repeat_memory` / `balanced` / `new_link_cold` from `train.csv` + unlabeled `test.csv` candidates (**never reads dataset names**), then auto-tunes the test-candidate negative-sampling ratio.

To add a model: implement `Ranker` under `rankers/<name>/`, register a lazy factory in
`registry.py`, add any flags to `cli.py`. Runner/submission/zip need no changes as long as
`predict_batch` returns `(batch, 100)`.

## Submission contract

- `result.zip` root contains `dataset1.csv`, `dataset2.csv` directly — no extra directory level.
- No CSV header; each row maps to the test row's 100 candidates in order; each probability has 8 decimals.
- Only upload `result.zip`. **Never commit `data/`, `result/`, `site/`, logs, or `*.zip`** (see `.gitignore`).

## Tuning notes

- `--max-fit-events 0` keeps the final encoder on full training history.
- Do not disable `structure` / `two_tower` / `candidate_prior` just to speed up — they are key to dataset2 gains.
- `--selection-metric mrr` aligns local selection with the online MRR metric (default is `ap`).
- `--test-candidate-negative-ratio` calibrates cold-start / new-link negative distribution.
- On the local 8GB-VRAM / 24GB-RAM box, re-run `dataset2` alone first, then combine with a stable `dataset1.csv`.

## Conventions

- Reply to the user in Chinese by default.
- No silent fallbacks / mocks / defensive guardrails added just to make things run — let failures surface. Boundary rules only when explicitly agreed.
- Hard limits: functions ≤50 lines, files ≤300 lines, nesting ≤3, ≤3 positional params, no magic numbers.
- ast-grep rules live in `rules/` (`sgconfig.yml`); docs live in `docs/` (zensical, config `zensical.toml`).
