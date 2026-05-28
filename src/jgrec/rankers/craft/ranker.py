from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import jittor as jt
import jittor_geometric
import numpy as np
from jittor_geometric.data import TemporalData
from jittor_geometric.nn.models.craft import CRAFT
from sklearn.metrics import average_precision_score, roc_auc_score

from jgrec.core.io import read_test_queries
from jgrec.core.types import FitContext, Interaction, TestQuery, TrainingReport
from jgrec.logging import log, track
from .config import CRAFTBaselineConfig


def _temporal_loader_api() -> tuple[type, Any]:
    root = Path(jittor_geometric.__file__).resolve().parent
    module_path = root / "dataloader" / "temporal_dataloader.py"
    spec = importlib.util.spec_from_file_location("_jgrec_temporal_dataloader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load temporal dataloader from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TemporalDataLoader, module.get_neighbor_sampler


class CRAFTBaselineRanker:
    name = "craft"

    def __init__(self, config: CRAFTBaselineConfig | None = None) -> None:
        self.config = config or CRAFTBaselineConfig()
        self.model: CRAFT | None = None
        self.neighbor_sampler = None
        self.num_neighbors = self.config.num_neighbors

    def fit(self, interactions: list[Interaction], context: FitContext) -> TrainingReport:
        if not interactions:
            raise ValueError("training interactions are empty")
        interactions = sorted(interactions, key=lambda item: item.time)
        src_np = np.asarray([item.src for item in interactions], dtype=np.int32)
        dst_np = np.asarray([item.dst for item in interactions], dtype=np.int32)
        time_np = np.asarray([item.time for item in interactions], dtype=np.int32)
        edge_ids_np = np.arange(len(interactions), dtype=np.int32) + 1
        test_candidates, test_src = _scan_test_nodes(context.dataset.test_path)

        num_total = len(interactions)
        num_val = max(1, int(num_total * self.config.val_ratio))
        num_train = max(1, num_total - num_val)
        if num_train >= num_total:
            num_train = num_total - 1

        train_data = TemporalData(
            src=jt.Var(src_np[:num_train]),
            dst=jt.Var(dst_np[:num_train]),
            t=jt.Var(time_np[:num_train]),
            edge_ids=jt.Var(edge_ids_np[:num_train]),
        )
        val_data = TemporalData(
            src=jt.Var(src_np[num_train:]),
            dst=jt.Var(dst_np[num_train:]),
            t=jt.Var(time_np[num_train:]),
            edge_ids=jt.Var(edge_ids_np[num_train:]),
        )
        full_data = TemporalData(
            src=jt.Var(src_np),
            dst=jt.Var(dst_np),
            t=jt.Var(time_np),
            edge_ids=jt.Var(edge_ids_np),
        )

        temporal_data_loader, get_neighbor_sampler = _temporal_loader_api()
        train_loader = temporal_data_loader(train_data, batch_size=self.config.batch_size, neg_sampling_ratio=1.0)
        val_loader = temporal_data_loader(val_data, batch_size=self.config.batch_size, neg_sampling_ratio=1.0)
        self.neighbor_sampler = get_neighbor_sampler(full_data, "recent", seed=context.seed)

        max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
        dst_min = min(int(dst_np.min()), int(test_candidates.min()))
        src_min = min(int(src_np.min()), int(test_src.min())) if test_src.size else int(src_np.min())
        self.model = CRAFT(
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            hidden_size=self.config.hidden_size,
            hidden_dropout_prob=self.config.dropout,
            attn_dropout_prob=self.config.dropout,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            initializer_range=0.02,
            n_nodes=max_node + 1,
            max_seq_length=self.config.num_neighbors,
            loss_type="BPR",
            use_pos=True,
            input_cat_time_intervals=False,
            output_cat_time_intervals=True,
            output_cat_repeat_times=True,
            num_output_layer=1,
            emb_dropout_prob=self.config.dropout,
            skip_connection=True,
        )
        self.model.set_min_idx(src_min, dst_min)
        optimizer = jt.nn.Adam(list(self.model.parameters()), lr=self.config.lr)

        best_ap = 0.0
        best_auc = 0.0
        best_state = _snapshot_state(self.model)
        patience_counter = 0
        epochs = range(1, self.config.epochs + 1)
        for epoch in track(epochs, description="craft", total=self.config.epochs, enabled=context.verbose):
            self.model.train()
            losses: list[float] = []
            for _, batch_data in enumerate(train_loader):
                src = jt.array(batch_data.src)
                dst = jt.array(batch_data.dst)
                times = jt.array(batch_data.t)
                neg_dst = jt.array(batch_data.neg_dst)

                src_neighb_seq, _, src_neighb_times = self.neighbor_sampler.get_historical_neighbors_left(
                    node_ids=src.numpy(),
                    node_interact_times=times.numpy(),
                    num_neighbors=self.config.num_neighbors,
                )
                neighbor_num = (src_neighb_seq != 0).sum(axis=1)
                if neighbor_num.sum() == 0:
                    continue

                test_dst = jt.cat([jt.Var(dst).unsqueeze(-1), jt.Var(neg_dst).unsqueeze(-1)], dim=-1)
                dst_last_update_time = self._dst_last_update_times(test_dst, times.numpy())
                loss, _, _ = self.model.calculate_loss(
                    src_neighb_seq=jt.Var(src_neighb_seq),
                    src_neighb_seq_len=jt.Var(neighbor_num),
                    src_neighb_interact_times=jt.Var(src_neighb_times),
                    cur_pred_times=jt.Var(times),
                    test_dst=test_dst,
                    dst_last_update_times=dst_last_update_time,
                )
                optimizer.zero_grad()
                optimizer.step(loss)
                jt.sync_all()
                losses.append(float(loss.item()))

            val_metrics = self._validate(val_loader)
            val_ap = val_metrics["AP"]
            if val_ap > best_ap:
                best_ap = val_ap
                best_auc = val_metrics["AUC"]
                best_state = _snapshot_state(self.model)
                patience_counter = 0
            else:
                patience_counter += 1
            mean_loss = float(np.mean(losses)) if losses else 0.0
            log(
                f"[craft] epoch={epoch} loss={mean_loss:.5f} "
                f"val_ap={val_ap:.5f} val_auc={val_metrics['AUC']:.5f} "
                f"best_ap={best_ap:.5f} patience={patience_counter}",
                enabled=context.verbose,
            )
            if self.config.early_stop_patience > 0 and patience_counter >= self.config.early_stop_patience:
                log(f"[craft] early_stop epoch={epoch}", enabled=context.verbose)
                break

        _load_state(self.model, best_state)
        return TrainingReport(
            train_events=num_train,
            val_events=num_total - num_train,
            best_val_ap=best_ap,
            best_val_mrr=0.0,
            selected_fusion="craft",
            feature_names=("craft_score",),
            model_name=self.name,
            metrics={"auc": best_auc},
        )

    def predict_batch(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 100), dtype=np.float64)
        if self.model is None or self.neighbor_sampler is None:
            raise RuntimeError("ranker is not fitted")

        batch_src = np.asarray([query.src for query in queries], dtype=np.int32)
        batch_time = np.asarray([query.time for query in queries], dtype=np.int32)
        batch_candidates = np.asarray([query.candidates for query in queries], dtype=np.int32)

        src_neighb_seq, _, src_neighb_times = self.neighbor_sampler.get_historical_neighbors_left(
            node_ids=batch_src,
            node_interact_times=batch_time,
            num_neighbors=self.config.num_neighbors,
        )
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)
        test_dst = jt.Var(batch_candidates)
        dst_last_update_time = self._dst_last_update_times(test_dst, batch_time)

        src_neighb_seq_adj = jt.Var(src_neighb_seq) - self.model.dst_min_idx + 1
        test_dst_adj = test_dst - self.model.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)

        with jt.no_grad():
            logits = self.model.forward(
                src_neighb_seq_adj,
                jt.Var(neighbor_num),
                jt.Var(src_neighb_times),
                jt.Var(batch_time),
                test_dst=test_dst_adj,
                dst_last_update_times=dst_last_update_time,
            )
            probs = jt.sigmoid(logits.squeeze(-1)).numpy()
        return np.asarray(probs, dtype=np.float64)

    def _validate(self, loader: Any) -> dict[str, float]:
        if self.model is None or self.neighbor_sampler is None:
            raise RuntimeError("ranker is not fitted")
        self.model.eval()
        ap_list: list[float] = []
        auc_list: list[float] = []
        for _, batch_data in enumerate(loader):
            src = jt.array(batch_data.src)
            dst = jt.array(batch_data.dst)
            times = jt.array(batch_data.t)
            neg_dst = jt.array(batch_data.neg_dst)

            src_neighb_seq, _, src_neighb_times = self.neighbor_sampler.get_historical_neighbors_left(
                node_ids=src.numpy(),
                node_interact_times=times.numpy(),
                num_neighbors=self.config.num_neighbors,
            )
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)
            test_dst = jt.cat([jt.Var(dst).unsqueeze(1), jt.Var(neg_dst).unsqueeze(1)], dim=1)
            dst_last_update_time = self._dst_last_update_times(test_dst, times.numpy())
            pos_score, neg_score = self.model.predict(
                src_neighb_seq=jt.Var(src_neighb_seq),
                src_neighb_seq_len=jt.Var(neighbor_num),
                src_neighb_interact_times=jt.Var(src_neighb_times),
                cur_pred_times=jt.Var(times),
                test_dst=test_dst,
                dst_last_update_times=dst_last_update_time,
            )
            y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
            y_score = np.concatenate([pos_score, neg_score.flatten()])
            ap_list.append(float(average_precision_score(y_true, y_score)))
            auc_list.append(float(roc_auc_score(y_true, y_score)))
        return {
            "AP": float(np.mean(ap_list)) if ap_list else 0.0,
            "AUC": float(np.mean(auc_list)) if auc_list else 0.0,
        }

    def _dst_last_update_times(self, test_dst: jt.Var, times: np.ndarray) -> jt.Var:
        if self.neighbor_sampler is None:
            raise RuntimeError("ranker is not fitted")
        dst_last_neighbor, _, dst_last_update_time = self.neighbor_sampler.get_historical_neighbors_left(
            node_ids=test_dst.flatten().numpy(),
            node_interact_times=np.broadcast_to(times[:, np.newaxis], (len(times), test_dst.shape[1])).flatten(),
            num_neighbors=1,
        )
        update_time = np.asarray(dst_last_update_time).reshape(len(test_dst), -1)
        update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000
        return jt.Var(update_time)


def _scan_test_nodes(test_path) -> tuple[np.ndarray, np.ndarray]:
    candidates: list[tuple[int, ...]] = []
    sources: list[int] = []
    for query in read_test_queries(test_path):
        candidates.append(query.candidates)
        sources.append(query.src)
    if not candidates:
        raise ValueError(f"no test queries loaded from {test_path}")
    return np.asarray(candidates, dtype=np.int32), np.asarray(sources, dtype=np.int32)


def _snapshot_state(model: CRAFT) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value.numpy(), dtype=np.float32).copy()
        for key, value in model.state_dict().items()
    }


def _load_state(model: CRAFT, state: dict[str, np.ndarray]) -> None:
    model.load_state_dict({key: jt.array(value, dtype=jt.float32) for key, value in state.items()})
