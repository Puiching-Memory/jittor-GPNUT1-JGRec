from __future__ import annotations

from dataclasses import dataclass

import jittor as jt
import numpy as np
from jittor_geometric.nn.models import LightGCN, XSimGCL

from .data import Interaction, TestQuery
from .idmap import NodeIdMap
from .logging import log, track


GRAPH_WINDOW_NAMES = ("gnn_full", "gnn_recent", "gnn_short")
GRAPH_WINDOW_FRACTIONS = (1.0, 0.35, 0.10)


@dataclass(frozen=True)
class GraphTowerConfig:
    enabled: bool = True
    model_name: str = "xsimgcl"
    embedding_dim: int = 128
    layers: int = 2
    epochs: int = 3
    batch_size: int = 8192
    max_graph_edges: int = 0
    max_train_edges: int = 200_000
    lr: float = 1e-3
    weight_decay: float = 0.0
    reg_weight: float = 1e-5
    cl_rate: float = 1e-4
    temperature: float = 0.05
    eps: float = 0.1


class GraphTower:
    def __init__(self, id_map: NodeIdMap, config: GraphTowerConfig) -> None:
        self.id_map = id_map
        self.config = config
        self.user_embeddings: dict[str, np.ndarray] = {}
        self.item_embeddings: dict[str, np.ndarray] = {}
        self.seen_users: dict[str, np.ndarray] = {}
        self.seen_items: dict[str, np.ndarray] = {}

    @property
    def feature_names(self) -> tuple[str, ...]:
        return GRAPH_WINDOW_NAMES

    def fit(self, interactions: list[Interaction], rng: np.random.Generator, verbose: bool = True) -> None:
        if not self.config.enabled or self.config.epochs < 1:
            return
        if self.id_map.num_src == 0 or self.id_map.num_dst == 0:
            return

        mapped_edges = _mapped_edges(interactions, self.id_map)
        if mapped_edges.shape[1] == 0:
            return

        total_edges = mapped_edges.shape[1]
        for name, fraction in zip(GRAPH_WINDOW_NAMES, GRAPH_WINDOW_FRACTIONS):
            edge_count = max(1, int(total_edges * fraction))
            edge_index = mapped_edges[:, total_edges - edge_count :]
            log(f"[gnn:{name}] train_edges={edge_index.shape[1]} model={self.config.model_name}", enabled=verbose)
            self._fit_one_window(name, edge_index, rng, verbose)

    def scores_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(GRAPH_WINDOW_NAMES)), dtype=np.float32)

        candidate_count = len(queries[0].candidates)
        scores = np.zeros((len(queries), candidate_count, len(GRAPH_WINDOW_NAMES)), dtype=np.float32)
        for feature_idx, name in enumerate(GRAPH_WINDOW_NAMES):
            user_emb = self.user_embeddings.get(name)
            item_emb = self.item_embeddings.get(name)
            if user_emb is None or item_emb is None:
                continue

            for row_idx, query in enumerate(queries):
                src_id = self.id_map.src_id(query.src)
                seen_user = self.seen_users.get(name)
                seen_item = self.seen_items.get(name)
                if src_id < 0 or seen_user is None or seen_item is None or not seen_user[src_id]:
                    continue
                dst_ids = self.id_map.dst_ids(query.candidates)
                valid = (dst_ids >= 0) & seen_item[dst_ids.clip(min=0)]
                if not np.any(valid):
                    continue
                scores[row_idx, valid, feature_idx] = item_emb[dst_ids[valid]] @ user_emb[src_id]
        return scores

    def _fit_one_window(
        self,
        name: str,
        edge_index: np.ndarray,
        rng: np.random.Generator,
        verbose: bool,
    ) -> None:
        if self.config.max_graph_edges > 0 and edge_index.shape[1] > self.config.max_graph_edges:
            edge_index = edge_index[:, -self.config.max_graph_edges :]
        seen_users = np.zeros(self.id_map.num_src, dtype=bool)
        seen_items = np.zeros(self.id_map.num_dst, dtype=bool)
        seen_users[np.unique(edge_index[0])] = True
        seen_items[np.unique(edge_index[1])] = True
        self.seen_users[name] = seen_users
        self.seen_items[name] = seen_items
        model = self._build_model(edge_index)
        optimizer = jt.nn.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        users = edge_index[0].astype(np.int32, copy=False)
        pos_items = edge_index[1].astype(np.int32, copy=False)
        train_size = users.shape[0]
        max_edges = train_size if self.config.max_train_edges <= 0 else min(train_size, self.config.max_train_edges)

        epochs = range(1, self.config.epochs + 1)
        for epoch in track(epochs, description=f"gnn:{name}", total=self.config.epochs, enabled=verbose):
            if max_edges < train_size:
                order = rng.choice(train_size, size=max_edges, replace=False)
                rng.shuffle(order)
            else:
                order = rng.permutation(train_size)

            losses: list[float] = []
            for start in range(0, order.shape[0], self.config.batch_size):
                batch_idx = order[start : start + self.config.batch_size]
                batch_users = users[batch_idx]
                batch_pos = pos_items[batch_idx]
                batch_neg = rng.integers(0, self.id_map.num_dst, size=batch_idx.shape[0], dtype=np.int32)
                same = batch_neg == batch_pos
                if np.any(same):
                    batch_neg[same] = (batch_neg[same] + 1) % self.id_map.num_dst

                loss = model(
                    jt.array(batch_users, dtype=jt.int32),
                    jt.array(batch_pos, dtype=jt.int32),
                    jt.array(batch_neg, dtype=jt.int32),
                )
                optimizer.step(loss)
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses)) if losses else 0.0
            log(f"[gnn:{name}] epoch={epoch} loss={mean_loss:.5f}", enabled=verbose)

        with jt.no_grad():
            user_all, item_all = model.get_all_embeddings()
            self.user_embeddings[name] = np.asarray(user_all.numpy(), dtype=np.float32)
            self.item_embeddings[name] = np.asarray(item_all.numpy(), dtype=np.float32)

    def _build_model(self, edge_index: np.ndarray):
        edge_var = jt.array(edge_index, dtype=jt.int32)
        model_name = self.config.model_name.lower()
        if model_name == "lightgcn":
            return LightGCN(
                self.id_map.num_src,
                self.id_map.num_dst,
                self.config.embedding_dim,
                self.config.layers,
                edge_var,
                reg_weight=self.config.reg_weight,
            )
        if model_name == "xsimgcl":
            return XSimGCL(
                self.id_map.num_src,
                self.id_map.num_dst,
                self.config.embedding_dim,
                self.config.layers,
                edge_var,
                reg_weight=self.config.reg_weight,
                cl_rate=self.config.cl_rate,
                temperature=self.config.temperature,
                eps=self.config.eps,
                layer_cl=max(1, min(self.config.layers, self.config.layers)),
            )
        raise ValueError(f"unsupported graph model: {self.config.model_name}")


def _mapped_edges(interactions: list[Interaction], id_map: NodeIdMap) -> np.ndarray:
    edges = np.empty((2, len(interactions)), dtype=np.int32)
    kept = 0
    for item in interactions:
        src_id = id_map.src_id(item.src)
        dst_id = id_map.dst_id(item.dst)
        if src_id < 0 or dst_id < 0:
            continue
        edges[0, kept] = src_id
        edges[1, kept] = dst_id
        kept += 1
    return edges[:, :kept]
