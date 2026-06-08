from __future__ import annotations

from collections import deque

import numpy as np

from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.logging import log, track

from .config import GRAPH_WINDOW_NAMES, GraphTowerConfig

GRAPH_WINDOW_FRACTIONS = (1.0, 0.35, 0.10)
DENSE_CPU_NODE_LIMIT = 5_000
DENSE_CPU_EDGE_LIMIT = 20_000


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

    def fit(self, interactions: InteractionTable, rng: np.random.Generator, verbose: bool = True) -> None:
        if not self.config.enabled or self.config.epochs < 1:
            return
        if self.id_map.num_src == 0 or self.id_map.num_dst == 0:
            return

        mapped_edges = _mapped_edges(interactions, self.id_map, self.config)
        if not mapped_edges:
            return

        total_edges = len(mapped_edges)
        for name, fraction in zip(GRAPH_WINDOW_NAMES, GRAPH_WINDOW_FRACTIONS):
            edge_count = max(1, int(total_edges * fraction))
            window_edges = mapped_edges[total_edges - edge_count :]
            edge_index = _graph_window_edges(window_edges, self.config, rng)
            if edge_index.shape[1] == 0:
                continue
            log(
                f"[gnn:{name}] train_edges={edge_index.shape[1]} model={self.config.model_name} "
                f"edge_weighting={self.config.edge_weighting}",
                enabled=verbose,
            )
            self._fit_one_window(name, edge_index, rng, verbose)

    def scores_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(GRAPH_WINDOW_NAMES)), dtype=np.float32)
        return self.scores_for_query_array(TestQueryArray.from_queries(queries))

    def scores_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(GRAPH_WINDOW_NAMES)), dtype=np.float32)

        scores = np.zeros((len(queries), queries.candidate_count, len(GRAPH_WINDOW_NAMES)), dtype=np.float32)
        src_ids = self.id_map.src_ids(queries.src)
        dst_ids = self.id_map.dst_ids(queries.candidates)
        for feature_idx, name in enumerate(GRAPH_WINDOW_NAMES):
            user_emb = self.user_embeddings.get(name)
            item_emb = self.item_embeddings.get(name)
            if user_emb is None or item_emb is None:
                continue

            seen_user = self.seen_users.get(name)
            seen_item = self.seen_items.get(name)
            if seen_user is None or seen_item is None:
                continue
            valid_src = (src_ids >= 0) & seen_user[src_ids.clip(min=0)]
            valid_dst = (dst_ids >= 0) & seen_item[dst_ids.clip(min=0)]
            for row_idx, src_id in enumerate(src_ids):
                if not valid_src[row_idx]:
                    continue
                valid = valid_dst[row_idx]
                if not np.any(valid):
                    continue
                scores[row_idx, valid, feature_idx] = item_emb[dst_ids[row_idx, valid]] @ user_emb[int(src_id)]
        return scores

    def _fit_one_window(
        self,
        name: str,
        edge_index: np.ndarray,
        rng: np.random.Generator,
        verbose: bool,
    ) -> None:
        import jittor as jt

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
        import jittor as jt

        model_name = self.config.model_name.lower()
        if model_name not in {"lightgcn", "xsimgcl"}:
            raise ValueError(f"unsupported graph model: {self.config.model_name}")

        if jt.flags.use_cuda == 0:
            return _DenseCPUGraphModel(
                num_users=self.id_map.num_src,
                num_items=self.id_map.num_dst,
                embedding_dim=self.config.embedding_dim,
                n_layers=self.config.layers,
                edge_index=edge_index,
                reg_weight=self.config.reg_weight,
            )

        from jittor_geometric.nn.models import LightGCN, XSimGCL

        edge_var = jt.array(edge_index, dtype=jt.int32)
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


class _DenseCPUGraphModel:
    def __new__(
        cls,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        n_layers: int,
        edge_index: np.ndarray,
        reg_weight: float,
    ):
        import jittor as jt

        node_count = int(num_users) + int(num_items)
        edge_count = int(edge_index.shape[1])
        if node_count > DENSE_CPU_NODE_LIMIT or edge_count > DENSE_CPU_EDGE_LIMIT:
            raise RuntimeError(
                "CPU graph fallback only supports small smoke graphs; "
                f"nodes={node_count}, edges={edge_count}. Use a CUDA/JittorGeometric environment "
                "or disable GNN for this run."
            )

        class DenseCPUGraphModel(jt.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_users = int(num_users)
                self.n_items = int(num_items)
                self.n_layers = max(int(n_layers), 1)
                self.reg_weight = float(reg_weight)
                self.user_embedding = jt.nn.Embedding(self.n_users, int(embedding_dim))
                self.item_embedding = jt.nn.Embedding(self.n_items, int(embedding_dim))
                self.adj = jt.array(_dense_normalized_bipartite_adj(edge_index, self.n_users, self.n_items))

            def execute(self, user_ids, item_ids, neg_item_ids):
                user_all, item_all = self.get_all_embeddings()
                user_emb = user_all[user_ids]
                pos_emb = item_all[item_ids]
                neg_emb = item_all[neg_item_ids]
                pos_scores = (user_emb * pos_emb).sum(dim=-1)
                neg_scores = (user_emb * neg_emb).sum(dim=-1)
                bpr_loss = -jt.log(jt.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                reg_loss = (
                    (self.user_embedding.weight[user_ids] ** 2).sum()
                    + (self.item_embedding.weight[item_ids] ** 2).sum()
                    + (self.item_embedding.weight[neg_item_ids] ** 2).sum()
                ) / max(int(user_ids.shape[0]), 1)
                return bpr_loss + self.reg_weight * reg_loss

            def get_all_embeddings(self):
                all_embeddings = jt.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
                embeddings = [all_embeddings]
                for _ in range(self.n_layers):
                    all_embeddings = jt.matmul(self.adj, all_embeddings)
                    embeddings.append(all_embeddings)
                output = embeddings[0]
                for value in embeddings[1:]:
                    output = output + value
                output = output / len(embeddings)
                return jt.split(output, [self.n_users, self.n_items], dim=0)

        return DenseCPUGraphModel()


def _dense_normalized_bipartite_adj(edge_index: np.ndarray, num_users: int, num_items: int) -> np.ndarray:
    node_count = int(num_users) + int(num_items)
    adj = np.zeros((node_count, node_count), dtype=np.float32)
    users = edge_index[0].astype(np.int64, copy=False)
    items = edge_index[1].astype(np.int64, copy=False) + int(num_users)
    np.add.at(adj, (users, items), 1.0)
    np.add.at(adj, (items, users), 1.0)
    degree = adj.sum(axis=1)
    valid = degree > 0
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])
    return adj * inv_sqrt[:, None] * inv_sqrt[None, :]


def _mapped_edges(
    interactions: InteractionTable,
    id_map: NodeIdMap,
    config: GraphTowerConfig,
) -> list[tuple[int, int, int]]:
    tail_limit = _mapped_edge_tail_limit(config)
    if tail_limit > 0:
        edge_buffer = deque(maxlen=tail_limit)
        for src, dst, time in zip(interactions.src, interactions.dst, interactions.time):
            src_id = id_map.src_id(int(src))
            dst_id = id_map.dst_id(int(dst))
            if src_id < 0 or dst_id < 0:
                continue
            edge_buffer.append((src_id, dst_id, int(time)))
        return list(edge_buffer)

    edges: list[tuple[int, int, int]] = []
    for src, dst, time in zip(interactions.src, interactions.dst, interactions.time):
        src_id = id_map.src_id(int(src))
        dst_id = id_map.dst_id(int(dst))
        if src_id < 0 or dst_id < 0:
            continue
        edges.append((src_id, dst_id, int(time)))
    return edges


def _mapped_edge_tail_limit(config: GraphTowerConfig) -> int:
    weighting = _normalized_edge_weighting(config.edge_weighting)
    if weighting != "none" or config.max_graph_edges <= 0:
        return 0
    smallest_fraction = min(GRAPH_WINDOW_FRACTIONS)
    return max(int(np.ceil(config.max_graph_edges / smallest_fraction)), config.max_graph_edges)


def _graph_window_edges(
    mapped_edges: list[tuple[int, int, int]],
    config: GraphTowerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if not mapped_edges:
        return np.empty((2, 0), dtype=np.int32)

    weighting = _normalized_edge_weighting(config.edge_weighting)
    if weighting == "none":
        edge_index = np.asarray([(src, dst) for src, dst, _ in mapped_edges], dtype=np.int32).T
        return _tail_edges(edge_index, config.max_graph_edges)

    edge_index, weights = _weighted_mapped_edges(mapped_edges, weighting, config.time_decay_ratio)
    if config.max_graph_edges > 0 and edge_index.shape[1] > config.max_graph_edges:
        edge_index = _sample_edges_by_weight(edge_index, weights, config.max_graph_edges, rng)
    return edge_index


def _weighted_mapped_edges(
    mapped_edges: list[tuple[int, int, int]],
    weighting: str,
    time_decay_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = _normalized_edge_weighting(weighting)
    train_end_time = max(time for _, _, time in mapped_edges)
    train_start_time = min(time for _, _, time in mapped_edges)
    graph_span = max(train_end_time - train_start_time, 1)
    tau = max(graph_span * max(time_decay_ratio, 1e-9), 1.0)

    stats: dict[tuple[int, int], list[int]] = {}
    for src_id, dst_id, event_time in mapped_edges:
        key = (int(src_id), int(dst_id))
        if key not in stats:
            stats[key] = [0, int(event_time)]
        stats[key][0] += 1
        stats[key][1] = max(stats[key][1], int(event_time))

    ordered = sorted(stats.items(), key=lambda item: item[1][1])
    edge_index = np.asarray([key for key, _ in ordered], dtype=np.int32).T
    weights = np.empty(edge_index.shape[1], dtype=np.float64)
    for idx, (_, (count, last_time)) in enumerate(ordered):
        weight = np.log1p(count)
        if normalized == "time_decay":
            weight *= np.exp(-max(train_end_time - last_time, 0) / tau)
        weights[idx] = max(float(weight), 1e-12)
    return edge_index, weights


def _sample_edges_by_weight(
    edge_index: np.ndarray,
    weights: np.ndarray,
    max_edges: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_edges <= 0 or edge_index.shape[1] <= max_edges:
        return edge_index
    total_weight = float(weights.sum())
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        return _tail_edges(edge_index, max_edges)
    probabilities = weights / total_weight
    selected = rng.choice(edge_index.shape[1], size=max_edges, replace=False, p=probabilities)
    selected.sort()
    return edge_index[:, selected]


def _tail_edges(edge_index: np.ndarray, max_edges: int) -> np.ndarray:
    if max_edges > 0 and edge_index.shape[1] > max_edges:
        return edge_index[:, -max_edges:]
    return edge_index


def _normalized_edge_weighting(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"none", "repeat", "time_decay"}:
        return normalized
    raise ValueError(f"unsupported graph edge weighting: {value}")
