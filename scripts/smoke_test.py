import sys
sys.path.insert(0, "/desay120T/ct/dev/uid01954/jittor-GPNUT1-JGRec/src")

import jittor as jt
import numpy as np
from jgrec.rankers.temporal_graph.model import (
    EndToEndTemporalGraphModel,
    TemporalGraphModelConfig,
)

# Enable CUDA for test
jt.flags.use_cuda = 1

config = TemporalGraphModelConfig(
    num_nodes=100,
    history_len=10,
    candidate_history_len=5,
    hidden_size=32,
    layers=2,
    heads=4,
    dropout=0.0,
    time_span=1,
    candidate_feature_dim=6,
)

model = EndToEndTemporalGraphModel(config)

batch_size = 2
candidate_count = 3

src_ids = jt.array([1, 2], dtype=jt.int32)
candidate_ids = jt.array([[3, 4, 5], [6, 7, 8]], dtype=jt.int32)
cur_times = jt.array([10, 20], dtype=jt.int32)
src_neighbor_ids = jt.array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0], [2, 3, 4, 5, 0, 0, 0, 0, 0, 0]], dtype=jt.int32)
src_neighbor_times = jt.array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0], [2, 3, 4, 5, 0, 0, 0, 0, 0, 0]], dtype=jt.int32)
candidate_neighbor_ids = jt.array([
    [[1, 2, 0, 0, 0], [2, 3, 4, 0, 0], [3, 4, 5, 6, 0]],
    [[4, 5, 0, 0, 0], [5, 6, 7, 0, 0], [6, 7, 8, 9, 0]]
], dtype=jt.int32)
candidate_neighbor_times = jt.array([
    [[1, 2, 0, 0, 0], [2, 3, 4, 0, 0], [3, 4, 5, 6, 0]],
    [[4, 5, 0, 0, 0], [5, 6, 7, 0, 0], [6, 7, 8, 9, 0]]
], dtype=jt.int32)
candidate_features = jt.array(np.random.randn(batch_size, candidate_count, 6).astype(np.float32))

logits = model(
    src_ids=src_ids,
    candidate_ids=candidate_ids,
    cur_times=cur_times,
    src_neighbor_ids=src_neighbor_ids,
    src_neighbor_times=src_neighbor_times,
    candidate_neighbor_ids=candidate_neighbor_ids,
    candidate_neighbor_times=candidate_neighbor_times,
    candidate_features=candidate_features,
)

assert logits.shape == (batch_size, candidate_count), f"Unexpected logits shape: {logits.shape}"
print("Smoke test passed! Logits shape:", logits.shape)
print("Logits:", logits)
