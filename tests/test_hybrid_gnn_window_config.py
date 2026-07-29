from jgrec.rankers.hybrid.config import (
    GraphTowerConfig,
    graph_window_edge_parameters,
)


def test_window_edge_parameters_override_only_named_graph_windows():
    config = GraphTowerConfig(
        edge_weighting="none",
        time_decay_ratio=0.05,
        recent_edge_weighting="time_decay",
        recent_time_decay_ratio=0.10,
        short_edge_weighting="repeat",
    )

    assert graph_window_edge_parameters(config, "gnn_full") == ("none", 0.05)
    assert graph_window_edge_parameters(config, "gnn_recent") == (
        "time_decay",
        0.10,
    )
    assert graph_window_edge_parameters(config, "gnn_short") == ("repeat", 0.05)
