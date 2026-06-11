import importlib


def _cli_symbols():
    module = importlib.import_module("jgrec.cli")
    return module.CLIConfig, module._ranker_config


def test_temporal_graph_cli_builds_config_without_cpu_switch() -> None:
    CLIConfig, _ranker_config = _cli_symbols()

    config = _ranker_config(CLIConfig(model="temporal-graph"))

    assert config.history_len == 64
    assert config.training_candidates == "test_like"
    assert config.validation_candidates == "test_like"
    assert config.candidate_recent_feature_group == "recency_rank"
    assert not hasattr(config, "use_cuda")


def test_temporal_graph_cli_rejects_cpu_flag() -> None:
    module = importlib.import_module("jgrec.cli")

    try:
        module._validate_device_args(module.CLIConfig(model="temporal-graph", cpu=True))
    except ValueError as exc:
        assert "requires CUDA" in str(exc)
    else:
        raise AssertionError("temporal-graph --cpu should be rejected")
