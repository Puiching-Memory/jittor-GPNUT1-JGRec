import importlib
import re
import sys

from jgrec.rankers.registry import create_ranker


def _cli_symbols():
    module = importlib.import_module("jgrec.cli")
    return module.CLIConfig, module._build_run_name, module._ranker_config


def test_cli_import_does_not_load_graph_backend():
    sys.modules.pop("jgrec.cli", None)
    sys.modules.pop("jgrec.rankers.hybrid.gnn", None)

    importlib.import_module("jgrec.cli")

    assert "jgrec.rankers.hybrid.gnn" not in sys.modules


def test_disabled_hybrid_ranker_does_not_load_heavy_backends():
    CLIConfig, _, _ranker_config = _cli_symbols()
    sys.modules.pop("jgrec.rankers.hybrid.gnn", None)
    sys.modules.pop("jgrec.rankers.hybrid.sequence", None)
    sys.modules.pop("jgrec.rankers.hybrid.two_tower", None)
    args = CLIConfig(disable_gnn=True, disable_seq=True, disable_two_tower=True)

    create_ranker("hybrid", _ranker_config(args))

    assert "jgrec.rankers.hybrid.gnn" not in sys.modules
    assert "jgrec.rankers.hybrid.sequence" not in sys.modules
    assert "jgrec.rankers.hybrid.two_tower" not in sys.modules


def test_run_name_is_human_readable_for_default_hybrid():
    CLIConfig, _build_run_name, _ranker_config = _cli_symbols()
    args = CLIConfig()

    name = _build_run_name(args, _ranker_config(args))

    assert re.fullmatch(
        r"hybrid_full_cuda_seed-42_gnn-xsimgcl_edges-none_auto-on_prior-on_tower-on_sequence-on_[0-9a-f]{8}",
        name,
    )
    assert "rw" not in name
    assert "vr" not in name
    assert "tbs" not in name


def test_run_name_describes_smoke_cpu_run():
    CLIConfig, _build_run_name, _ranker_config = _cli_symbols()
    args = CLIConfig(limit_rows=2, cpu=True, disable_gnn=True, disable_seq=True, disable_two_tower=True)

    name = _build_run_name(args, _ranker_config(args))

    assert re.fullmatch(
        r"hybrid_sample-2-rows_cpu_seed-42_gnn-off_edges-off_auto-on_prior-on_tower-off_sequence-off_[0-9a-f]{8}",
        name,
    )


def test_run_name_digest_keeps_hidden_config_distinct():
    CLIConfig, _build_run_name, _ranker_config = _cli_symbols()
    default_name = _build_run_name(CLIConfig(), _ranker_config(CLIConfig()))
    tuned_args = CLIConfig(max_train_events=32)

    tuned_name = _build_run_name(tuned_args, _ranker_config(tuned_args))

    assert tuned_name.startswith("hybrid_full_cuda_seed-42_gnn-xsimgcl_edges-none_auto-on_prior-on_tower-on_sequence-on_")
    assert tuned_name != default_name


def test_run_name_describes_graph_edge_weighting():
    CLIConfig, _build_run_name, _ranker_config = _cli_symbols()
    args = CLIConfig(gnn_edge_weighting="time_decay")

    name = _build_run_name(args, _ranker_config(args))

    assert name.startswith(
        "hybrid_full_cuda_seed-42_gnn-xsimgcl_edges-time-decay_auto-on_prior-on_tower-on_sequence-on_"
    )


def test_run_name_describes_two_tower_status():
    CLIConfig, _build_run_name, _ranker_config = _cli_symbols()
    args = CLIConfig(disable_two_tower=True)

    name = _build_run_name(args, _ranker_config(args))

    assert name.startswith("hybrid_full_cuda_seed-42_gnn-xsimgcl_edges-none_auto-on_prior-on_tower-off_sequence-on_")


def test_run_name_describes_auto_strategy_and_candidate_prior_status():
    CLIConfig, _build_run_name, _ranker_config = _cli_symbols()
    args = CLIConfig(auto_strategy=False, disable_candidate_prior=True)

    name = _build_run_name(args, _ranker_config(args))

    assert name.startswith("hybrid_full_cuda_seed-42_gnn-xsimgcl_edges-none_auto-off_prior-off_tower-on_sequence-on_")


def test_cli_config_passes_negative_sampling_workers_to_hybrid():
    CLIConfig, _, _ranker_config = _cli_symbols()
    args = CLIConfig(negative_sampling_workers=4, seq_score_batch_size=128, two_tower_score_batch_size=256)

    config = _ranker_config(args)

    assert config.negative_sampling_workers == 4
    assert config.two_tower_config().negative_sampling_workers == 4
    assert config.sequence_config().score_batch_size == 128
    assert config.two_tower_config().score_batch_size == 256


def test_cli_config_passes_auto_strategy_and_candidate_prior_to_hybrid():
    CLIConfig, _, _ranker_config = _cli_symbols()
    args = CLIConfig(auto_strategy=False, disable_candidate_prior=True, test_candidate_negative_ratio=0.4)

    config = _ranker_config(args)

    assert not config.auto_strategy_enabled
    assert not config.candidate_prior_enabled
    assert config.test_candidate_negative_ratio == 0.4


def test_encoder_state_cache_is_operational_and_does_not_change_run_name_digest():
    CLIConfig, _build_run_name, _ranker_config = _cli_symbols()
    enabled_args = CLIConfig()
    disabled_args = CLIConfig(encoder_state_cache=False)

    enabled_config = _ranker_config(enabled_args)
    disabled_config = _ranker_config(disabled_args)

    assert enabled_config.encoder_state_cache_enabled
    assert not disabled_config.encoder_state_cache_enabled
    assert _build_run_name(enabled_args, enabled_config) == _build_run_name(disabled_args, disabled_config)


def test_operational_resume_options_do_not_change_run_name_digest():
    CLIConfig, _build_run_name, _ranker_config = _cli_symbols()
    base_args = CLIConfig()
    resume_args = CLIConfig(dataset="dataset2", run_name="manual", resume_existing=True)

    assert _build_run_name(resume_args, _ranker_config(resume_args)) == _build_run_name(
        base_args,
        _ranker_config(base_args),
    )
