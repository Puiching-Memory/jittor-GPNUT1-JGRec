import re

from jgrec.cli import CLIConfig, _build_run_name, _ranker_config


def test_run_name_is_human_readable_for_default_hybrid():
    args = CLIConfig()

    name = _build_run_name(args, _ranker_config(args))

    assert re.fullmatch(r"hybrid_full_cuda_seed-42_gnn-xsimgcl_sequence-on_[0-9a-f]{8}", name)
    assert "rw" not in name
    assert "vr" not in name
    assert "tbs" not in name


def test_run_name_describes_smoke_cpu_run():
    args = CLIConfig(limit_rows=2, cpu=True, disable_gnn=True, disable_seq=True)

    name = _build_run_name(args, _ranker_config(args))

    assert re.fullmatch(r"hybrid_sample-2-rows_cpu_seed-42_gnn-off_sequence-off_[0-9a-f]{8}", name)


def test_run_name_digest_keeps_hidden_config_distinct():
    default_name = _build_run_name(CLIConfig(), _ranker_config(CLIConfig()))
    tuned_args = CLIConfig(max_train_events=32)

    tuned_name = _build_run_name(tuned_args, _ranker_config(tuned_args))

    assert tuned_name.startswith("hybrid_full_cuda_seed-42_gnn-xsimgcl_sequence-on_")
    assert tuned_name != default_name
