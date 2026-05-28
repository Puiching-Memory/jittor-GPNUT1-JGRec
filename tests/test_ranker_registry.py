import pytest

from jgrec.rankers.registry import RankerRegistry


def test_registry_normalizes_names_and_returns_sorted_names():
    registry = RankerRegistry()
    registry.register(" beta ", lambda config: ("beta", config))
    registry.register("alpha", lambda config: ("alpha", config))

    assert registry.names() == ("alpha", "beta")
    assert registry.create("BETA", {"seed": 42}) == ("beta", {"seed": 42})


def test_registry_rejects_empty_name():
    registry = RankerRegistry()

    with pytest.raises(ValueError, match="ranker name cannot be empty"):
        registry.register("   ", lambda config: config)


def test_registry_reports_available_names_for_unknown_ranker():
    registry = RankerRegistry()
    registry.register("alpha", lambda config: config)

    with pytest.raises(ValueError, match="unknown ranker 'missing', available: alpha"):
        registry.create("missing", object())
