import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_prediction_ties.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnose_prediction_ties",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
diagnose_csv = MODULE.diagnose_csv


def test_diagnose_csv_counts_exact_ties(tmp_path: Path):
    path = tmp_path / "scores.csv"
    path.write_text(
        "0.5,0.5,0.2\n0.3,0.2,0.1\n0.7,0.7,0.7\n",
        encoding="utf-8",
    )

    report = diagnose_csv(path)

    assert report["rows"] == 3
    assert report["rows_with_ties"] == 2
    assert report["duplicate_adjacencies"] == 3
    assert report["maximum_duplicate_adjacencies_per_row"] == 2
