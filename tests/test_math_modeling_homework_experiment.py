from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "数学建模作业" / "实验代码_图论模型动态推荐预测.py"


def _write_csv(zf: zipfile.ZipFile, name: str, text: str) -> None:
    zf.writestr(name, text.strip() + "\n")


def _make_zip(path: Path) -> None:
    candidate_header = ",".join(f"c{i}" for i in range(1, 101))
    candidate_values = ",".join(str(10 + i) for i in range(100))
    with zipfile.ZipFile(path, "w") as zf:
        _write_csv(
            zf,
            "dataset1/train.csv",
            """
src,dst,time
1,10,1
1,10,2
1,11,3
2,20,4
2,20,5
3,30,6
1,10,7
2,20,8
3,30,9
            """,
        )
        _write_csv(zf, "dataset1/test.csv", f"src,time,{candidate_header}\n1,10,{candidate_values}")
        _write_csv(
            zf,
            "dataset2/train.csv",
            """
src,dst,time,split
100,200,1,0
101,201,2,0
100,202,3,0
101,202,4,0
100,200,5,0
101,201,6,0
            """,
        )
        _write_csv(zf, "dataset2/test.csv", f"src,time,{candidate_header}\n100,10,{candidate_values}")


def _load_homework_module():
    spec = importlib.util.spec_from_file_location("math_modeling_homework_experiment", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_experiment_reads_zip_and_reports_two_non_jittor_models(tmp_path: Path) -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "import jittor" not in source
    assert "jittor_geometric" not in source

    zip_path = tmp_path / "data_A.zip"
    _make_zip(zip_path)

    module = _load_homework_module()
    dataset = module.load_dataset_from_zip(zip_path, "dataset1", max_train_rows=20, max_test_rows=1)
    result = module.evaluate_dataset(dataset, max_val_events=3, candidate_count=5, seed=123)

    assert result["dataset"] == "dataset1"
    assert set(result["models"]) == {"popularity_recency", "graph_structure"}
    assert 0.0 <= result["models"]["popularity_recency"]["mrr"] <= 1.0
    assert 0.0 <= result["models"]["graph_structure"]["hit_at_5"] <= 1.0
    assert result["validation_events"] == 3
