import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from lsmp_ai.common.constants import FEATURE_COLUMNS, LABEL_ANOMALY, LABEL_NORMAL
from lsmp_ai.scripts.evaluate import run_evaluation


def test_run_evaluation_without_db_url(monkeypatch):
    """Test that evaluation utility cleanly returns False when DATABASE_URL is missing."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    success = run_evaluation(db_url=None)
    assert success is False


def test_run_evaluation_mocked(tmp_path):
    """Test evaluation utility with mocked DBClient and small synthetic dataset."""
    # Create a small dataset for fast unit test execution
    rows = []
    for i in range(30):
        row = {
            "window_end": f"2026-07-01 10:{i:02d}:00",
            "label": LABEL_ANOMALY if i < 5 else LABEL_NORMAL,
        }
        for col in FEATURE_COLUMNS:
            row[col] = float(np.random.rand())
        rows.append(row)
    df_mini = pd.DataFrame(rows)
    mini_csv = tmp_path / "eval_mini.csv"
    df_mini.to_csv(mini_csv, index=False)

    with patch("lsmp_ai.scripts.evaluate.DBClient") as MockDBClient:
        mock_instance = MagicMock()
        mock_instance.db_url = "postgresql://postgres:postgres@localhost:5432/test_db"
        MockDBClient.return_value = mock_instance

        res = run_evaluation(
            db_url="postgresql://postgres:postgres@localhost:5432/test_db",
            dataset_path=str(mini_csv),
            output_dir=tmp_path,
            benchmark_runs=1,
        )
        assert res is not False
        assert isinstance(res, pd.DataFrame)
