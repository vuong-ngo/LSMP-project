# ============================================================================
# file: tests/test_real_db_import.py
# Description: Pytest module for verifying real database import and fallback pipeline.
# ============================================================================

import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from lsmp_ai.common.constants import FEATURE_COLUMNS, LABEL_NORMAL, LABEL_ANOMALY
from lsmp_ai.io.db_client import DBClient


def prepare_test_dataset(test_dir: Path | str) -> tuple[Path, Path]:
    """Generates synthetic test datasets for database import verification."""
    p_dir = Path(test_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    raw_path = p_dir / "sample_raw_logs.csv"
    features_path = p_dir / "sample_features.csv"

    # 1. Raw logs
    raw_rows = [
        {
            "timestamp": "2026-07-01 10:00:00+00:00",
            "source_ip": "192.168.1.100",
            "raw_log": "Failed password for root from 192.168.1.100 port 22 ssh2",
            "event_type": "auth",
            "severity": 8,
            "rule_id": 5710,
        },
        {
            "timestamp": "2026-07-01 10:00:05+00:00",
            "source_ip": "192.168.1.100",
            "raw_log": "Failed password for admin from 192.168.1.100 port 22 ssh2",
            "event_type": "auth",
            "severity": 8,
            "rule_id": 5710,
        },
    ]
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    # 2. Computed features
    feature_rows = []
    for i in range(10):
        row = {
            "window_start": f"2026-07-01 10:{i:02d}:00+00:00",
            "src_ip": f"192.168.1.{100 + i}",
            "label": LABEL_ANOMALY if i < 2 else LABEL_NORMAL,
        }
        for col in FEATURE_COLUMNS:
            row[col] = float(np.random.rand())
        feature_rows.append(row)
    pd.DataFrame(feature_rows).to_csv(features_path, index=False)

    return raw_path, features_path


def run_database_import_test(db_url: str | None = None) -> bool:
    """Tests real database connectivity and feature vector upsert."""
    client = DBClient(db_url)
    if not client.is_connected:
        return False
    try:
        df_sample = pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}])
        df_sample["src_ip"] = "192.168.1.250"
        df_sample["window_start"] = pd.Timestamp.now(tz="UTC")
        client.write_feature_vectors(df_sample)
        return True
    except Exception:
        return False


def test_prepare_test_dataset(tmp_path: Path):
    raw_path, features_path = prepare_test_dataset(tmp_path)
    assert raw_path.exists()
    assert features_path.exists()


def test_real_db_import_runner():
    db_url = os.getenv("DATABASE_URL")
    db_client = DBClient(db_url)
    try:
        with db_client.engine.connect() as conn:
            pass
    except Exception as conn_err:
        pytest.skip(f"Local PostgreSQL database not reachable: {conn_err}. Test skipped cleanly.")

    # DB is reachable, run real database test without masking errors
    success = run_database_import_test(db_url)
    assert success is True
