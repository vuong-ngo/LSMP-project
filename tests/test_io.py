from datetime import datetime
import pandas as pd
import pytest

from lsmp_ai.common.constants import FEATURE_COLUMNS
from lsmp_ai.io.data_loader import DataLoader
from lsmp_ai.io.db_client import DatabaseClient


def test_data_loader_csv(tmp_path):
    data = {col: [1.0] for col in FEATURE_COLUMNS}
    data["window_start"] = ["2026-07-01"]
    data["src_ip"] = ["192.168.1.1"]
    data["feature_version"] = ["v1"]
    data["label"] = ["Normal"]

    df = pd.DataFrame(data)
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)

    loader = DataLoader()
    loaded = loader.load_from_csv(str(path))
    assert len(loaded) == 1

    X, y = loader.split_features_target(loaded)
    assert X.shape[1] == 14
    assert y is not None
    assert y.iloc[0] == "Normal"


def test_validate_table_name_strict_whitelist():
    """Verify that validate_table_name only allows whitelisted tables and rejects arbitrary identifiers (S6)."""
    from lsmp_ai.io.db_client import validate_table_name, VALID_TABLE_NAMES

    for table in VALID_TABLE_NAMES:
        assert validate_table_name(table) == table

    invalid_tables = [
        "users", "secrets", "passwords", "auth_tokens",
        "log_event; DROP TABLE log_event;", "unknown_table"
    ]
    for bad_table in invalid_tables:
        with pytest.raises(ValueError, match="Invalid or unsafe SQL table name"):
            validate_table_name(bad_table)


def test_sanitize_db_url_and_errors():
    """Verify that database connection URLs and exception strings are sanitized of credentials (S8)."""
    from lsmp_ai.io.db_client import sanitize_db_url, sanitize_error_msg

    raw_url = "postgresql://myuser:SuperSecretPassword123@prod-db.internal:5432/lsmp_db"
    masked = sanitize_db_url(raw_url)
    assert "SuperSecretPassword123" not in masked
    assert "myuser:***@prod-db.internal:5432" in masked

    error_str = "OperationalError: could not connect to postgresql://admin:RootSecret456@10.0.0.5:5432/db"
    sanitized = sanitize_error_msg(error_str)
    assert "RootSecret456" not in sanitized
    assert "admin:***@" in sanitized


def test_db_client_fetch_df_parameter_binding():
    """Verify DBClient.fetch_df accepts arbitrary bind parameters without raising TypeError."""
    from unittest.mock import MagicMock, patch
    client = DatabaseClient(db_url="postgresql://user:pass@localhost:5432/db")

    with patch.object(DatabaseClient, "engine", new_callable=MagicMock) as mock_engine:
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        with patch("pandas.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({"col": [1, 2]})

            df = client.fetch_df(
                "SELECT * FROM log_event WHERE timestamp >= :start_time LIMIT :limit",
                start_time="2026-07-01",
                limit=10,
            )
            assert len(df) == 2
            mock_read_sql.assert_called_once()
            call_kwargs = mock_read_sql.call_args[1]
            assert call_kwargs["params"]["start_time"] == "2026-07-01"
            assert call_kwargs["params"]["limit"] == 10


def test_data_loader_prepare_labels():
    """Verify that DataLoader.prepare_labels works properly without NameError."""
    import numpy as np
    labels = pd.Series(["Normal", "Anomaly", "unknown"])
    prepared = DataLoader.prepare_labels(labels)
    assert prepared.iloc[0] == 1.0
    assert prepared.iloc[1] == -1.0
    assert np.isnan(prepared.iloc[2])


def test_audit_and_separate_unknowns_structured_storage(tmp_path):
    """Verify scientific storage structure and metadata generation for unknown data (Policy A2)."""
    data = {col: [1.0, 2.0, 3.0, 4.0] for col in FEATURE_COLUMNS}
    data["label"] = ["Normal", "Anomaly", "unknown", None]
    df = pd.DataFrame(data)

    out_dir = tmp_path / "unlabeled_test"
    df_labeled, df_unknown = DataLoader.audit_and_separate_unknowns(df, output_dir=out_dir)

    assert len(df_labeled) == 2
    assert len(df_unknown) == 2

    csv_path = out_dir / "unlabeled_samples.csv"
    assert csv_path.exists()
    loaded_unknown = DataLoader.load_unlabeled_samples(csv_path)
    assert len(loaded_unknown) == 2

    meta_path = out_dir / "audit_metadata.json"
    assert meta_path.exists()
    meta = DataLoader.load_audit_metadata(meta_path)
    assert meta["total_input_samples"] == 4
    assert meta["unknown_samples_count"] == 2
    assert meta["unknown_ratio_percent"] == 50.0
    assert meta["breakdown"]["missing_or_nan_count"] == 1
    assert meta["breakdown"]["explicit_unknown_alias_count"] == 1





