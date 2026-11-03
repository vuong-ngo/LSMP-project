# ============================================================================
# file: tests/test_data_preparation.py
# Description: Comprehensive unit tests for modular dataset preparation architecture
#              covering BaseDataPreparer, CICIDS2017Preparer, LoghubPreparer,
#              and UnifiedDatasetPreparer.
# ============================================================================

import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from lsmp_ai.common.constants import FEATURE_COLUMNS, LABEL_NORMAL, LABEL_ANOMALY
from lsmp_ai.data_preparation.base import BaseDataPreparer
from lsmp_ai.data_preparation.cicids2017 import CICIDS2017Preparer
from lsmp_ai.data_preparation.loghub import LoghubPreparer
from lsmp_ai.data_preparation.unified import UnifiedDatasetPreparer
import scripts.prepare_data as prepare_data_module


@pytest.fixture
def dummy_cicids_csv(tmp_path: Path) -> Path:
    """Creates a temporary synthetic CICIDS2017 CSV file."""
    csv_file = tmp_path / "Monday-WorkingHours.pcap_ISCX.csv"
    data = {
        "Flow Packets/s": [10.5, 20.0, 100.0, 500.0, 1200.0],
        "Flow Bytes/s": [1000.0, 2000.0, 50000.0, 200000.0, 500000.0],
        "Total Fwd Packets": [5, 10, 50, 100, 200],
        "Total Backward Packets": [4, 8, 45, 90, 180],
        "Packet Length Variance": [12.0, 15.0, 30.0, 45.0, 90.0],
        "SYN Flag Count": [0, 0, 1, 5, 20],
        "ACK Flag Count": [1, 2, 5, 10, 25],
        "RST Flag Count": [0, 0, 0, 1, 1],
        "PSH Flag Count": [0, 1, 2, 5, 10],
        "Fwd PSH Flags": [0, 0, 0, 0, 0],
        "Active Mean": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Subflow Fwd Packets": [5, 10, 50, 100, 200],
        "Flow IAT Mean": [100000.0, 200000.0, 300000.0, 400000.0, 500000.0],
        "Flow IAT Std": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0],
        "Avg Bwd Segment Size": [50.0, 60.0, 70.0, 80.0, 90.0],
        "ECE Flag Count": [0, 0, 0, 0, 0],
        "CWE Flag Count": [0, 0, 0, 0, 0],
        "Label": ["BENIGN", "BENIGN", "BENIGN", "PortScan", "DDoS"],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def dummy_loghub_log(tmp_path: Path) -> Path:
    """Creates a temporary synthetic OpenSSH Syslog file."""
    log_file = tmp_path / "SSH.log"
    lines = [
        "Dec 10 07:07:38 server sshd[1234]: Failed password for invalid user admin from 192.168.1.100 port 3456 ssh2",
        "Dec 10 07:07:40 server sshd[1234]: Failed password for invalid user root from 192.168.1.100 port 3458 ssh2",
        "Dec 10 07:07:42 server sshd[1234]: Failed password for invalid user test from 192.168.1.100 port 3460 ssh2",
        "Dec 10 07:08:00 server sshd[5678]: Accepted password for deploy from 10.0.0.5 port 4567 ssh2",
        "Dec 10 07:09:00 server sshd[9012]: POSSIBLE BREAK-IN ATTEMPT! [192.168.1.200]",
    ]
    log_file.write_text("\n".join(lines))
    return log_file


class TestBaseDataPreparer:
    """Tests for BaseDataPreparer abstract contract and data transformation helpers."""

    def test_sanitize_dataframe(self):
        preparer = CICIDS2017Preparer()
        df = pd.DataFrame({
            "auth_fail_count": [1.0, np.inf, np.nan],
            "http_req_rate": [10.0, -np.inf, 30.0],
        })
        sanitized = preparer.sanitize_dataframe(df)

        for col in FEATURE_COLUMNS:
            assert col in sanitized.columns
            assert not np.isnan(sanitized[col].values).any()
            assert not np.isinf(sanitized[col].values).any()

        assert "label" in sanitized.columns
        assert "attack_category" in sanitized.columns

    def test_scientific_sanitization_retains_missing_for_train_only_imputation(self):
        preparer = CICIDS2017Preparer()
        df = pd.DataFrame({"auth_fail_count": [1.0, np.inf, np.nan]})

        sanitized = preparer.sanitize_dataframe(df, fill_missing=False)

        assert sanitized["auth_fail_count"].isna().sum() == 2
        assert not np.isinf(sanitized["auth_fail_count"]).any()

    def test_temporal_split(self):
        preparer = CICIDS2017Preparer()
        data = {col: np.random.randn(100) for col in FEATURE_COLUMNS}
        data["label"] = [LABEL_NORMAL] * 80 + [LABEL_ANOMALY] * 20
        df = pd.DataFrame(data)

        train, val, test = preparer.temporal_split(df, frac=(0.6, 0.2, 0.2))
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_temporal_split_does_not_cross_identical_timestamps(self):
        preparer = CICIDS2017Preparer()
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(
                ["2020-01-01 00:00:00"] * 4
                + ["2020-01-01 00:01:00"] * 4
                + ["2020-01-01 00:02:00"] * 4
            ),
            **{col: np.arange(12, dtype=float) for col in FEATURE_COLUMNS},
            "label": [LABEL_NORMAL] * 12,
        })

        train, val, test = preparer.temporal_split(df, frac=(0.5, 0.25, 0.25))

        assert set(train["timestamp"]).isdisjoint(set(val["timestamp"]))
        assert set(val["timestamp"]).isdisjoint(set(test["timestamp"]))


class TestCICIDS2017Preparer:
    """Tests for CICIDS2017Preparer lifecycle and feature standardization."""

    def test_sort_chronologically(self):
        files = [
            Path("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
            Path("Monday-WorkingHours.pcap_ISCX.csv"),
            Path("Wednesday-workingHours.pcap_ISCX.csv"),
            Path("Tuesday-WorkingHours.pcap_ISCX.csv"),
        ]
        sorted_files = CICIDS2017Preparer.sort_chronologically(files)
        names = [f.name for f in sorted_files]
        assert "Monday" in names[0]
        assert "Tuesday" in names[1]
        assert "Wednesday" in names[2]
        assert "Friday" in names[3]

    def test_discover_prefers_generated_labelled_flows(self, tmp_path: Path):
        generated = tmp_path / "GeneratedLabelledFlows" / "TrafficLabelling "
        generated.mkdir(parents=True)
        generated_file = generated / "Monday-WorkingHours.pcap_ISCX.csv"
        generated_file.write_text("Timestamp, Label\n2017-07-03 09:00:00,BENIGN\n")
        legacy_file = tmp_path / generated_file.name
        legacy_file.write_text(" Label\nBENIGN\n")

        preparer = CICIDS2017Preparer(raw_dir=tmp_path)
        discovered = preparer.discover_raw_files()

        assert discovered == [generated_file]

    def test_clean_and_transform(self, dummy_cicids_csv: Path):
        preparer = CICIDS2017Preparer()
        df, stats = preparer.clean_and_transform(dummy_cicids_csv)

        assert len(df) == 5
        assert stats["normal_count"] == 3
        assert stats["anomaly_count"] == 2
        for col in FEATURE_COLUMNS:
            assert col in df.columns
        assert "label" in df.columns
        assert "attack_category" in df.columns
        assert set(df["timestamp_source"]) == {"synthetic_row_order_proxy"}

    def test_clean_and_transform_uses_original_timestamp(self, dummy_cicids_csv: Path):
        source = pd.read_csv(dummy_cicids_csv)
        source.insert(
            0,
            "Timestamp",
            [
                "2017-07-03 09:00:00",
                "2017-07-03 09:00:01",
                "2017-07-03 09:00:02",
                "2017-07-03 09:00:03",
                "2017-07-03 09:00:04",
            ],
        )
        source.to_csv(dummy_cicids_csv, index=False)

        preparer = CICIDS2017Preparer()
        df, stats = preparer.clean_and_transform(dummy_cicids_csv)

        assert stats["timestamp_source"] == "original_flow_timestamp"
        assert df["timestamp"].iloc[0] == pd.Timestamp("2017-07-03 09:00:00", tz="UTC")

    def test_prepare(self, dummy_cicids_csv: Path, tmp_path: Path):
        output_dir = tmp_path / "output_cicids"
        preparer = CICIDS2017Preparer(raw_dir=dummy_cicids_csv.parent, output_dir=output_dir)
        dataset_path = preparer.prepare(max_samples=4)

        assert dataset_path.exists()
        assert (output_dir / "train.csv").exists()
        assert (output_dir / "val.csv").exists()
        assert (output_dir / "test.csv").exists()


class TestLoghubPreparer:
    """Tests for LoghubPreparer log parsing and feature vectorization."""

    def test_parse_log_line(self):
        preparer = LoghubPreparer()
        fail_line = "Dec 10 07:07:38 server sshd[1234]: Failed password for invalid user admin from 1.2.3.4 port 3456 ssh2"
        rec1 = preparer.parse_log_line(fail_line)
        assert rec1["event_type"] == "failed_password"
        assert rec1["username"] == "admin"
        assert rec1["src_ip"] == "1.2.3.4"

        acc_line = "Dec 10 07:08:00 server sshd[5678]: Accepted password for root from 5.6.7.8 port 4567 ssh2"
        rec2 = preparer.parse_log_line(acc_line)
        assert rec2["event_type"] == "accepted"
        assert rec2["username"] == "root"
        assert rec2["src_ip"] == "5.6.7.8"

        brk_line = "Dec 10 07:09:00 server sshd[9012]: POSSIBLE BREAK-IN ATTEMPT! [9.10.11.12]"
        rec3 = preparer.parse_log_line(brk_line)
        assert rec3["event_type"] == "break_in_attempt"
        assert rec3["src_ip"] == "9.10.11.12"

    def test_clean_and_transform(self, dummy_loghub_log: Path):
        preparer = LoghubPreparer()
        df, stats = preparer.clean_and_transform(dummy_loghub_log)

        assert not df.empty
        for col in FEATURE_COLUMNS:
            assert col in df.columns
        assert "label" in df.columns
        assert "attack_category" in df.columns
        assert stats["initial_log_lines"] == 5

    def test_prepare(self, dummy_loghub_log: Path, tmp_path: Path):
        output_dir = tmp_path / "output_loghub"
        preparer = LoghubPreparer(raw_dir=dummy_loghub_log.parent, output_dir=output_dir)
        dataset_path = preparer.prepare()

        assert dataset_path.exists()
        assert (output_dir / "train.csv").exists()
        assert (output_dir / "val.csv").exists()
        assert (output_dir / "test.csv").exists()


class TestUnifiedDatasetPreparer:
    """Tests for UnifiedDatasetPreparer multi-source aggregation."""

    def test_unified_preparation(self, dummy_cicids_csv: Path, dummy_loghub_log: Path, tmp_path: Path):
        raw_dir = tmp_path
        output_dir = tmp_path / "unified_output"
        orchestrator = UnifiedDatasetPreparer(raw_dir=raw_dir, output_dir=output_dir)

        dataset_path = orchestrator.prepare_all(max_samples=10)
        assert dataset_path.exists()
        df = pd.read_csv(dataset_path)
        assert len(df) > 0
        for col in FEATURE_COLUMNS:
            assert col in df.columns
        assert "label" in df.columns
        assert "attack_category" in df.columns


class TestPrepareDataCompatibility:
    """Tests backward compatibility helper functions in scripts/prepare_data.py."""

    def test_prepare_data_helpers(self):
        raw_dir = prepare_data_module.find_raw_data_dir()
        assert raw_dir.exists()
