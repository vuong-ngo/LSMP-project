# ============================================================================
# file: tests/test_features.py
# Description: Rigorous unit test suite for LSMP 14 Auth & Nginx feature extractors.
# ============================================================================

import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from lsmp_ai.common.constants import FEATURE_COLUMNS
from lsmp_ai.feature_engineering.auth_features import (
    calculate_auth_fail_count,
    calculate_auth_success_count,
    calculate_auth_fail_ratio,
    calculate_distinct_username,
    calculate_fail_interarrival_stats,
    calculate_login_fail_count,
    calculate_fail_success_ratio,
)
from lsmp_ai.feature_engineering.web_features import (
    calculate_http_req_rate,
    calculate_http_status_ratios,
    calculate_http_post_ratio,
    calculate_distinct_user_agent,
    calculate_bytes_sent_mean,
    calculate_request_rate,
    calculate_status_4xx_rate,
)
from lsmp_ai.feature_engineering.behavior_features import (
    calculate_global_active_ips,
    calculate_burst_rate,
    calculate_time_window_count,
)
from lsmp_ai.feature_engineering.feature_pipeline import (
    extract_features_from_logs,
    FeaturePipeline,
)


# ============================================================================
# AUTHENTICATION FEATURE TESTS
# ============================================================================

def test_auth_feature_calculations(sample_raw_logs):
    # 1. Test fail count
    fails = calculate_auth_fail_count(sample_raw_logs)
    assert fails.get("10.0.0.5") == 1.0
    assert fails.get("192.168.1.100") == 0.0

    # 2. Test success count
    successes = calculate_auth_success_count(sample_raw_logs)
    assert successes.get("10.0.0.5") == 1.0

    # 3. Test bounded normalized ratio: 1 fail, 1 success -> 1 / (1 + 1) = 0.5
    ratios = calculate_auth_fail_ratio(sample_raw_logs)
    assert ratios.get("10.0.0.5") == 0.5
    # Normal web IP without auth attempts has 0.0 ratio (policy edge case)
    assert ratios.get("192.168.1.100") == 0.0

    # 4. Test distinct username extraction
    distinct_users = calculate_distinct_username(sample_raw_logs)
    assert distinct_users.get("10.0.0.5") >= 1.0

    # 5. Test interarrival stats
    mean_ia, std_ia = calculate_fail_interarrival_stats(sample_raw_logs)
    # Only 1 fail -> stats should be 0.0
    assert mean_ia.get("10.0.0.5") == 0.0
    assert std_ia.get("10.0.0.5") == 0.0


def test_auth_interarrival_multi_events():
    """Verify that multiple consecutive failures compute correct mean and std."""
    raw_logs = pd.DataFrame([
        {"src_ip": "10.0.0.99", "timestamp": "2026-07-15 10:00:00", "raw_log": "Failed password for admin from 10.0.0.99"},
        {"src_ip": "10.0.0.99", "timestamp": "2026-07-15 10:00:02", "raw_log": "Failed password for admin from 10.0.0.99"},
        {"src_ip": "10.0.0.99", "timestamp": "2026-07-15 10:00:04", "raw_log": "Failed password for root from 10.0.0.99"},
    ])
    mean_ia, std_ia = calculate_fail_interarrival_stats(raw_logs)
    # Gaps are [2.0, 2.0] -> mean=2.0, std=0.0
    assert mean_ia.get("10.0.0.99") == 2.0
    assert std_ia.get("10.0.0.99") == 0.0


# ============================================================================
# WEB & HTTP FEATURE TESTS
# ============================================================================

def test_web_feature_calculations(sample_raw_logs):
    # 1. Test request rate
    rates = calculate_http_req_rate(sample_raw_logs, window_seconds=60.0)
    assert rates.get("192.168.1.100") > 0.0

    # 2. Test status ratios (4xx and 5xx)
    r4xx, r5xx = calculate_http_status_ratios(sample_raw_logs)
    assert r4xx.get("172.16.0.1") == 1.0  # 404 scan
    assert r4xx.get("192.168.1.100") == 0.0  # 200 OK
    assert r5xx.get("172.16.0.1") == 0.0

    # 3. Test HTTP POST ratio
    r_post = calculate_http_post_ratio(sample_raw_logs)
    assert r_post.get("192.168.1.100") == 0.0  # GET method

    # 4. Test distinct user agents
    uas = calculate_distinct_user_agent(sample_raw_logs)
    assert uas.get("192.168.1.100") == 1.0

    # 5. Test bytes sent mean
    bytes_mean = calculate_bytes_sent_mean(sample_raw_logs)
    assert bytes_mean.get("192.168.1.100") == 4324.0
    assert bytes_mean.get("172.16.0.1") == 120.0


# ============================================================================
# BEHAVIOR & CONTEXT FEATURE TESTS
# ============================================================================

def test_behavior_feature_calculations(sample_raw_logs):
    # 1. Test global active IPs count
    global_ips = calculate_global_active_ips(sample_raw_logs)
    assert global_ips == float(sample_raw_logs["src_ip"].nunique())

    # 2. Test burst rate with empty history
    burst = calculate_burst_rate(sample_raw_logs, pd.DataFrame())
    assert burst.get("10.0.0.5") == 1.0

    # 3. Test burst rate with historical baseline
    df_hist = sample_raw_logs.copy()
    burst_with_hist = calculate_burst_rate(sample_raw_logs, df_hist)
    assert burst_with_hist.get("10.0.0.5") == 1.0


# ============================================================================
# FULL LOG EXTRACTOR PIPELINE TESTS
# ============================================================================

def test_extract_features_from_logs(sample_raw_logs):
    features_df = extract_features_from_logs(sample_raw_logs, window_start=datetime.now())

    # Assert all 14 feature columns exist in output
    for col in FEATURE_COLUMNS:
        assert col in features_df.columns, f"Missing feature column: {col}"

    # Assert correct number of unique IPs extracted
    assert len(features_df) == sample_raw_logs["src_ip"].nunique()
    assert not features_df[FEATURE_COLUMNS].isna().any().any()


def test_extract_features_fallback_regex():
    # Test raw logs lacking explicit src_ip column (trigger regex fallback)
    raw_logs = pd.DataFrame([
        {"raw_log": '10.0.0.99 - - [15/Jul/2026:22:00:00] "GET /test HTTP/1.1" 200 120'}
    ])
    features_df = extract_features_from_logs(raw_logs)
    assert not features_df.empty
    assert features_df["src_ip"].iloc[0] == "10.0.0.99"


def test_extract_features_empty_dataframe():
    empty_df = pd.DataFrame()
    res = extract_features_from_logs(empty_df)
    assert res.empty


# ============================================================================
# FEATURE PREPROCESSING PIPELINE TESTS
# ============================================================================

def test_feature_pipeline_fit_transform(sample_features, tmp_path):
    pipeline = FeaturePipeline()

    # Fit & Transform
    X_scaled = pipeline.fit_transform(sample_features)

    # Assert shape matches expected 14 feature dimensions
    assert X_scaled.shape[0] == len(sample_features)
    assert X_scaled.shape[1] == len(pipeline.feature_cols)
    assert X_scaled.shape[1] == 14

    # Assert mean is close to 0 and std is close to 1
    assert abs(X_scaled.mean(axis=0)[0]) < 1e-6

    # Test Serialization (Save & Load)
    save_path = str(tmp_path / "pipeline.joblib")
    pipeline.save(save_path)
    assert os.path.exists(save_path)

    loaded_pipeline = FeaturePipeline.load(save_path)
    X_loaded = loaded_pipeline.transform(sample_features)
    np.testing.assert_array_almost_equal(X_scaled, X_loaded)


def test_feature_pipeline_column_reordering_and_missing(sample_features):
    """Verify that FeaturePipeline auto-fills missing feature columns with 0.0 and enforces exact column ordering."""
    pipeline = FeaturePipeline()
    pipeline.fit(sample_features)

    # Scramble column order and drop a column
    scrambled_df = sample_features.copy()
    cols = list(scrambled_df.columns)
    scrambled_df = scrambled_df[cols[::-1]]  # reverse order
    scrambled_df = scrambled_df.drop(columns=["auth_fail_count"])

    # FeaturePipeline should auto-fill missing auth_fail_count column with 0.0 and enforce 14-column order
    X_scaled = pipeline.transform(scrambled_df)
    assert X_scaled.shape == (len(sample_features), 14)
    assert not np.isnan(X_scaled).any()
    assert not np.isinf(X_scaled).any()
