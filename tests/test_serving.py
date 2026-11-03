import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from lsmp_ai.serving.inference_service import InferenceService
from lsmp_ai.serving.scheduler import InferenceScheduler
from lsmp_ai.common.constants import FEATURE_COLUMNS
from lsmp_ai.models.cascade_model import CascadeModel


@pytest.fixture
def mock_db_client():
    client = MagicMock()
    return client


@pytest.fixture
def mock_result_writer():
    writer = MagicMock()
    writer.write_predictions.return_value = 5
    return writer


@pytest.fixture
def mock_model_registry():
    registry = MagicMock()
    cascade = CascadeModel()
    X = np.random.randn(10, len(FEATURE_COLUMNS))
    cascade.fit(X)
    registry.load_latest.return_value = cascade
    registry.load_version.return_value = cascade
    return registry


@pytest.fixture
def sample_feature_df():
    data = {col: np.random.randn(5) for col in FEATURE_COLUMNS}
    data["src_ip"] = [f"192.168.1.{i+1}" for i in range(5)]
    data["window_start"] = pd.date_range("2026-07-23 10:00:00", periods=5, freq="1min")
    return pd.DataFrame(data)


def test_inference_service_load_and_infer(mock_db_client, mock_result_writer, mock_model_registry, sample_feature_df):
    from lsmp_ai.feature_engineering.feature_pipeline import FeaturePipeline
    service = InferenceService(
        db_client=mock_db_client,
        result_writer=mock_result_writer,
        model_registry=mock_model_registry,
    )
    
    # 1. Test infer before model loaded raises RuntimeError
    with pytest.raises(RuntimeError):
        service.infer_on_features(sample_feature_df)
        
    # 2. Test model loading
    service.load_model()
    assert service._model is not None
    mock_model_registry.load_latest.assert_called_once()
    
    # 3. Test missing feature pipeline raises RuntimeError (H4 protection)
    with pytest.raises(RuntimeError, match="FeaturePipeline is missing"):
        service.infer_on_features(sample_feature_df)

    # 4. Attach valid feature pipeline and test infer_on_features
    service._feat_pipeline = FeaturePipeline().fit(sample_feature_df)
    results_df = service.infer_on_features(sample_feature_df)
    assert len(results_df) == 5
    assert "anomaly_score" in results_df.columns
    assert "risk_score" in results_df.columns
    assert "risk_class" in results_df.columns


def test_inference_service_run_inference(mock_db_client, mock_result_writer, mock_model_registry, sample_feature_df):
    from lsmp_ai.feature_engineering.feature_pipeline import FeaturePipeline
    mock_db_client.fetch_feature_vectors.return_value = sample_feature_df
    
    service = InferenceService(
        db_client=mock_db_client,
        result_writer=mock_result_writer,
        model_registry=mock_model_registry,
    )
    service.load_model()
    service._feat_pipeline = FeaturePipeline().fit(sample_feature_df)
    
    written_count = service.run_inference(limit=50)
    mock_db_client.fetch_feature_vectors.assert_called_with(limit=50)
    mock_result_writer.write_predictions.assert_called_once()
    assert written_count == 5


def test_inference_service_missing_features_raises_error(mock_db_client, mock_result_writer, mock_model_registry):
    service = InferenceService(
        db_client=mock_db_client,
        result_writer=mock_result_writer,
        model_registry=mock_model_registry,
    )
    service.load_model()
    
    bad_df = pd.DataFrame({"src_ip": ["1.1.1.1"]})
    with pytest.raises(ValueError, match="Missing feature columns"):
        service.infer_on_features(bad_df)


def test_inference_scheduler_start_stop():
    mock_service = MagicMock()
    mock_service.run_inference.return_value = 3
    
    scheduler = InferenceScheduler(
        inference_service=mock_service,
        poll_interval_seconds=0.1,
        batch_size=10
    )
    
    with patch("time.sleep", side_effect=InterruptedError("Stop loop")):
        with pytest.raises(InterruptedError):
            scheduler.start()
            
    mock_service.run_inference.assert_called_with(limit=10)


def test_fastapi_writer_service_endpoints():
    try:
        from fastapi.testclient import TestClient
        from lsmp_ai.serving.lsmp_writer_service import app
    except ImportError:
        pytest.skip("fastapi or httpx is not installed in the environment.")
    
    client = TestClient(app)
    
    # 1. Health check (unauthenticated public endpoint - S5: no reconnaissance data leaked)
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert "model_version" not in json_data
    assert "model_loaded" not in json_data
    assert "database_configured" not in json_data
    
    payload = {
        "start_time": "2026-07-23T10:00:00",
        "end_time": "2026-07-23T10:05:00"
    }

    # 2. Fail-Closed: LSMP_API_KEY is not set and LSMP_ALLOW_INSECURE_DEV=0 -> 500 error
    with patch.dict(os.environ, {"LSMP_API_KEY": "", "LSMP_ALLOW_INSECURE_DEV": "0"}):
        resp_unconfigured = client.post("/process", json=payload)
        assert resp_unconfigured.status_code == 500
        assert "misconfiguration" in resp_unconfigured.json()["detail"].lower()

    # 3. Dev mode bypass: LSMP_ALLOW_INSECURE_DEV=1 allows execution without API key
    with patch.dict(os.environ, {"LSMP_API_KEY": "", "LSMP_ALLOW_INSECURE_DEV": "1"}):
        with patch("lsmp_ai.serving.lsmp_writer_service.run_serve_pipeline"):
            resp_dev = client.post("/process", json=payload)
            assert resp_dev.status_code == 200
            assert resp_dev.json()["status"] == "queued"

    # 4. Strict authentication with configured key:
    test_key = "secure_secret_api_key_456"
    with patch.dict(os.environ, {"LSMP_API_KEY": test_key, "LSMP_ALLOW_INSECURE_DEV": "0"}):
        # 4a. Missing API key header -> 401
        resp_missing = client.post("/process", json=payload)
        assert resp_missing.status_code == 401

        # 4b. Invalid API key header -> 403
        resp_invalid = client.post("/process", json=payload, headers={"X-API-Key": "wrong_key"})
        assert resp_invalid.status_code == 403

        # 4c. Valid API key header -> 200 queued
        with patch("lsmp_ai.serving.lsmp_writer_service.run_serve_pipeline"):
            resp_valid = client.post("/process", json=payload, headers={"X-API-Key": test_key})
            assert resp_valid.status_code == 200
            assert resp_valid.json()["status"] == "queued"


def test_rate_limiter_memory_safety_and_enforcement():
    from lsmp_ai.serving.lsmp_writer_service import RateLimiter
    # 1. Enforce max requests per IP
    limiter = RateLimiter(max_requests=2, window_seconds=10, max_tracked_ips=5)
    assert limiter.is_allowed("192.168.1.10") is True
    assert limiter.is_allowed("192.168.1.10") is True
    assert limiter.is_allowed("192.168.1.10") is False

    # 2. Memory bound check: does not grow unbounded with spoofed IPs (S4 defense)
    for i in range(20):
        limiter.is_allowed(f"10.0.0.{i}")
    assert len(limiter._requests) <= 5


