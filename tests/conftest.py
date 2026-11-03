import pytest
import pandas as pd
from datetime import datetime, timedelta

@pytest.fixture
def sample_raw_logs():
    """Returns a mock DataFrame of raw web and authentication logs."""
    data = [
        # Normal web log
        {
            "src_ip": "192.168.1.100",
            "raw_message": '192.168.1.100 - - [15/Jul/2026:22:00:00 +0700] "GET /index.html HTTP/1.1" 200 4324',
            "parsed_json": {"status": "200", "url": "/index.html", "method": "GET", "user_agent": "Mozilla/5.0"}
        },
        # Nginx 404 scan log
        {
            "src_ip": "172.16.0.1",
            "raw_message": '172.16.0.1 - - [15/Jul/2026:22:00:01 +0700] "GET /admin/config.php HTTP/1.1" 404 120',
            "parsed_json": {"status": "404", "url": "/admin/config.php", "method": "GET", "user_agent": "Gobuster/v3.0"}
        },
        # SSH failed login log
        {
            "src_ip": "10.0.0.5",
            "raw_message": "sshd[1234]: Failed password for invalid user admin from 10.0.0.5 port 54322 ssh2",
            "parsed_json": {"user": "admin"}
        },
        # SSH success log
        {
            "src_ip": "10.0.0.5",
            "raw_message": "sshd[1234]: Accepted password for root from 10.0.0.5 port 54322 ssh2",
            "parsed_json": {"user": "root"}
        }
    ]
    return pd.DataFrame(data)

@pytest.fixture
def sample_features():
    """Returns a mock feature vectors DataFrame (14 features)."""
    base_time = datetime.now()
    data = []
    
    # 5 Normal vectors
    for i in range(5):
        data.append({
            "window_start": base_time - timedelta(minutes=i),
            "src_ip": f"192.168.1.{10 + i}",
            "auth_fail_count": 0.0,
            "auth_success_count": 10.0,
            "auth_fail_ratio": 0.0,
            "distinct_username": 1.0,
            "fail_interarrival_mean": 0.0,
            "fail_interarrival_std": 0.0,
            "http_req_rate": 2.5,
            "http_4xx_ratio": 0.02,
            "http_5xx_ratio": 0.0,
            "http_post_ratio": 0.1,
            "distinct_user_agent": 1.0,
            "bytes_sent_mean": 4324.0,
            "global_active_ips": 5.0,
            "burst_rate": 1.0,
            "label": "Normal",
            "label_source": "baseline_normal",
            "confidence": 1.0
        })
        
    # 2 Anomaly vectors (Brute force & Scanning profiles)
    data.append({
        "window_start": base_time,
        "src_ip": "172.16.0.99",
        "auth_fail_count": 85.0,  # high fail
        "auth_success_count": 0.0,
        "auth_fail_ratio": 1.0,
        "distinct_username": 45.0,  # password spraying
        "fail_interarrival_mean": 0.7,
        "fail_interarrival_std": 0.02,  # automated tool regular timing
        "http_req_rate": 0.0,
        "http_4xx_ratio": 0.0,
        "http_5xx_ratio": 0.0,
        "http_post_ratio": 0.0,
        "distinct_user_agent": 0.0,
        "bytes_sent_mean": 0.0,
        "global_active_ips": 6.0,
        "burst_rate": 15.0,  # high burst
        "label": "Anomaly",
        "label_source": "lab_scenario",
        "confidence": 1.0
    })
    
    data.append({
        "window_start": base_time,
        "src_ip": "172.16.0.100",
        "auth_fail_count": 0.0,
        "auth_success_count": 0.0,
        "auth_fail_ratio": 0.0,
        "distinct_username": 0.0,
        "fail_interarrival_mean": 0.0,
        "fail_interarrival_std": 0.0,
        "http_req_rate": 120.0,  # high rate
        "http_4xx_ratio": 0.9,  # high 404 scan
        "http_5xx_ratio": 0.05,
        "http_post_ratio": 0.8,  # web brute force post
        "distinct_user_agent": 5.0,  # cycling UAs
        "bytes_sent_mean": 120.0,
        "global_active_ips": 6.0,
        "burst_rate": 8.0,
        "label": "Anomaly",
        "label_source": "lab_scenario",
        "confidence": 1.0
    })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_feature_df(sample_features):
    """Returns a mock feature vectors DataFrame (14 features)."""
    return sample_features

