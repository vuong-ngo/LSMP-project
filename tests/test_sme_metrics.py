import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from lsmp_ai.evaluation.metrics import (
    basic_metrics,
    ranking_metrics,
    bayesian_detection_rate,
    bdr_table,
    alarms_per_hour,
    cid,
    bootstrap_ci,
    pick_threshold,
    recall_at_k,
    full_report,
    rule_only_score,
)
from lsmp_ai.evaluation.comparison import ModelComparator
from lsmp_ai.evaluation.ablation import (
    run_feature_group_ablation,
    run_hyperparameter_sensitivity,
    run_gate_rate_sweep,
)


def test_basic_metrics():
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0])
    m = basic_metrics(y_true, y_pred)
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 2
    assert "precision" in m
    assert "recall" in m
    assert "f1" in m
    assert "f2" in m
    assert "fpr" in m
    assert "tnr" in m


def test_ranking_and_bdr_metrics():
    y_true = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.4, 0.8, 0.9])
    rm = ranking_metrics(y_true, score)
    assert 0 <= rm["roc_auc"] <= 1.0
    assert 0 <= rm["pr_auc"] <= 1.0
    assert "pr_auc_over_prevalence" in rm

    bdr = bayesian_detection_rate(tpr=0.9, fpr=0.01, base_rate=0.01)
    assert 0 <= bdr <= 1.0

    b_table = bdr_table(tpr=0.9, fpr=0.01)
    assert "bdr@0.1" in b_table
    assert "bdr@0.01" in b_table
    assert "bdr@0.001" in b_table

    c = cid(tpr=0.9, fpr=0.01, base_rate=0.01)
    assert 0 <= c <= 1.0

    aph = alarms_per_hour(fpr=0.01, tpr=0.9, base_rate=0.01)
    assert aph > 0


def test_pick_threshold_and_full_report():
    y_val = np.array([0, 0, 0, 0, 1, 1, 1])
    s_val = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9])
    thr = pick_threshold(y_val, s_val, criterion="f2")
    assert 0.1 <= thr <= 0.9

    thr_budget = pick_threshold(y_val, s_val, criterion="fpr_budget", max_fpr=0.1)
    assert 0.1 <= thr_budget <= 0.9

    rep = full_report(y_val, s_val, threshold=thr)
    assert "precision" in rep
    assert "recall" in rep
    assert "bdr@0.01" in rep
    assert "cid@0.01" in rep
    assert "alarms_per_hour@0.01" in rep
    assert "recall@10" in rep
    assert "pr_auc_ci" in rep
    assert "classification_ci" in rep
    assert "f1" in rep["classification_ci"]
    assert rep["n_samples"] == len(y_val)


def test_rule_only_score():
    df_auth = pd.DataFrame({"auth_fail_count": [0, 5, 10]})
    r = rule_only_score(df_auth)
    assert len(r) == 3
    assert r[2] == 10.0


def test_model_comparator_generates_sme_artifacts(sample_feature_df, tmp_path):
    from lsmp_ai.common.constants import FEATURE_COLUMNS
    df = sample_feature_df
    X = df[FEATURE_COLUMNS].values
    y = np.where(df["label"] == "Anomaly", 1, 0)

    n = len(df)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    X_train, y_train = X[:i1], y[:i1]
    X_val, y_val = X[i1:i2], y[i1:i2]
    X_test, y_test = X[i2:], y[i2:]

    comparator = ModelComparator(benchmark_runs=2, random_state=42)
    df_comp, full_details = comparator.compare(
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        include_rule_only=True,
    )

    assert "metrics_json" in full_details
    assert "table_baselines" in full_details
    assert "table_bdr" in full_details
    assert "table_topk" in full_details

    m_json = full_details["metrics_json"]
    assert "cascade" in m_json
    assert "iforest_only" in m_json
    assert "ocsvm_only" in m_json
    assert "rule_only" in m_json
    assert "alert_reduction" in m_json


def test_ablation_and_sensitivity_functions(sample_feature_df, tmp_path):
    df = sample_feature_df
    # 1. Feature group ablation
    df_abl = run_feature_group_ablation(df, output_dir=tmp_path)
    assert not df_abl.empty
    assert "full_14_features" in df_abl["config"].values
    assert (tmp_path / "table_ablation.csv").exists()

    # 2. Hyperparameter sensitivity
    df_sens = run_hyperparameter_sensitivity(df, output_dir=tmp_path)
    assert not df_sens.empty
    assert "contamination" in df_sens["param"].values
    assert (tmp_path / "table_sensitivity.csv").exists()

    # 3. Gate rate sweep
    df_gate = run_gate_rate_sweep(df, output_dir=tmp_path)
    assert not df_gate.empty
    assert "gate_rate" in df_gate.columns
    assert (tmp_path / "table_gate.csv").exists()
