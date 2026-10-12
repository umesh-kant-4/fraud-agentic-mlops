"""SageMaker Processing step: score challenger (and champion) on the same test window.

Runs inside the SageMaker XGBoost container (has xgboost, pandas, sklearn, boto3).
Writes /opt/ml/processing/evaluation/evaluation.json which the pipeline's
ConditionStep reads and which is attached to the registered model package.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
from urllib.parse import urlparse

sys.path.insert(0, "/opt/ml/processing/lib")

import boto3  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from fraud_metrics import compare_to_champion, fraud_metrics  # noqa: E402
from preprocess import FEATURES, LABEL  # noqa: E402

BASE = "/opt/ml/processing"


def load_booster(tar_path: str) -> xgb.Booster:
    d = tempfile.mkdtemp()
    with tarfile.open(tar_path) as t:
        t.extractall(d)  # trusted: our own training output
    booster = xgb.Booster()
    booster.load_model(os.path.join(d, "xgboost-model"))
    return booster


def download_s3(uri: str) -> str:
    u = urlparse(uri)
    local = os.path.join(tempfile.mkdtemp(), "champion.tar.gz")
    boto3.client("s3").download_file(u.netloc, u.path.lstrip("/"), local)
    return local


def score(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(xgb.DMatrix(X.values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion-model-uri", default="")
    ap.add_argument("--alert-rate", type=float, default=0.01)
    args = ap.parse_args()

    test = pd.read_csv(f"{BASE}/test/test.csv")
    X, y, amount = test[FEATURES], test[LABEL].values, test["amount"].values

    challenger = fraud_metrics(y, score(load_booster(f"{BASE}/model/model.tar.gz"), X),
                               amount, args.alert_rate)
    champion = None
    if args.champion_model_uri.startswith("s3://"):
        champion = fraud_metrics(y, score(load_booster(download_s3(args.champion_model_uri)), X),
                                 amount, args.alert_rate)
    comparison = compare_to_champion(challenger, champion)

    report = {
        "metrics": challenger,
        "champion": champion,
        "comparison": {**comparison, "passes_int": int(comparison["passes"])},
        # SageMaker Model Registry renders this block in Studio
        "binary_classification_metrics": {
            "auc": {"value": challenger["roc_auc"], "standard_deviation": "NaN"},
            "aucpr": {"value": challenger["aucpr"], "standard_deviation": "NaN"},
            "recall": {"value": challenger["recall_at_alert_rate"], "standard_deviation": "NaN"},
            "precision": {"value": challenger["precision_at_alert_rate"], "standard_deviation": "NaN"},
        },
    }
    os.makedirs(f"{BASE}/evaluation", exist_ok=True)
    with open(f"{BASE}/evaluation/evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
