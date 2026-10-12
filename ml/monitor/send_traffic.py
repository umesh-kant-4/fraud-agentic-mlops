"""Replay transactions against the prod endpoint so data capture has something to monitor.

  python ml/monitor/send_traffic.py --csv data/live.csv --rows 3000
  python ml/monitor/send_traffic.py --csv data/drifted.csv --rows 3000   # trigger drift
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from preprocess import FEATURES  # noqa: E402

PREFIX = os.environ.get("FRAUD_SSM_PREFIX", "/fraud-mlops")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rows", type=int, default=3000)
    # one row per request keeps captured records 1:1 with transactions,
    # which is what the data-quality monitor expects
    ap.add_argument("--batch", type=int, default=1, help="rows per request")
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    endpoint = boto3.client("ssm").get_parameter(Name=f"{PREFIX}/endpoint-name")["Parameter"]["Value"]
    rt = boto3.client("sagemaker-runtime")
    df = pd.read_csv(args.csv).head(args.rows)[FEATURES]
    sent = 0
    for start in range(0, len(df), args.batch):
        body = df.iloc[start:start + args.batch].to_csv(index=False, header=False)
        rt.invoke_endpoint(EndpointName=endpoint, ContentType="text/csv", Body=body)
        sent += min(args.batch, len(df) - start)
        time.sleep(args.sleep)
    print(f"sent {sent} transactions to {endpoint}")


if __name__ == "__main__":
    main()
