"""SageMaker Processing step: time-based split + XGBoost-ready CSVs.

Fraud data must be split by time, not randomly: a random split leaks
future fraud patterns into training and overstates performance.

Outputs (under /opt/ml/processing/):
  train/train.csv            label first, no header  (XGBoost built-in format)
  validation/validation.csv  label first, no header
  test/test.csv              header, features + label + amount (for evaluation)
  baseline/baseline.csv      no header, features only (Model Monitor baseline)
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

FEATURES = [
    "amount", "hour_of_day", "merchant_category", "distance_from_home_km",
    "txn_count_1h", "txn_count_24h", "avg_amount_30d", "amount_to_avg_ratio",
    "is_foreign", "card_present", "account_age_days", "new_device",
]
LABEL = "is_fraud"
BASE = "/opt/ml/processing"


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    df = df.sort_values("event_time").reset_index(drop=True)
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    files = glob.glob(f"{BASE}/input/*.csv")
    if not files:
        raise SystemExit("no CSV files in input channel")
    df = pd.concat((pd.read_csv(f, parse_dates=["event_time"]) for f in files), ignore_index=True)
    missing = set(FEATURES + [LABEL]) - set(df.columns)
    if missing:
        raise SystemExit(f"input is missing columns: {sorted(missing)}")
    df = df.dropna(subset=FEATURES + [LABEL])

    train, val, test = time_split(df, args.train_frac, args.val_frac)
    for name, part in (("train", train), ("validation", val)):
        os.makedirs(f"{BASE}/{name}", exist_ok=True)
        part[[LABEL, *FEATURES]].to_csv(f"{BASE}/{name}/{name}.csv", index=False, header=False)
    os.makedirs(f"{BASE}/test", exist_ok=True)
    test[[*FEATURES, LABEL]].to_csv(f"{BASE}/test/test.csv", index=False)
    os.makedirs(f"{BASE}/baseline", exist_ok=True)
    # header-less, like captured endpoint traffic, so Model Monitor names both _c0.._cN
    train[FEATURES].sample(min(len(train), 50_000), random_state=0).to_csv(
        f"{BASE}/baseline/baseline.csv", index=False, header=False)

    print({"train": len(train), "validation": len(val), "test": len(test),
           "fraud_rate_train": round(train[LABEL].mean(), 5),
           "fraud_rate_test": round(test[LABEL].mean(), 5)})


if __name__ == "__main__":
    main()
