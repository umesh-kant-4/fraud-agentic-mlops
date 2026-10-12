"""Fraud-specific evaluation metrics (pure numpy/sklearn, unit-tested).

AUC alone is a poor release criterion for fraud. What the business runs on is
"given the alerts our analysts can review, how much fraud (and how many
dollars of fraud) do we catch, and how many good customers do we bother?"
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def top_k_mask(scores: np.ndarray, alert_rate: float) -> np.ndarray:
    """Boolean mask of the top `alert_rate` fraction of scores (the alert queue)."""
    k = max(1, int(round(len(scores) * alert_rate)))
    idx = np.argsort(-scores, kind="stable")[:k]
    mask = np.zeros(len(scores), dtype=bool)
    mask[idx] = True
    return mask


def fraud_metrics(y: np.ndarray, scores: np.ndarray, amount: np.ndarray,
                  alert_rate: float = 0.01) -> dict:
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    amount = np.asarray(amount, dtype=float)
    alerts = top_k_mask(scores, alert_rate)
    tp = int((alerts & (y == 1)).sum())
    fp = int((alerts & (y == 0)).sum())
    pos = int(y.sum())
    fraud_dollars = float(amount[y == 1].sum())
    caught_dollars = float(amount[alerts & (y == 1)].sum())
    return {
        "aucpr": float(average_precision_score(y, scores)) if pos else 0.0,
        "roc_auc": float(roc_auc_score(y, scores)) if 0 < pos < len(y) else 0.0,
        "alert_rate": alert_rate,
        "precision_at_alert_rate": tp / max(1, tp + fp),
        "recall_at_alert_rate": tp / max(1, pos),
        "dollar_recall_at_alert_rate": caught_dollars / fraud_dollars if fraud_dollars else 0.0,
        "false_positives_per_1k_txn": 1000 * fp / max(1, len(y)),
        "n_test": int(len(y)),
        "n_fraud_test": pos,
    }


# Release policy: a challenger must not regress the champion beyond these
# tolerances on the SAME test window. Keep this in code so both the pipeline
# and the agent tools apply identical rules.
RELEASE_POLICY = {
    "aucpr": {"min_delta": -0.005},
    "recall_at_alert_rate": {"min_delta": -0.01},
    "dollar_recall_at_alert_rate": {"min_delta": -0.01},
    "false_positives_per_1k_txn": {"max_delta": 0.5},
}


def compare_to_champion(challenger: dict, champion: dict | None,
                        policy: dict = RELEASE_POLICY) -> dict:
    """Return per-metric deltas and an overall pass/fail with reasons."""
    if not champion:
        return {"passes": True, "reasons": ["no champion: first model"], "deltas": {}}
    deltas, reasons = {}, []
    for metric, rule in policy.items():
        if metric not in challenger or metric not in champion:
            reasons.append(f"{metric}: missing")
            continue
        d = challenger[metric] - champion[metric]
        deltas[metric] = round(d, 6)
        if "min_delta" in rule and d < rule["min_delta"]:
            reasons.append(f"{metric} regressed by {d:+.4f} (limit {rule['min_delta']:+})")
        if "max_delta" in rule and d > rule["max_delta"]:
            reasons.append(f"{metric} worsened by {d:+.4f} (limit +{rule['max_delta']})")
    return {"passes": not reasons, "reasons": reasons or ["within release policy"], "deltas": deltas}
