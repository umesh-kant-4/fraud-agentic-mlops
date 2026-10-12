import unittest

import numpy as np

import _stubs  # noqa: F401
from fraud_metrics import compare_to_champion, fraud_metrics, top_k_mask
from generate_synthetic import FEATURES as GEN_FEATURES
from generate_synthetic import LABEL, generate
from preprocess import FEATURES, time_split


class TestMetrics(unittest.TestCase):
    def test_top_k_mask_selects_highest(self):
        m = top_k_mask(np.array([0.1, 0.9, 0.5, 0.8]), 0.5)
        self.assertEqual(m.tolist(), [False, True, False, True])

    def test_perfect_model(self):
        y = np.array([0] * 98 + [1] * 2)
        scores = y.astype(float)
        amount = np.ones(100) * 10
        m = fraud_metrics(y, scores, amount, alert_rate=0.02)
        self.assertAlmostEqual(m["aucpr"], 1.0)
        self.assertEqual(m["recall_at_alert_rate"], 1.0)
        self.assertEqual(m["precision_at_alert_rate"], 1.0)
        self.assertEqual(m["false_positives_per_1k_txn"], 0.0)

    def test_dollar_recall_weights_amounts(self):
        y = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.1, 0.2, 0.3])  # catches the $900 fraud only
        amount = np.array([900.0, 100.0, 5.0, 5.0])
        m = fraud_metrics(y, scores, amount, alert_rate=0.25)
        self.assertAlmostEqual(m["dollar_recall_at_alert_rate"], 0.9)
        self.assertAlmostEqual(m["recall_at_alert_rate"], 0.5)

    def test_compare_first_model_passes(self):
        self.assertTrue(compare_to_champion({"aucpr": 0.5}, None)["passes"])

    def test_compare_flags_regression(self):
        champ = {"aucpr": 0.60, "recall_at_alert_rate": 0.70,
                 "dollar_recall_at_alert_rate": 0.80, "false_positives_per_1k_txn": 5.0}
        worse = {**champ, "aucpr": 0.55, "false_positives_per_1k_txn": 6.0}
        out = compare_to_champion(worse, champ)
        self.assertFalse(out["passes"])
        self.assertEqual(len(out["reasons"]), 2)
        better = {**champ, "aucpr": 0.62}
        self.assertTrue(compare_to_champion(better, champ)["passes"])


class TestData(unittest.TestCase):
    def test_generator_schema_matches_pipeline(self):
        self.assertEqual(GEN_FEATURES, FEATURES)
        df = generate(5000, seed=1)
        self.assertEqual(len(df), 5000)
        self.assertTrue(set(FEATURES + [LABEL, "event_time"]).issubset(df.columns))
        self.assertFalse(df[FEATURES].isna().any().any())
        self.assertTrue(0.003 < df[LABEL].mean() < 0.02)

    def test_drift_changes_fraud_pattern(self):
        base = generate(60000, fraud_rate=0.02, seed=2)
        drift = generate(60000, fraud_rate=0.02, drift=True, seed=2)
        fb, fd = base[base[LABEL] == 1], drift[drift[LABEL] == 1]
        self.assertLess(fd["card_present"].mean(), fb["card_present"].mean() - 0.2)
        self.assertGreater(fd["is_foreign"].mean(), fb["is_foreign"].mean() + 0.2)
        self.assertLess(fd["amount"].median(), fb["amount"].median())

    def test_time_split_is_chronological(self):
        df = generate(1000, seed=3)
        train, val, test = time_split(df.sample(frac=1, random_state=0), 0.7, 0.15)
        self.assertEqual((len(train), len(val), len(test)), (700, 150, 150))
        self.assertLessEqual(train["event_time"].max(), val["event_time"].min())
        self.assertLessEqual(val["event_time"].max(), test["event_time"].min())

    def test_model_learns_signal(self):
        """Sanity: the synthetic data is learnable (sklearn stand-in for XGBoost)."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        df = generate(30000, fraud_rate=0.02, seed=4)
        train, _, test = time_split(df, 0.7, 0.0)
        clf = HistGradientBoostingClassifier(max_iter=100).fit(train[FEATURES], train[LABEL])
        m = fraud_metrics(test[LABEL].values, clf.predict_proba(test[FEATURES])[:, 1],
                          test["amount"].values, alert_rate=0.03)
        self.assertGreater(m["aucpr"], 0.5)


if __name__ == "__main__":
    unittest.main()
